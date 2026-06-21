"""Model track record: log every 2026 group-stage prediction and score it
against the actual result, updating incrementally day by day.

Traceability is about accountability: to judge the model we record what it
predicted for each fixture *before* that fixture was played, then compare to
what actually happened. We reconstruct each prediction with a strict
**walk-forward** rule -- for every matchday we train only on data dated strictly
*before* that day, so no result ever informs its own prediction (zero
look-ahead). Once a fixture is logged its prediction is **locked**; subsequent
runs only append newly completed fixtures. This makes the log a faithful,
immutable record that grows as the tournament unfolds.

Stored columns (public/prediction_log.csv):
    date, home_team, away_team,
    p_home, p_draw, p_away,        # pre-match model probabilities
    pred_outcome,                  # argmax: 0 home / 1 draw / 2 away
    pred_home_goals, pred_away_goals,   # most-likely scoreline
    home_goals, away_goals,        # actual result
    actual_outcome, correct, rps   # scoring vs the actual outcome
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.gbm import GBMModel
from src.ratings.bayesian import _is_2026_wc
from src.evaluation.metrics import ranked_probability_score, log_loss_3way

LOG_COLUMNS = [
    "date", "home_team", "away_team",
    "p_home", "p_draw", "p_away", "pred_outcome",
    "pred_home_goals", "pred_away_goals",
    "home_goals", "away_goals", "actual_outcome", "correct", "rps",
]

OUTCOME_LABELS = {0: "Home win", 1: "Draw", 2: "Away win"}

# The model's public track record begins on this date. Fixtures played before
# this are excluded from the log and from all reported statistics, so the
# metrics reflect only predictions made from the model's go-live date onward.
PREDICTIONS_START = pd.Timestamp("2026-06-18")


def _outcome(hg: int, ag: int) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns have proper dtypes (concat with an empty
    object-dtype frame can otherwise leave them as object)."""
    float_cols = ["p_home", "p_draw", "p_away", "rps"]
    int_cols = ["pred_outcome", "pred_home_goals", "pred_away_goals",
                "home_goals", "away_goals", "actual_outcome", "correct"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def load_log(path: Path) -> pd.DataFrame:
    """Load an existing prediction log, or an empty one with the right schema."""
    if Path(path).exists():
        df = pd.read_csv(path, parse_dates=["date"])
        for c in LOG_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        return _coerce(df[LOG_COLUMNS])
    return pd.DataFrame(columns=LOG_COLUMNS)


def _key(df: pd.DataFrame) -> set:
    return {
        (pd.Timestamp(d).strftime("%Y-%m-%d"), h, a)
        for d, h, a in zip(df["date"], df["home_team"], df["away_team"])
    }


def update_log(matches: pd.DataFrame, features: pd.DataFrame,
               log_path: Path) -> pd.DataFrame:
    """Append predictions for any newly-completed 2026 group matches.

    Walk-forward: matches are processed in date order and each matchday's model
    is trained only on data strictly before that date.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    wc = matches[_is_2026_wc(matches)].copy()
    # Only track fixtures from the model's go-live date onward.
    wc = wc[wc["date"] >= PREDICTIONS_START]
    if wc.empty:
        return load_log(log_path)

    log = load_log(log_path)
    # Drop any previously-logged fixtures from before the go-live date so the
    # record (and its stats) start cleanly on PREDICTIONS_START.
    if not log.empty:
        before = len(log)
        log = log[log["date"] >= PREDICTIONS_START].reset_index(drop=True)
        if len(log) != before:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            log.to_csv(log_path, index=False)
    have = _key(log)

    # Fixtures not yet logged.
    todo = [m for m in wc.itertuples(index=False)
            if (pd.Timestamp(m.date).strftime("%Y-%m-%d"),
                m.home_team, m.away_team) not in have]
    if not todo:
        return log

    new_rows = []
    # Group the outstanding fixtures by date and train one model per matchday.
    by_date: dict[pd.Timestamp, list] = {}
    for m in todo:
        by_date.setdefault(pd.Timestamp(m.date), []).append(m)

    for cutoff in sorted(by_date):
        train_m = matches[matches["date"] < cutoff]
        train_f = features[features["date"] < cutoff]
        if train_m.empty:
            continue
        model = GBMModel().fit(train_f, train_m)
        known = set(model.teams)

        for m in by_date[cutoff]:
            if m.home_team not in known or m.away_team not in known:
                continue
            p_home, p_draw, p_away = model.predict_match(
                m.home_team, m.away_team, neutral=True)
            mat = model.score_matrix(m.home_team, m.away_team, neutral=True)
            i, j = np.unravel_index(int(np.argmax(mat)), mat.shape)
            probs = np.array([[p_home, p_draw, p_away]])
            pred_out = int(np.argmax(probs))
            act_out = _outcome(int(m.home_goals), int(m.away_goals))
            new_rows.append({
                "date": pd.Timestamp(m.date),
                "home_team": m.home_team, "away_team": m.away_team,
                "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
                "pred_outcome": pred_out,
                "pred_home_goals": int(i), "pred_away_goals": int(j),
                "home_goals": int(m.home_goals), "away_goals": int(m.away_goals),
                "actual_outcome": act_out,
                "correct": int(pred_out == act_out),
                "rps": ranked_probability_score(probs, np.array([act_out])),
            })

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        log = _coerce(log).sort_values("date").reset_index(drop=True)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(log_path, index=False)
    return log


def summarize(log: pd.DataFrame) -> dict:
    """Aggregate accuracy statistics over all logged predictions."""
    if log.empty:
        return {"n": 0}
    probs = log[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    actual = log["actual_outcome"].to_numpy(dtype=int)

    # Base-rate benchmark = the empirical outcome split of the logged matches.
    base = np.bincount(actual, minlength=3) / len(actual)
    base_probs = np.tile(base, (len(actual), 1))

    return {
        "n": int(len(log)),
        "hit_rate": float(log["correct"].mean()),
        "avg_rps": float(log["rps"].mean()),
        "base_rps": ranked_probability_score(base_probs, actual),
        "log_loss": log_loss_3way(probs, actual),
        "exact_score": float(
            ((log["pred_home_goals"] == log["home_goals"])
             & (log["pred_away_goals"] == log["away_goals"])).mean()),
    }
