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


def _normalize_names(log: pd.DataFrame) -> pd.DataFrame:
    """Map alternate team spellings to the dataset's canonical names so the same
    nation never appears under two labels (e.g. 'Cape Verde Islands')."""
    try:
        from src.data.live_results import TEAM_NAME_MAP
    except Exception:
        return log
    log = log.copy()
    for col in ("home_team", "away_team"):
        log[col] = log[col].astype(str).map(lambda n: TEAM_NAME_MAP.get(n, n))
    return log


def _dedup(log: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate fixtures arising when two data sources name or date
    the same game differently (e.g. 'Cape Verde' vs 'Cape Verde Islands', or a
    one-day skew). Keyed by the unordered team pair within +/-1 day; the most
    recently logged row is kept.
    """
    if log.empty:
        return log
    log = _normalize_names(log)
    log = log.sort_values("date").reset_index(drop=True)
    a = log["home_team"].astype(str)
    b = log["away_team"].astype(str)
    pair = (a.where(a < b, b) + "|" + a.where(a >= b, b))
    d = pd.to_datetime(log["date"], errors="coerce")

    keep, seen = [], []
    for i in range(len(log)):
        dup = any(pair.iloc[i] == p and abs((d.iloc[i] - dt).days) <= 1
                  for p, dt in seen)
        if not dup:
            keep.append(i)
            seen.append((pair.iloc[i], d.iloc[i]))
    return log.iloc[keep].reset_index(drop=True)


def _refresh_actuals(log: pd.DataFrame, wc: pd.DataFrame) -> pd.DataFrame:
    """Refresh the *actual* result of every logged game from current data.

    Predictions (p_home/draw/away, pred_*) stay locked -- they were made before
    kickoff and must never change. But the actual score can be provisional or
    corrected upstream (e.g. a penalty score briefly logged as goals, or a late
    data fix), so we always re-read home_goals/away_goals from the latest data
    and recompute the realised outcome, hit flag, and RPS. Matching is on the
    unordered team pair within +/-1 day to tolerate cross-source date skew.
    """
    if log.empty or wc.empty:
        return log

    # Index current results by sorted team pair -> (date, hg, ag), keeping the
    # most recent record per pair.
    cur = {}
    for m in wc.sort_values("date").itertuples(index=False):
        key = tuple(sorted([str(m.home_team), str(m.away_team)]))
        cur[key] = (pd.Timestamp(m.date), str(m.home_team),
                    int(m.home_goals), int(m.away_goals))

    log = log.copy()
    changed = False
    for idx in log.index:
        h, a = str(log.at[idx, "home_team"]), str(log.at[idx, "away_team"])
        rec = cur.get(tuple(sorted([h, a])))
        if rec is None:
            continue
        _, src_home, shg, sag = rec
        # Orient the source score to this row's (home, away) order.
        hg, ag = (shg, sag) if src_home == h else (sag, shg)
        if (int(log.at[idx, "home_goals"]) == hg
                and int(log.at[idx, "away_goals"]) == ag):
            continue                          # already correct
        act_out = _outcome(hg, ag)
        probs = np.array([[float(log.at[idx, "p_home"]),
                           float(log.at[idx, "p_draw"]),
                           float(log.at[idx, "p_away"])]])
        log.at[idx, "home_goals"] = hg
        log.at[idx, "away_goals"] = ag
        log.at[idx, "actual_outcome"] = act_out
        log.at[idx, "correct"] = int(int(log.at[idx, "pred_outcome"]) == act_out)
        log.at[idx, "rps"] = ranked_probability_score(probs, np.array([act_out]))
        changed = True
    log.attrs["_changed"] = changed
    return log


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
    # Drop any previously-logged fixtures from before the go-live date, and
    # collapse cross-source duplicates, so the record stays clean.
    if not log.empty:
        before = len(log)
        log = log[log["date"] >= PREDICTIONS_START].reset_index(drop=True)
        log = _dedup(log)
        # Refresh actual results from current data (corrects stale/provisional
        # scores; predictions stay locked).
        log = _refresh_actuals(log, wc)
        if len(log) != before or log.attrs.get("_changed"):
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
        log = _coerce(log)
        # Collapse any cross-source duplicate that the new rows introduced
        # (same tie logged under a slightly different date by another source).
        log = _dedup(log).sort_values("date").reset_index(drop=True)
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
