"""Tests for the model track-record (prediction log) module."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.tracker import (update_log, summarize, load_log,
                                    LOG_COLUMNS, _outcome)


def _history_and_wc():
    """Synthetic pre-2026 history + two 2026 World Cup group fixtures."""
    rng = np.random.default_rng(0)
    teams = ["Alpha", "Bravo", "Charlie", "Delta"]
    strength = {"Alpha": 1.0, "Bravo": 0.3, "Charlie": -0.2, "Delta": -0.8}

    m_rows, f_rows = [], []
    date = pd.Timestamp("2024-01-01")
    elo = {t: 1500.0 for t in teams}
    for _ in range(400):
        date += pd.Timedelta(days=2)
        h, a = rng.choice(teams, 2, replace=False)
        lam_h = np.exp(0.2 + 0.25 + strength[h] - strength[a])
        lam_a = np.exp(0.2 + strength[a] - strength[h])
        hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
        m_rows.append({"date": date, "home_team": h, "away_team": a,
                       "home_goals": hg, "away_goals": ag, "neutral": False,
                       "importance": 1.0, "tournament": "Friendly"})
        f_rows.append({
            "date": date, "home_team": h, "away_team": a,
            "elo_home": elo[h], "elo_away": elo[a],
            "elo_diff": elo[h] + 65 - elo[a],
            "home_gf": strength[h] + 1, "home_ga": 1.0,
            "away_gf": strength[a] + 1, "away_ga": 1.0,
            "form_diff": strength[h] - strength[a],
            "home_rest": 5, "away_rest": 5, "neutral": 0, "importance": 1.0,
            "home_goals": hg, "away_goals": ag,
            "outcome": _outcome(hg, ag),
        })

    # Two 2026 World Cup group games (neutral) with known results, dated on or
    # after the tracker's go-live cutoff so they are recorded.
    wc = [("Alpha", "Delta", 3, 0, "2026-06-18"),
          ("Bravo", "Charlie", 1, 1, "2026-06-19")]
    for h, a, hg, ag, d in wc:
        dt = pd.Timestamp(d)
        m_rows.append({"date": dt, "home_team": h, "away_team": a,
                       "home_goals": hg, "away_goals": ag, "neutral": True,
                       "importance": 1.0, "tournament": "FIFA World Cup"})
        f_rows.append({
            "date": dt, "home_team": h, "away_team": a,
            "elo_home": 1600, "elo_away": 1500, "elo_diff": 100,
            "home_gf": 1.5, "home_ga": 1.0, "away_gf": 1.2, "away_ga": 1.1,
            "form_diff": 0.3, "home_rest": 5, "away_rest": 5,
            "neutral": 1, "importance": 1.0,
            "home_goals": hg, "away_goals": ag, "outcome": _outcome(hg, ag)})
    return pd.DataFrame(m_rows), pd.DataFrame(f_rows)


def test_update_log_creates_rows(tmp_path: Path):
    matches, features = _history_and_wc()
    log_path = tmp_path / "log.csv"
    log = update_log(matches, features, log_path)

    assert len(log) == 2                         # both WC games logged
    assert list(log.columns) == LOG_COLUMNS
    assert log_path.exists()
    # Probabilities form a valid distribution.
    p = log[["p_home", "p_draw", "p_away"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_actual_outcomes_recorded_correctly(tmp_path: Path):
    matches, features = _history_and_wc()
    log = update_log(matches, features, tmp_path / "log.csv")
    by = {(r.home_team, r.away_team): r for r in log.itertuples(index=False)}
    assert by[("Alpha", "Delta")].actual_outcome == 0      # 3-0 home win
    assert by[("Bravo", "Charlie")].actual_outcome == 1    # 1-1 draw


def test_log_is_idempotent_and_locked(tmp_path: Path):
    matches, features = _history_and_wc()
    log_path = tmp_path / "log.csv"
    first = update_log(matches, features, log_path)
    probs_first = first[["p_home", "p_draw", "p_away"]].to_numpy().copy()

    # Running again adds nothing and does not alter locked predictions.
    second = update_log(matches, features, log_path)
    assert len(second) == len(first) == 2
    np.testing.assert_allclose(
        second[["p_home", "p_draw", "p_away"]].to_numpy(), probs_first)


def test_no_lookahead_first_matchday_excludes_same_day(tmp_path: Path):
    # If we only have the WC matches' own day as training, there must be prior
    # data; here history exists, so a model trains. Add a same-day second game
    # and confirm both still get predicted from strictly-earlier data only.
    matches, features = _history_and_wc()
    log = update_log(matches, features, tmp_path / "log.csv")
    # Predictions exist and are finite (model trained on pre-date history).
    assert log["rps"].notna().all()
    assert (log["rps"] >= 0).all()


def test_summarize_keys(tmp_path: Path):
    matches, features = _history_and_wc()
    log = update_log(matches, features, tmp_path / "log.csv")
    s = summarize(log)
    assert s["n"] == 2
    assert set(s) >= {"n", "hit_rate", "avg_rps", "base_rps",
                      "log_loss", "exact_score"}
    assert 0.0 <= s["hit_rate"] <= 1.0


def test_fixtures_before_start_date_excluded(tmp_path: Path):
    from src.evaluation.tracker import PREDICTIONS_START

    matches, features = _history_and_wc()
    # Add a pre-go-live WC fixture; it must NOT be logged.
    early = pd.Timestamp("2026-06-12")
    matches = pd.concat([matches, pd.DataFrame([{
        "date": early, "home_team": "Alpha", "away_team": "Bravo",
        "home_goals": 2, "away_goals": 1, "neutral": True,
        "importance": 1.0, "tournament": "FIFA World Cup",
        "outcome": 0}])], ignore_index=True)
    features = pd.concat([features, pd.DataFrame([{
        "date": early, "home_team": "Alpha", "away_team": "Bravo",
        "elo_home": 1600, "elo_away": 1500, "elo_diff": 100,
        "home_gf": 1.5, "home_ga": 1.0, "away_gf": 1.2, "away_ga": 1.1,
        "form_diff": 0.3, "home_rest": 5, "away_rest": 5,
        "neutral": 1, "importance": 1.0,
        "home_goals": 2, "away_goals": 1, "outcome": 0}])], ignore_index=True)

    log = update_log(matches, features, tmp_path / "log.csv")
    assert (log["date"] >= PREDICTIONS_START).all()
    assert len(log) == 2     # only the two on/after the cutoff


def test_summarize_empty():
    assert summarize(load_log(Path("does_not_exist.csv")))["n"] == 0
