"""Tests for Elo 3-way conversion and parameter tuning."""
import numpy as np
import pandas as pd

from src.ratings.elo_tuning import elo_3way, EloParams, evaluate_elo, tune_elo


def test_elo_3way_sums_to_one():
    for dr in (-400, -100, 0, 100, 400):
        p = elo_3way(dr, d0=0.26, sigma=250)
        assert abs(sum(p) - 1.0) < 1e-9
        assert all(x >= 0 for x in p)


def test_even_match_has_highest_draw():
    p_even = elo_3way(0, 0.26, 250)
    p_lop = elo_3way(400, 0.26, 250)
    assert p_even[1] > p_lop[1]          # draw mass peaks when evenly matched


def test_stronger_team_favoured():
    p_home, _, p_away = elo_3way(200, 0.26, 250)
    assert p_home > p_away


def _matches(n=400, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 1.0, "B": 0.3, "C": -0.3, "D": -1.0}
    rows = []
    date = pd.Timestamp("2018-01-01")
    for _ in range(n):
        date += pd.Timedelta(days=3)
        h, a = rng.choice(teams, 2, replace=False)
        lam_h = np.exp(0.3 + strength[h] - strength[a])
        lam_a = np.exp(0.3 + strength[a] - strength[h])
        hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
        rows.append({"date": date, "home_team": h, "away_team": a,
                     "home_goals": hg, "away_goals": ag, "neutral": False,
                     "importance": 1.0,
                     "outcome": 0 if hg > ag else (1 if hg == ag else 2)})
    return pd.DataFrame(rows)


def test_evaluate_returns_valid_rps():
    m = _matches()
    start = m["date"].quantile(0.6)
    rps, n = evaluate_elo(m, EloParams(40, 65, 0.26, 250), start)
    assert 0.0 < rps < 0.5
    assert n > 0


def test_tune_finds_a_config_at_least_as_good_as_default():
    m = _matches()
    start = m["date"].quantile(0.6)
    default_rps, _ = evaluate_elo(m, EloParams(40, 65, 0.26, 250), start)
    best, best_rps, results = tune_elo(
        m, start, k_grid=(30, 40), home_grid=(40, 65),
        d0_grid=(0.26,), sigma_grid=(250,))
    assert best_rps <= default_rps + 1e-9     # tuning never worse than a member
    assert len(results) == 4
