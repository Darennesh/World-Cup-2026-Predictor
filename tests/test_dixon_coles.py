"""Tests for the Dixon-Coles baseline model."""
import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColes, _tau


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["Strong", "Mid", "Weak"]
    strength = {"Strong": 0.6, "Mid": 0.0, "Weak": -0.6}
    rows = []
    date = pd.Timestamp("2021-01-01")
    for _ in range(n):
        date += pd.Timedelta(days=3)
        h, a = rng.choice(teams, 2, replace=False)
        lam_h = np.exp(0.2 + 0.25 + strength[h] - strength[a])
        lam_a = np.exp(0.2 + strength[a] - strength[h])
        rows.append({"date": date, "home_team": h, "away_team": a,
                     "home_goals": int(rng.poisson(lam_h)),
                     "away_goals": int(rng.poisson(lam_a)),
                     "neutral": False, "importance": 1.0})
    return pd.DataFrame(rows)


def test_probabilities_sum_to_one():
    m = DixonColes().fit(_synthetic())
    p = m.predict_match("Strong", "Weak", neutral=True)
    assert abs(sum(p) - 1.0) < 1e-6


def test_stronger_team_favoured():
    m = DixonColes().fit(_synthetic())
    p_home, _, p_away = m.predict_match("Strong", "Weak", neutral=True)
    assert p_home > p_away


def test_ratings_recover_ordering():
    m = DixonColes().fit(_synthetic())
    table = m.ratings_table().set_index("team")
    assert table.loc["Strong", "attack"] > table.loc["Weak", "attack"]


def test_score_matrix_normalised():
    m = DixonColes().fit(_synthetic())
    mat = m.score_matrix("Mid", "Mid")
    assert abs(mat.sum() - 1.0) < 1e-9


def test_tau_unaffected_for_high_scores():
    out = _tau(np.array(3), np.array(2), np.array(1.5), np.array(1.2), -0.05)
    assert out.item() == 1.0
