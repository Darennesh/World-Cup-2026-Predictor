"""Tests for the LightGBM challenger model."""
import numpy as np
import pandas as pd

from src.models.gbm import GBMModel, FEATURES


def _data(n=800, seed=0):
    """Synthetic matches + matching feature rows with a learnable signal."""
    rng = np.random.default_rng(seed)
    teams = ["Strong", "Mid", "Weak"]
    strength = {"Strong": 1.0, "Mid": 0.0, "Weak": -1.0}
    m_rows, f_rows = [], []
    date = pd.Timestamp("2021-01-01")
    elo = {t: 1500.0 for t in teams}
    for _ in range(n):
        date += pd.Timedelta(days=3)
        h, a = rng.choice(teams, 2, replace=False)
        lam_h = np.exp(0.2 + 0.25 + strength[h] - strength[a])
        lam_a = np.exp(0.2 + strength[a] - strength[h])
        hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
        m_rows.append({"date": date, "home_team": h, "away_team": a,
                       "home_goals": hg, "away_goals": ag,
                       "neutral": False, "importance": 1.0})
        f_rows.append({
            "date": date, "home_team": h, "away_team": a,
            "elo_home": elo[h], "elo_away": elo[a],
            "elo_diff": elo[h] + 65 - elo[a],
            "home_gf": strength[h] + 1, "home_ga": 1.0,
            "away_gf": strength[a] + 1, "away_ga": 1.0,
            "form_diff": strength[h] - strength[a],
            "home_rest": 5, "away_rest": 5, "neutral": 0, "importance": 1.0,
            "home_goals": hg, "away_goals": ag,
            "outcome": 0 if hg > ag else (1 if hg == ag else 2),
        })
    return pd.DataFrame(m_rows), pd.DataFrame(f_rows)


def test_score_matrix_normalised():
    m, f = _data()
    model = GBMModel(n_estimators=60).fit(f, m)
    mat = model.score_matrix("Strong", "Weak", neutral=True)
    assert abs(mat.sum() - 1.0) < 1e-9


def test_probabilities_sum_to_one():
    m, f = _data()
    model = GBMModel(n_estimators=60).fit(f, m)
    p = model.predict_match("Strong", "Weak")
    assert abs(sum(p) - 1.0) < 1e-6


def test_stronger_team_favoured():
    m, f = _data()
    model = GBMModel(n_estimators=80).fit(f, m)
    p_home, _, p_away = model.predict_match("Strong", "Weak", neutral=True)
    assert p_home > p_away


def test_expected_goals_positive_and_clipped():
    m, f = _data()
    model = GBMModel(n_estimators=60).fit(f, m)
    lam_h, lam_a = model.expected_goals("Strong", "Weak")
    assert 0 < lam_h <= model.max_lambda
    assert 0 < lam_a <= model.max_lambda


def test_compatible_with_simulation_interface():
    # The engine only needs score_matrix(home, away, neutral).
    from src.simulation.sampler import ScoreSampler
    m, f = _data()
    model = GBMModel(n_estimators=60).fit(f, m)
    sampler = ScoreSampler(model)
    rng = np.random.default_rng(0)
    hg, ag = sampler.sample("Strong", "Weak", rng)
    assert hg >= 0 and ag >= 0
