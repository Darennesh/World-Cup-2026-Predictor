"""Tests for the Dixon-Coles + LightGBM ensemble model."""
import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColes
from src.models.gbm import GBMModel
from src.models.ensemble import EnsembleModel, tune_weight


def _data(n=700, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["Strong", "Mid", "Weak"]
    strength = {"Strong": 1.0, "Mid": 0.0, "Weak": -1.0}
    m_rows, f_rows = [], []
    date = pd.Timestamp("2021-01-01")
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
            "elo_home": 1500 + 100 * strength[h], "elo_away": 1500 + 100 * strength[a],
            "elo_diff": 100 * (strength[h] - strength[a]) + 65,
            "home_gf": strength[h] + 1, "home_ga": 1.0,
            "away_gf": strength[a] + 1, "away_ga": 1.0,
            "form_diff": strength[h] - strength[a],
            "home_rest": 5, "away_rest": 5, "neutral": 0, "importance": 1.0,
            "home_goals": hg, "away_goals": ag,
            "outcome": 0 if hg > ag else (1 if hg == ag else 2)})
    return pd.DataFrame(m_rows), pd.DataFrame(f_rows)


def _fit():
    m, f = _data()
    dc = DixonColes().fit(m)
    gbm = GBMModel(n_estimators=60).fit(f, m)
    return dc, gbm, m


def test_score_matrix_normalised():
    dc, gbm, _ = _fit()
    ens = EnsembleModel(dc=dc, gbm=gbm, weight=0.5)
    mat = ens.score_matrix("Strong", "Weak", neutral=True)
    assert abs(mat.sum() - 1.0) < 1e-9


def test_probabilities_sum_to_one():
    dc, gbm, _ = _fit()
    ens = EnsembleModel(dc=dc, gbm=gbm, weight=0.4)
    p = ens.predict_match("Strong", "Weak")
    assert abs(sum(p) - 1.0) < 1e-6


def test_weight_extremes_match_base_models():
    dc, gbm, _ = _fit()
    # weight=0 -> pure Dixon-Coles; weight=1 -> pure LightGBM.
    ens0 = EnsembleModel(dc=dc, gbm=gbm, weight=0.0)
    ens1 = EnsembleModel(dc=dc, gbm=gbm, weight=1.0)
    np.testing.assert_allclose(ens0.predict_match("Strong", "Weak"),
                               dc.predict_match("Strong", "Weak"), atol=1e-6)
    np.testing.assert_allclose(ens1.predict_match("Strong", "Weak"),
                               gbm.predict_match("Strong", "Weak"), atol=1e-6)


def test_stronger_team_favoured():
    dc, gbm, _ = _fit()
    ens = EnsembleModel(dc=dc, gbm=gbm, weight=0.5)
    p_home, _, p_away = ens.predict_match("Strong", "Weak", neutral=True)
    assert p_home > p_away


def test_tune_weight_returns_valid_weight():
    dc, gbm, m = _fit()
    m = m.copy()
    m["outcome"] = [0 if h > a else (1 if h == a else 2)
                    for h, a in zip(m["home_goals"], m["away_goals"])]
    w, rps = tune_weight(dc, gbm, m)
    assert 0.0 <= w <= 1.0
    assert rps >= 0.0


def test_compatible_with_simulation_interface():
    from src.simulation.sampler import ScoreSampler
    dc, gbm, _ = _fit()
    ens = EnsembleModel(dc=dc, gbm=gbm, weight=0.5)
    sampler = ScoreSampler(ens)
    rng = np.random.default_rng(0)
    hg, ag = sampler.sample("Strong", "Weak", rng)
    assert hg >= 0 and ag >= 0
