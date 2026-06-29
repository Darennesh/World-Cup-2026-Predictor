"""Tests for the Negative Binomial goal model."""
import numpy as np
import pandas as pd

from src.models.negbinom import NegativeBinomialModel, _nb_pmf
from src.models.dixon_coles import DixonColes


def _synthetic(n=600, seed=0, overdisperse=False):
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
        if overdisperse:
            # Inject extra variance via a gamma-mixed Poisson (i.e. NB).
            hg = rng.negative_binomial(3, 3 / (3 + lam_h))
            ag = rng.negative_binomial(3, 3 / (3 + lam_a))
        else:
            hg, ag = rng.poisson(lam_h), rng.poisson(lam_a)
        rows.append({"date": date, "home_team": h, "away_team": a,
                     "home_goals": int(hg), "away_goals": int(ag),
                     "neutral": False, "importance": 1.0})
    return pd.DataFrame(rows)


def test_nb_pmf_sums_to_one():
    k = np.arange(60)
    p = _nb_pmf(k, mu=1.5, r=10.0)
    assert abs(p.sum() - 1.0) < 1e-4


def test_nb_pmf_approaches_poisson_for_large_r():
    from scipy.stats import poisson
    k = np.arange(30)
    nb = _nb_pmf(k, mu=1.3, r=1e6)
    po = poisson.pmf(k, 1.3)
    assert np.max(np.abs(nb - po)) < 1e-3


def test_probabilities_sum_to_one():
    m = NegativeBinomialModel().fit(_synthetic())
    p = m.predict_match("Strong", "Weak", neutral=True)
    assert abs(sum(p) - 1.0) < 1e-6


def test_score_matrix_normalised():
    m = NegativeBinomialModel().fit(_synthetic())
    mat = m.score_matrix("Mid", "Mid")
    assert abs(mat.sum() - 1.0) < 1e-9


def test_stronger_team_favoured():
    m = NegativeBinomialModel().fit(_synthetic())
    p_home, _, p_away = m.predict_match("Strong", "Weak", neutral=True)
    assert p_home > p_away


def test_detects_overdispersion():
    # Over-dispersed data should yield a finite, moderate dispersion r; clean
    # Poisson data should push r toward the large (Poisson) limit.
    over = NegativeBinomialModel().fit(_synthetic(overdisperse=True))
    clean = NegativeBinomialModel().fit(_synthetic(overdisperse=False))
    assert over.r < clean.r


def test_compatible_with_simulation_interface():
    from src.simulation.sampler import ScoreSampler
    m = NegativeBinomialModel().fit(_synthetic())
    sampler = ScoreSampler(m)
    rng = np.random.default_rng(0)
    hg, ag = sampler.sample("Strong", "Weak", rng)
    assert hg >= 0 and ag >= 0
