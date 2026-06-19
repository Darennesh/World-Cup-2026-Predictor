"""Tests for probability calibration and back-testing."""
import numpy as np
import pandas as pd

from src.evaluation.calibration import (apply_temperature, fit_temperature,
                                        TemperatureCalibrator)
from src.evaluation.backtest import wc_edition


def test_apply_temperature_keeps_simplex():
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    out = apply_temperature(probs, T=1.5)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= 0).all()


def test_temperature_one_is_identity():
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    out = apply_temperature(probs, T=1.0)
    assert np.allclose(out, probs)


def test_higher_temperature_softens():
    probs = np.array([[0.8, 0.15, 0.05]])
    softened = apply_temperature(probs, T=2.5)
    # Max probability should decrease toward uniform.
    assert softened.max() < probs.max()


def test_fit_temperature_improves_overconfident():
    # Construct over-confident predictions: always 0.9 on the truth's neighbour.
    rng = np.random.default_rng(0)
    n = 500
    outcomes = rng.integers(0, 3, size=n)
    probs = np.full((n, 3), 0.05)
    probs[np.arange(n), outcomes] = 0.9
    # Corrupt 40% so confidence is unjustified.
    flip = rng.random(n) < 0.4
    for i in np.where(flip)[0]:
        wrong = (outcomes[i] + 1) % 3
        probs[i] = 0.05
        probs[i, wrong] = 0.9
    T = fit_temperature(probs, outcomes)
    assert T > 1.0   # should soften over-confidence


def test_calibrator_report_keys():
    rng = np.random.default_rng(1)
    n = 200
    outcomes = rng.integers(0, 3, size=n)
    probs = rng.dirichlet([2, 2, 2], size=n)
    rep = TemperatureCalibrator().fit(probs, outcomes).report(probs, outcomes)
    assert set(rep) >= {"T", "rps_before", "rps_after",
                        "logloss_before", "logloss_after"}


def test_wc_edition_filters_year_and_finals():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2018-06-14", "2017-09-01", "2022-11-20"]),
        "home_team": ["Russia", "A", "Qatar"],
        "away_team": ["Saudi Arabia", "B", "Ecuador"],
        "home_goals": [5, 1, 0], "away_goals": [0, 0, 2],
        "tournament": ["FIFA World Cup", "FIFA World Cup qualification",
                       "FIFA World Cup"],
        "neutral": [False, False, True], "outcome": [0, 0, 2],
        "importance": [1.0, 0.7, 1.0],
    })
    e2018 = wc_edition(df, 2018)
    assert len(e2018) == 1 and e2018.iloc[0]["home_team"] == "Russia"
