"""Probability calibration for 3-way (win/draw/loss) forecasts.

A model can rank matches well yet be mis-calibrated -- e.g. systematically
over-confident, saying 80% when the true rate is 70%. Calibration corrects the
*magnitude* of probabilities without changing their order, so that a stated 70%
actually happens ~70% of the time. Good calibration is essential for a
forecasting product: the tournament simulation multiplies these probabilities
across many rounds, so small biases compound.

We use **temperature scaling** -- a single scalar T applied in log-space:

    p_calibrated ∝ p ** (1 / T)      (then renormalised)

T > 1 softens over-confident predictions toward uniform; T < 1 sharpens them.
One parameter is robust to fit on the limited number of matches in a single
tournament, unlike per-class methods that can overfit. T is chosen to minimise
log loss on a held-out set via a simple line search.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import log_loss_3way, ranked_probability_score


def apply_temperature(probs: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    """Return temperature-scaled, renormalised probabilities."""
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    logits = np.log(p) / T
    logits -= logits.max(axis=1, keepdims=True)   # stability
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, outcomes: np.ndarray,
                    grid: np.ndarray | None = None) -> float:
    """Find the temperature T minimising log loss on (probs, outcomes)."""
    if grid is None:
        grid = np.linspace(0.5, 3.0, 51)
    best_T, best_ll = 1.0, np.inf
    for T in grid:
        ll = log_loss_3way(apply_temperature(probs, T), outcomes)
        if ll < best_ll:
            best_ll, best_T = ll, float(T)
    return best_T


class TemperatureCalibrator:
    """Fit once on a validation tournament, then transform future predictions."""

    def __init__(self):
        self.T: float = 1.0

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "TemperatureCalibrator":
        self.T = fit_temperature(probs, outcomes)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return apply_temperature(probs, self.T)

    def report(self, probs: np.ndarray, outcomes: np.ndarray) -> dict:
        """Before/after RPS + log loss, to show the calibration effect."""
        cal = self.transform(probs)
        return {
            "T": self.T,
            "rps_before": ranked_probability_score(probs, outcomes),
            "rps_after": ranked_probability_score(cal, outcomes),
            "logloss_before": log_loss_3way(probs, outcomes),
            "logloss_after": log_loss_3way(cal, outcomes),
        }
