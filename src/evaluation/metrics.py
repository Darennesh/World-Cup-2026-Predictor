"""Evaluation metrics for probabilistic football forecasting.

The headline metric is the Ranked Probability Score (RPS), the standard for
football outcome forecasting because it rewards getting the *ordered*
win/draw/loss distribution right (a prediction that puts mass on "draw" when
the truth is "away win" is penalised less than one that backs "home win").
"""
from __future__ import annotations

import numpy as np


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean RPS over a set of ordered 3-way predictions. Lower is better.

    Parameters
    ----------
    probs : array shape (n, 3)
        Predicted probabilities ordered [home_win, draw, away_win]. Each row
        should sum to 1.
    outcomes : array shape (n,)
        Realised outcome index: 0 = home_win, 1 = draw, 2 = away_win.

    Returns
    -------
    float
        Average RPS across all matches (0 = perfect, ~0.5 worst case).
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n, k = probs.shape

    # One-hot the realised outcomes.
    actual = np.zeros((n, k))
    actual[np.arange(n), outcomes] = 1.0

    cum_pred = np.cumsum(probs, axis=1)
    cum_act = np.cumsum(actual, axis=1)
    # RPS = (1/(k-1)) * sum over categories of (cumPred - cumAct)^2
    rps = np.sum((cum_pred - cum_act) ** 2, axis=1) / (k - 1)
    return float(np.mean(rps))


def log_loss_3way(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Multiclass log loss for win/draw/loss predictions. Lower is better."""
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    outcomes = np.asarray(outcomes, dtype=int)
    picked = probs[np.arange(len(outcomes)), outcomes]
    return float(-np.mean(np.log(picked)))


def calibration_table(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Reliability data: for each confidence bin, predicted vs observed freq.

    Flattens all (match, class) probability/outcome pairs and bins them, so you
    can plot calibration: when the model says ~70%, does it happen ~70%?
    Returns (bin_centers, predicted_mean, observed_freq, counts).
    """
    probs = np.asarray(probs, dtype=float).ravel()
    n, k = np.asarray(outcomes).shape[0], 3
    actual = np.zeros((n, k))
    actual[np.arange(n), np.asarray(outcomes, dtype=int)] = 1.0
    actual = actual.ravel()

    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)

    centers, pred_mean, obs_freq, counts = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        centers.append((bins[b] + bins[b + 1]) / 2)
        pred_mean.append(probs[mask].mean())
        obs_freq.append(actual[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(pred_mean), np.array(obs_freq), np.array(counts)
