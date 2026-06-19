"""Sampling scorelines from a match model.

The simulation needs to *sample* concrete scorelines, not just read win/draw/
loss probabilities, because group standings depend on goals (goal difference,
goals for) and knockout ties need a winner. Any model exposing a
`score_matrix(home, away, neutral) -> 2D array of P(i, j)` can be plugged in
here, so the same engine works for the Dixon-Coles baseline today and the
LightGBM model later.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class ScoreModel(Protocol):
    """Anything that can produce a joint score-probability matrix."""
    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        ...


class ScoreSampler:
    """Caches each pairing's score distribution and samples scorelines fast.

    For speed we precompute, per pairing, the flattened score-probability
    vector's cumulative sum once, then sample with a single uniform draw +
    binary search (`searchsorted`). This is markedly faster than
    `rng.choice(p=...)` when the same pairings are sampled millions of times
    across a Monte Carlo run.
    """

    def __init__(self, model: ScoreModel):
        self.model = model
        # key -> (cumulative_probs, n_cols, p_home_win_2way)
        self._cache: dict[tuple[str, str, bool], tuple[np.ndarray, int, float]] = {}

    def _prep(self, home: str, away: str, neutral: bool):
        key = (home, away, neutral)
        cached = self._cache.get(key)
        if cached is None:
            mat = self.model.score_matrix(home, away, neutral)
            mat = mat / mat.sum()
            flat = mat.ravel()
            cum = np.cumsum(flat)
            cum[-1] = 1.0  # guard against fp drift so searchsorted is safe
            p_home = float(np.tril(mat, -1).sum() + 0.5 * np.trace(mat))
            cached = (cum, mat.shape[1], p_home)
            self._cache[key] = cached
        return cached

    def sample(self, home: str, away: str, rng: np.random.Generator,
               neutral: bool = True) -> tuple[int, int]:
        """Sample a single (home_goals, away_goals) scoreline."""
        cum, ncols, _ = self._prep(home, away, neutral)
        idx = int(np.searchsorted(cum, rng.random()))
        return idx // ncols, idx % ncols

    def win_probability(self, home: str, away: str, neutral: bool = True) -> float:
        """P(home wins) with draws split evenly -- weights the shootout flip."""
        _, _, p_home = self._prep(home, away, neutral)
        return p_home
