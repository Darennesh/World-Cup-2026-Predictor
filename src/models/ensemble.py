"""Ensemble match model: a weighted blend of Dixon-Coles and LightGBM.

The two base models make partly *uncorrelated* errors -- Dixon-Coles is an
additive statistical model with a principled low-score correction, while
LightGBM captures non-linear feature interactions. Averaging their predicted
score matrices therefore tends to lower RPS below either model alone, the
classic ensemble benefit.

The blend weight `w` (probability mass on LightGBM, 1-w on Dixon-Coles) is a
single parameter tuned on held-out data to minimise RPS -- robust to fit even
with limited international matches, unlike many-parameter stacking. The ensemble
exposes the same `score_matrix(home, away, neutral)` interface as the base
models, so it drops straight into the simulation engine and every metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.dixon_coles import DixonColes
from src.models.gbm import GBMModel


@dataclass
class EnsembleModel:
    dc: DixonColes
    gbm: GBMModel
    weight: float = 0.5          # mass on GBM; (1 - weight) on Dixon-Coles
    max_goals: int = 10

    @property
    def teams(self) -> list[str]:
        return sorted(set(self.dc.teams) & set(self.gbm.teams))

    def can_predict(self, home: str, away: str) -> bool:
        known = set(self.dc.teams) & set(self.gbm.teams)
        return home in known and away in known

    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """Weighted average of the two base models' score matrices.

        Matrices are aligned to a common (max_goals+1) shape before blending so
        a difference in either model's truncation never breaks the mix.
        """
        m_dc = self._aligned(self.dc.score_matrix(home, away, neutral))
        m_gbm = self._aligned(self.gbm.score_matrix(home, away, neutral))
        mat = (1.0 - self.weight) * m_dc + self.weight * m_gbm
        return mat / mat.sum()

    def _aligned(self, mat: np.ndarray) -> np.ndarray:
        n = self.max_goals + 1
        out = np.zeros((n, n))
        r = min(n, mat.shape[0])
        c = min(n, mat.shape[1])
        out[:r, :c] = mat[:r, :c]
        s = out.sum()
        return out / s if s > 0 else out

    def predict_match(self, home: str, away: str, neutral: bool = True):
        mat = self.score_matrix(home, away, neutral)
        p_home = float(np.tril(mat, -1).sum())
        p_draw = float(np.trace(mat))
        p_away = float(np.triu(mat, 1).sum())
        return p_home, p_draw, p_away

    def predict_matches(self, pairs) -> np.ndarray:
        """Vectorised win/draw/loss probabilities for many fixtures (n, 3)."""
        out = np.empty((len(pairs), 3))
        for i, p in enumerate(pairs.itertuples(index=False)):
            out[i] = self.predict_match(p.home_team, p.away_team,
                                        neutral=bool(p.neutral))
        return out


def tune_weight(dc: DixonColes, gbm: GBMModel, val,
                grid: np.ndarray | None = None) -> tuple[float, float]:
    """Choose the blend weight minimising RPS on a validation frame.

    `val` needs columns home_team, away_team, neutral, outcome. Returns
    (best_weight, best_rps). Only fixtures both base models can predict are used.
    """
    from src.evaluation.metrics import ranked_probability_score

    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)

    known = set(dc.teams) & set(gbm.teams)
    val = val[val["home_team"].isin(known) & val["away_team"].isin(known)]
    if val.empty:
        return 0.5, float("nan")
    outcomes = val["outcome"].to_numpy(dtype=int)

    # Precompute each base model's W/D/L probabilities once.
    dc_p, gbm_p = [], []
    for m in val.itertuples(index=False):
        n = bool(m.neutral)
        dc_p.append(dc.predict_match(m.home_team, m.away_team, neutral=n))
        gbm_p.append(gbm.predict_match(m.home_team, m.away_team, neutral=n))
    dc_p, gbm_p = np.array(dc_p), np.array(gbm_p)

    best_w, best_rps = 0.5, np.inf
    for w in grid:
        blended = (1.0 - w) * dc_p + w * gbm_p
        blended /= blended.sum(axis=1, keepdims=True)
        rps = ranked_probability_score(blended, outcomes)
        if rps < best_rps:
            best_rps, best_w = rps, float(w)
    return best_w, float(best_rps)
