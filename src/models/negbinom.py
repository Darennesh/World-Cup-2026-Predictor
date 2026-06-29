"""Negative Binomial goal model (over-dispersion-aware).

The Poisson assumption underlying Dixon-Coles forces the variance of goals to
equal the mean. Real football is **over-dispersed**: blowouts (5-0, 7-1) happen
more often than Poisson predicts, so the true variance exceeds the mean. The
Negative Binomial distribution adds a single dispersion parameter `r` that lets
variance exceed the mean (variance = mu + mu**2 / r); as r -> infinity it
collapses back to Poisson.

This model reuses the Dixon-Coles attack/defence means (so it is a strict
"NB version" of the baseline) and fits `r` by method of moments on the observed
goal distribution. It keeps the Dixon-Coles low-score (tau) correction and
exposes the same `score_matrix` interface, so it drops into the simulation and
every metric, and can be compared head-to-head on RPS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln

from src.models.dixon_coles import DixonColes, _tau


def _nb_pmf(k: np.ndarray, mu: float, r: float) -> np.ndarray:
    """Negative Binomial pmf with mean `mu` and dispersion `r` (size).

    P(k) = Gamma(k+r)/(Gamma(r) k!) * (r/(r+mu))**r * (mu/(r+mu))**k
    """
    log_p = (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
             + r * np.log(r / (r + mu))
             + k * np.log(mu / (r + mu)))
    return np.exp(log_p)


@dataclass
class NegativeBinomialModel:
    max_goals: int = 10
    xi: float = 0.0018
    r: float = 20.0                       # dispersion; large -> ~Poisson
    dc: DixonColes = field(default=None)

    # ---- fitting --------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "NegativeBinomialModel":
        """Fit Dixon-Coles for the means, then estimate the dispersion `r` from
        the gap between observed goal variance and the model's expected goals."""
        self.dc = DixonColes(max_goals=self.max_goals, xi=self.xi).fit(df)

        # Predicted mean goals per match (both sides), computed directly from the
        # fitted parameters (fast, no per-match score matrix).
        mu0, home_adv, _rho, att, deff = self.dc._unpack(self.dc.params)
        hi = df["home_team"].map(self.dc._idx).to_numpy()
        ai = df["away_team"].map(self.dc._idx).to_numpy()
        neutral = df.get("neutral", pd.Series(True, index=df.index)).to_numpy().astype(float)
        adv = home_adv * (1.0 - neutral)
        lam_h = np.exp(mu0 + adv + att[hi] - deff[ai])
        lam_a = np.exp(mu0 + att[ai] - deff[hi])

        mu = np.concatenate([lam_h, lam_a])
        obs = np.concatenate([df["home_goals"].to_numpy(dtype=float),
                              df["away_goals"].to_numpy(dtype=float)])

        mean_mu = mu.mean()
        # Residual variance around the model mean (over-dispersion signal).
        resid_var = np.mean((obs - mu) ** 2)
        excess = resid_var - mean_mu
        if excess <= 1e-6:
            self.r = 1e6                  # no over-dispersion -> Poisson limit
        else:
            self.r = float(mean_mu ** 2 / excess)
        self.r = float(np.clip(self.r, 1.0, 1e6))
        return self

    @property
    def teams(self):
        return self.dc.teams if self.dc else []

    # ---- prediction -----------------------------------------------------
    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        if self.dc is None:
            raise RuntimeError("Model not fitted.")
        mu, home_adv, rho, att, deff = self.dc._unpack(self.dc.params)
        h, a = self.dc._idx[home], self.dc._idx[away]
        adv = 0.0 if neutral else home_adv
        lam_h = float(np.exp(mu + adv + att[h] - deff[a]))
        lam_a = float(np.exp(mu + att[a] - deff[h]))

        g = np.arange(self.max_goals + 1)
        pmf_h = _nb_pmf(g, lam_h, self.r)
        pmf_a = _nb_pmf(g, lam_a, self.r)
        mat = np.outer(pmf_h, pmf_a)

        # Keep the Dixon-Coles low-score correction.
        for i in (0, 1):
            for j in (0, 1):
                mat[i, j] *= _tau(np.array(i), np.array(j),
                                  np.array(lam_h), np.array(lam_a), rho).item()
        return mat / mat.sum()

    def predict_match(self, home: str, away: str, neutral: bool = True):
        mat = self.score_matrix(home, away, neutral)
        p_home = float(np.tril(mat, -1).sum())
        p_draw = float(np.trace(mat))
        p_away = float(np.triu(mat, 1).sum())
        return p_home, p_draw, p_away
