"""Dixon-Coles bivariate-Poisson match model (interpretable baseline).

Each team gets an **attack** and **defence** rating. Expected goals are:

    log(lambda_home) = mu + home_adv + attack[home] - defence[away]
    log(lambda_away) = mu            + attack[away] - defence[home]

Goals are modelled as Poisson, with the **Dixon-Coles tau correction** that
fixes the well-known underestimation of low scores (0-0, 1-0, 0-1, 1-1) caused
by the independence assumption. Recent matches are weighted more via exponential
time decay (the `xi` half-life parameter), and match importance further scales
the weight so friendlies count less.

This model is transparent -- every prediction decomposes into named team
strengths -- which makes it ideal as a trustworthy baseline to beat on RPS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


def _tau(h, a, lam_h, lam_a, rho):
    """Dixon-Coles low-score correction (vectorised over arrays h, a)."""
    t = np.ones_like(lam_h, dtype=float)
    t = np.where((h == 0) & (a == 0), 1.0 - lam_h * lam_a * rho, t)
    t = np.where((h == 0) & (a == 1), 1.0 + lam_h * rho, t)
    t = np.where((h == 1) & (a == 0), 1.0 + lam_a * rho, t)
    t = np.where((h == 1) & (a == 1), 1.0 - rho, t)
    return t


@dataclass
class DixonColes:
    max_goals: int = 10          # truncation for the score matrix
    xi: float = 0.0018           # time-decay rate (~half-life ≈ 1 year)
    teams: list[str] = field(default_factory=list)
    params: np.ndarray | None = None
    _idx: dict[str, int] = field(default_factory=dict)

    # ---- parameter packing ---------------------------------------------
    @property
    def n(self) -> int:
        return len(self.teams)

    def _unpack(self, p):
        mu, home_adv, rho = p[0], p[1], p[2]
        att = p[3:3 + self.n]
        deff = p[3 + self.n:3 + 2 * self.n]
        # Center for identifiability (attack/defence are relative).
        att = att - att.mean()
        deff = deff - deff.mean()
        return mu, home_adv, rho, att, deff

    # ---- fitting --------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "DixonColes":
        """Fit on a frame with home_team, away_team, home_goals, away_goals,
        date, and optional importance."""
        df = df.copy()
        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._idx = {t: i for i, t in enumerate(self.teams)}

        hi = df["home_team"].map(self._idx).to_numpy()
        ai = df["away_team"].map(self._idx).to_numpy()
        hg = df["home_goals"].to_numpy(dtype=int)
        ag = df["away_goals"].to_numpy(dtype=int)
        neutral = df.get("neutral", pd.Series(0, index=df.index)).to_numpy().astype(float)
        importance = df.get("importance", pd.Series(1.0, index=df.index)).to_numpy(dtype=float)

        # Time-decay weights: most recent match weight 1.
        age_days = (df["date"].max() - df["date"]).dt.days.to_numpy()
        weights = np.exp(-self.xi * age_days) * importance

        log_fact_h = gammaln(hg + 1)
        log_fact_a = gammaln(ag + 1)

        def neg_ll(p):
            mu, home_adv, rho, att, deff = self._unpack(p)
            adv = home_adv * (1.0 - neutral)
            lam_h = np.exp(mu + adv + att[hi] - deff[ai])
            lam_a = np.exp(mu + att[ai] - deff[hi])
            tau = _tau(hg, ag, lam_h, lam_a, rho)
            tau = np.clip(tau, 1e-10, None)
            ll = (np.log(tau)
                  + hg * np.log(lam_h) - lam_h - log_fact_h
                  + ag * np.log(lam_a) - lam_a - log_fact_a)
            return -np.sum(weights * ll)

        x0 = np.concatenate([[0.1, 0.25, -0.05],
                             np.zeros(self.n), np.zeros(self.n)])
        bounds = [(-2, 2), (-1, 1), (-0.2, 0.2)] + [(-3, 3)] * (2 * self.n)
        res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
        self.params = res.x
        return self

    # ---- prediction -----------------------------------------------------
    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """Full P(home_goals=i, away_goals=j) matrix, shape (G+1, G+1)."""
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        mu, home_adv, rho, att, deff = self._unpack(self.params)
        h, a = self._idx[home], self._idx[away]
        adv = 0.0 if neutral else home_adv
        lam_h = np.exp(mu + adv + att[h] - deff[a])
        lam_a = np.exp(mu + att[a] - deff[h])

        g = np.arange(self.max_goals + 1)
        # Poisson pmfs.
        pmf_h = np.exp(g * np.log(lam_h) - lam_h - gammaln(g + 1))
        pmf_a = np.exp(g * np.log(lam_a) - lam_a - gammaln(g + 1))
        mat = np.outer(pmf_h, pmf_a)

        # Apply tau on the 2x2 low-score corner.
        for i in (0, 1):
            for j in (0, 1):
                mat[i, j] *= _tau(np.array(i), np.array(j),
                                  np.array(lam_h), np.array(lam_a), rho).item()
        return mat / mat.sum()

    def predict_match(self, home: str, away: str, neutral: bool = True):
        """Return (p_home_win, p_draw, p_away_win)."""
        mat = self.score_matrix(home, away, neutral)
        p_home = np.tril(mat, -1).sum()   # home_goals > away_goals
        p_draw = np.trace(mat)
        p_away = np.triu(mat, 1).sum()
        return float(p_home), float(p_draw), float(p_away)

    # ---- introspection --------------------------------------------------
    def ratings_table(self) -> pd.DataFrame:
        """Attack/defence ratings per team -- the interpretable output."""
        _, _, _, att, deff = self._unpack(self.params)
        return (pd.DataFrame({"team": self.teams, "attack": att, "defence": deff})
                .sort_values("attack", ascending=False)
                .reset_index(drop=True))
