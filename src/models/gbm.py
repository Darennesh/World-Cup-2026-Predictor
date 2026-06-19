"""LightGBM challenger match model (pure predictive power).

Two gradient-boosted regressors with a **Poisson objective** predict the
expected goals for the home and away side from the engineered features (Elo,
recent form, rest, neutrality, importance). Predicted goal rates are turned into
a score-probability matrix assuming (near-)independent Poisson counts, which
exposes the same `score_matrix(home, away, neutral)` interface as Dixon-Coles --
so it plugs straight into the simulation engine and is scored on the same RPS.

Gradient boosting can capture non-linear feature interactions (e.g. rest-day
effects that depend on Elo gap) that the additive Dixon-Coles model cannot. We
verify on a temporal split whether that flexibility actually lowers RPS; with
limited international data it is not guaranteed, which is exactly why we keep the
interpretable baseline to compare against. SHAP values (see scripts) explain the
black box per prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln

from src.models.team_state import build_team_states, TeamState

FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "home_gf", "home_ga", "away_gf", "away_ga", "form_diff",
    "home_rest", "away_rest", "neutral", "importance",
]


def _poisson_pmf(lam: float, k: np.ndarray) -> np.ndarray:
    return np.exp(k * np.log(lam) - lam - gammaln(k + 1))


@dataclass
class GBMModel:
    max_goals: int = 10
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 50
    max_lambda: float = 8.0          # clip predicted goal rate for stability
    home_model: object = None
    away_model: object = None
    states: dict[str, TeamState] = field(default_factory=dict)
    home_adv: float = 65.0

    # ---- training ------------------------------------------------------
    def fit(self, features: pd.DataFrame, matches: pd.DataFrame) -> "GBMModel":
        """Fit on the engineered feature table; `matches` is replayed to capture
        each team's latest state for future-fixture prediction."""
        import lightgbm as lgb

        X = features[FEATURES]
        params = dict(objective="poisson", n_estimators=self.n_estimators,
                      learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                      min_child_samples=self.min_child_samples,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      verbosity=-1)
        # Weight recent / important matches more (mirrors the baseline's intent).
        w = features["importance"].to_numpy()

        self.home_model = lgb.LGBMRegressor(**params).fit(
            X, features["home_goals"], sample_weight=w)
        self.away_model = lgb.LGBMRegressor(**params).fit(
            X, features["away_goals"], sample_weight=w)

        self.states, elo = build_team_states(matches)
        self.home_adv = elo.home_adv
        return self

    def apply_adjustments(self, adjustments: dict[str, float]) -> "GBMModel":
        """Shift team Elo states by curated team-news deltas (injuries, etc.).

        Mutates this model's states in place and returns self, so downstream
        predictions reflect the adjusted strengths.
        """
        from src.ratings.adjustments import apply_to_states
        if adjustments:
            self.states = apply_to_states(self.states, adjustments)
        return self

    # ---- feature construction for an arbitrary fixture -----------------
    def _make_features(self, home: str, away: str, neutral: bool) -> pd.DataFrame:
        sh, sa = self.states[home], self.states[away]
        adv = 0.0 if neutral else self.home_adv
        row = {
            "elo_home": sh.elo, "elo_away": sa.elo,
            "elo_diff": (sh.elo + adv) - sa.elo,
            "home_gf": sh.gf, "home_ga": sh.ga,
            "away_gf": sa.gf, "away_ga": sa.ga,
            "form_diff": (sh.gf - sh.ga) - (sa.gf - sa.ga),
            "home_rest": 5, "away_rest": 5,        # typical tournament rest
            "neutral": int(neutral), "importance": 1.0,
        }
        return pd.DataFrame([row])[FEATURES]

    def expected_goals(self, home: str, away: str, neutral: bool = True):
        X = self._make_features(home, away, neutral)
        lam_h = float(np.clip(self.home_model.predict(X)[0], 0.05, self.max_lambda))
        lam_a = float(np.clip(self.away_model.predict(X)[0], 0.05, self.max_lambda))
        return lam_h, lam_a

    # ---- prediction interface (matches DixonColes) ---------------------
    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(home, away, neutral)
        g = np.arange(self.max_goals + 1)
        mat = np.outer(_poisson_pmf(lam_h, g), _poisson_pmf(lam_a, g))
        return mat / mat.sum()

    def predict_match(self, home: str, away: str, neutral: bool = True):
        mat = self.score_matrix(home, away, neutral)
        p_home = float(np.tril(mat, -1).sum())
        p_draw = float(np.trace(mat))
        p_away = float(np.triu(mat, 1).sum())
        return p_home, p_draw, p_away

    # ---- batched prediction (fast path for evaluation) -----------------
    def predict_matches(self, pairs: pd.DataFrame) -> np.ndarray:
        """Vectorised win/draw/loss probabilities for many fixtures at once.

        `pairs` needs columns home_team, away_team, neutral. Returns an array of
        shape (n, 3) ordered [home_win, draw, away_win]. Building one feature
        matrix and calling LightGBM once is far faster than per-match loops.
        """
        rows = []
        for p in pairs.itertuples(index=False):
            sh, sa = self.states[p.home_team], self.states[p.away_team]
            adv = 0.0 if bool(p.neutral) else self.home_adv
            rows.append({
                "elo_home": sh.elo, "elo_away": sa.elo,
                "elo_diff": (sh.elo + adv) - sa.elo,
                "home_gf": sh.gf, "home_ga": sh.ga,
                "away_gf": sa.gf, "away_ga": sa.ga,
                "form_diff": (sh.gf - sh.ga) - (sa.gf - sa.ga),
                "home_rest": 5, "away_rest": 5,
                "neutral": int(bool(p.neutral)), "importance": 1.0,
            })
        X = pd.DataFrame(rows)[FEATURES]
        lam_h = np.clip(self.home_model.predict(X), 0.05, self.max_lambda)
        lam_a = np.clip(self.away_model.predict(X), 0.05, self.max_lambda)

        g = np.arange(self.max_goals + 1)
        out = np.empty((len(X), 3))
        for i in range(len(X)):
            mat = np.outer(_poisson_pmf(lam_h[i], g), _poisson_pmf(lam_a[i], g))
            mat /= mat.sum()
            out[i, 0] = np.tril(mat, -1).sum()
            out[i, 1] = np.trace(mat)
            out[i, 2] = np.triu(mat, 1).sum()
        return out

    @property
    def teams(self) -> list[str]:
        return sorted(self.states.keys())
