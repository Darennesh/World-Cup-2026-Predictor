"""Model factory for the dashboard pipeline -- lets the live build select which
match model to use, so models can be A/B compared on the public site.

Supported choices:
  * "gbm"        -- LightGBM goal model (fast; the long-standing default)
  * "ensemble"   -- Dixon-Coles + LightGBM blend, weight tuned out-of-sample
                    (best back-tested RPS)
  * "dixon-coles"-- interpretable Poisson baseline
  * "negbinom"   -- over-dispersion-aware Negative Binomial

Each returned model exposes the common `score_matrix(home, away, neutral)`
interface (plus `.teams`), so the simulation engine, locked-bracket path, and
every metric work identically regardless of choice. Team-news adjustments are
applied where the model supports them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.gbm import GBMModel
from src.models.dixon_coles import DixonColes
from src.models.negbinom import NegativeBinomialModel
from src.models.ensemble import EnsembleModel, tune_weight

MODEL_CHOICES = ("gbm", "ensemble", "dixon-coles", "negbinom")
MODEL_LABELS = {
    "gbm": "LightGBM",
    "ensemble": "Ensemble (Dixon-Coles + LightGBM)",
    "dixon-coles": "Dixon-Coles",
    "negbinom": "Negative Binomial",
}

# Validation slice (most recent fraction of training data) used to tune the
# ensemble blend weight out-of-sample.
_VAL_FRAC = 0.15


def build_model(choice: str, features: pd.DataFrame, matches: pd.DataFrame,
                adjustments: dict[str, float] | None = None):
    """Fit and return the requested model, with team-news adjustments applied
    where supported. Returns (model, info_dict)."""
    choice = (choice or "gbm").lower()
    if choice not in MODEL_CHOICES:
        raise ValueError(f"Unknown model '{choice}'. Choose from {MODEL_CHOICES}.")

    info: dict = {"choice": choice, "label": MODEL_LABELS[choice]}

    if choice == "gbm":
        model = GBMModel().fit(features, matches)
        _maybe_adjust(model, adjustments, info)
        return model, info

    if choice == "dixon-coles":
        return DixonColes().fit(matches), info

    if choice == "negbinom":
        return NegativeBinomialModel().fit(matches), info

    # ----- ensemble -----
    m = matches.sort_values("date").copy()
    if "outcome" not in m.columns:
        m["outcome"] = np.where(m["home_goals"] > m["away_goals"], 0,
                                np.where(m["home_goals"] == m["away_goals"], 1, 2))
    f = features.sort_values("date")
    split = int(len(m) * (1 - _VAL_FRAC))
    train_m, val_m = m.iloc[:split], m.iloc[split:]
    train_f = f.iloc[:split]

    # Tune the blend weight on the validation slice (out-of-sample), then refit
    # both base models on ALL data for the live prediction.
    dc_val = DixonColes().fit(train_m)
    gbm_val = GBMModel().fit(train_f, train_m)
    weight, val_rps = tune_weight(dc_val, gbm_val, val_m)

    dc = DixonColes().fit(matches)
    gbm = GBMModel().fit(features, matches)
    _maybe_adjust(gbm, adjustments, info)
    model = EnsembleModel(dc=dc, gbm=gbm, weight=weight)
    info.update({"weight": weight, "val_rps": val_rps})
    return model, info


def _maybe_adjust(model, adjustments, info) -> None:
    if adjustments and hasattr(model, "apply_adjustments"):
        model.apply_adjustments(adjustments)
        info["adjustments"] = adjustments
