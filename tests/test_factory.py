"""Tests for the model factory used by the dashboard pipeline."""
import numpy as np
import pandas as pd

from src.models.factory import build_model, MODEL_CHOICES, MODEL_LABELS
from src.models.ensemble import EnsembleModel
from src.models.gbm import GBMModel
from src.models.dixon_coles import DixonColes
from src.models.negbinom import NegativeBinomialModel


def _data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["Strong", "Mid", "Weak"]
    strength = {"Strong": 0.8, "Mid": 0.0, "Weak": -0.8}
    m_rows, f_rows = [], []
    date = pd.Timestamp("2021-01-01")
    for _ in range(n):
        date += pd.Timedelta(days=3)
        h, a = rng.choice(teams, 2, replace=False)
        lam_h = np.exp(0.2 + 0.25 + strength[h] - strength[a])
        lam_a = np.exp(0.2 + strength[a] - strength[h])
        hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
        m_rows.append({"date": date, "home_team": h, "away_team": a,
                       "home_goals": hg, "away_goals": ag,
                       "neutral": False, "importance": 1.0})
        f_rows.append({
            "date": date, "home_team": h, "away_team": a,
            "elo_home": 1500 + 100 * strength[h], "elo_away": 1500 + 100 * strength[a],
            "elo_diff": 100 * (strength[h] - strength[a]) + 65,
            "home_gf": strength[h] + 1, "home_ga": 1.0,
            "away_gf": strength[a] + 1, "away_ga": 1.0,
            "form_diff": strength[h] - strength[a],
            "home_rest": 5, "away_rest": 5, "neutral": 0, "importance": 1.0,
            "home_goals": hg, "away_goals": ag,
            "outcome": 0 if hg > ag else (1 if hg == ag else 2)})
    return pd.DataFrame(f_rows), pd.DataFrame(m_rows)


def test_factory_builds_each_type():
    f, m = _data()
    types = {"gbm": GBMModel, "dixon-coles": DixonColes,
             "negbinom": NegativeBinomialModel, "ensemble": EnsembleModel}
    for choice, cls in types.items():
        model, info = build_model(choice, f, m)
        assert isinstance(model, cls)
        assert info["label"] == MODEL_LABELS[choice]
        # All expose the shared interface.
        p = model.predict_match("Strong", "Weak", neutral=True)
        assert abs(sum(p) - 1.0) < 1e-6


def test_factory_rejects_unknown():
    f, m = _data()
    try:
        build_model("not-a-model", f, m)
        assert False, "should have raised"
    except ValueError:
        pass


def test_ensemble_reports_tuned_weight():
    f, m = _data()
    _, info = build_model("ensemble", f, m)
    assert 0.0 <= info["weight"] <= 1.0
    assert info["val_rps"] >= 0.0


def test_adjustments_applied_for_gbm():
    f, m = _data()
    model, info = build_model("gbm", f, m, adjustments={"Strong": 50})
    assert info.get("adjustments") == {"Strong": 50}


def test_all_choices_listed():
    assert set(MODEL_CHOICES) == set(MODEL_LABELS)
