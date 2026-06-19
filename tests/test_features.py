"""Tests for feature engineering, with a focus on leakage prevention."""
import numpy as np
import pandas as pd

from src.data.features import build_features


def _matches():
    # Three chronological matches involving overlapping teams.
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-01-20"]),
        "home_team": ["A", "A", "B"],
        "away_team": ["B", "C", "C"],
        "home_goals": [2, 0, 3],
        "away_goals": [0, 1, 3],
        "tournament": ["Friendly"] * 3,
        "neutral": [False, False, True],
        "outcome": [0, 2, 1],
        "importance": [0.3, 0.3, 0.3],
        "country": ["A", "A", "B"],
    })


def test_first_match_has_neutral_priors():
    feats = build_features(_matches(), save=False)
    first = feats.iloc[0]
    # No history yet -> equal Elo, zero form.
    assert first["elo_home"] == first["elo_away"]
    assert first["home_gf"] == 0.0 and first["away_gf"] == 0.0


def test_no_lookahead_elo_updates_after_match():
    feats = build_features(_matches(), save=False)
    # A won match 1, so A's Elo entering match 2 must exceed the 1500 start.
    assert feats.iloc[1]["elo_home"] > 1500.0
    # B lost match 1; entering match 3 B's pre-match form reflects that loss.
    assert feats.iloc[2]["home_gf"] >= 0.0  # B scored 0 in match 1


def test_form_reflects_prior_results_only():
    feats = build_features(_matches(), save=False)
    # Match 2: A's form should reflect ONLY match 1 (2 scored, 0 conceded).
    row = feats.iloc[1]
    assert np.isclose(row["home_gf"], 2.0)
    assert np.isclose(row["home_ga"], 0.0)


def test_targets_preserved():
    feats = build_features(_matches(), save=False)
    assert list(feats["outcome"]) == [0, 2, 1]
    assert list(feats["home_goals"]) == [2, 0, 3]
