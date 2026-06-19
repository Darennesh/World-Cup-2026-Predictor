"""Tests for the data ingestion pipeline."""
import pandas as pd
import pytest

from src.data.ingest import clean, _importance, _outcome


def _raw():
    return pd.DataFrame({
        "date": ["2022-11-20", "bad-date", "2018-06-14"],
        "home_team": [" Qatar ", "X", "Russia"],
        "away_team": ["Ecuador", "Y", "Saudi Arabia"],
        "home_score": [0, 1, 5],
        "away_score": [2, 1, 0],
        "tournament": ["FIFA World Cup", "Friendly", "FIFA World Cup"],
        "neutral": [False, True, False],
    })


def test_clean_drops_bad_dates_and_sorts():
    out = clean(_raw().rename(columns={"home_score": "home_goals",
                                       "away_score": "away_goals"}))
    assert len(out) == 2                      # bad-date row dropped
    assert list(out["date"]) == sorted(out["date"])  # chronological
    assert out.iloc[0]["home_team"] == "Russia"      # 2018 before 2022


def test_clean_strips_team_names_and_types():
    out = clean(_raw().rename(columns={"home_score": "home_goals",
                                       "away_score": "away_goals"}))
    assert "Qatar" in out["home_team"].values   # stripped whitespace
    assert out["home_goals"].dtype.kind == "i"


def test_outcome_encoding():
    assert _outcome(2, 0) == 0   # home win
    assert _outcome(1, 1) == 1   # draw
    assert _outcome(0, 3) == 2   # away win


def test_importance_specificity():
    # Qualification must not be swallowed by the broader 'FIFA World Cup' key.
    assert _importance("FIFA World Cup qualification") < _importance("FIFA World Cup")
    assert _importance("Friendly") < _importance("FIFA World Cup")


def test_clean_requires_columns():
    with pytest.raises(ValueError):
        clean(pd.DataFrame({"date": ["2020-01-01"]}))
