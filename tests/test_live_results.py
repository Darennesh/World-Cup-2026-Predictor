"""Tests for live results fetching/merging (football-data.org integration)."""
import pandas as pd

from src.data.live_results import (_map_team, merge_live_into_mirror,
                                   fetch_live_results)


def test_team_name_mapping():
    assert _map_team("Korea Republic") == "South Korea"
    assert _map_team("Czechia") == "Czech Republic"
    assert _map_team("Türkiye") == "Turkey"
    assert _map_team("Brazil") == "Brazil"        # unknown passes through


def test_fetch_without_key_returns_empty(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    df = fetch_live_results()
    assert df.empty
    assert "home_team" in df.columns       # schema preserved


def _mirror():
    return pd.DataFrame({
        "date": ["2026-06-22", "2026-06-22"],
        "home_team": ["France", "Argentina"],
        "away_team": ["Iraq", "Austria"],
        "home_score": [None, None],        # not yet published
        "away_score": [None, None],
        "tournament": ["FIFA World Cup", "FIFA World Cup"],
        "city": [None, None], "country": [None, None],
        "neutral": [True, True],
    })


def _live():
    return pd.DataFrame({
        "date": ["2026-06-22"],
        "home_team": ["Argentina"], "away_team": ["Austria"],
        "home_score": [2], "away_score": [0],
        "tournament": ["FIFA World Cup"],
        "city": [None], "country": [None], "neutral": [True],
    })


def test_merge_prefers_live_for_duplicate():
    merged = merge_live_into_mirror(_mirror(), _live())
    arg = merged[(merged["home_team"] == "Argentina")
                 & (merged["away_team"] == "Austria")]
    # Exactly one Argentina-Austria row, and it carries the live score.
    assert len(arg) == 1
    assert int(arg.iloc[0]["home_score"]) == 2
    assert int(arg.iloc[0]["away_score"]) == 0


def test_merge_keeps_unmatched_mirror_rows():
    merged = merge_live_into_mirror(_mirror(), _live())
    # France-Iraq (no live result) is still present.
    assert ((merged["home_team"] == "France")
            & (merged["away_team"] == "Iraq")).any()
    assert len(merged) == 2


def test_merge_empty_live_is_noop():
    mirror = _mirror()
    out = merge_live_into_mirror(mirror, pd.DataFrame())
    assert len(out) == len(mirror)
