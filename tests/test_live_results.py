"""Tests for live results fetching/merging (football-data.org integration)."""
import pandas as pd

from src.data.live_results import (_map_team, merge_live_into_mirror,
                                   fetch_live_results, shootouts_to_meta,
                                   load_knockout_meta)


def test_team_name_mapping():
    assert _map_team("Korea Republic") == "South Korea"
    assert _map_team("Czechia") == "Czech Republic"
    assert _map_team("Türkiye") == "Turkey"
    assert _map_team("Cape Verde Islands") == "Cape Verde"
    assert _map_team("Brazil") == "Brazil"        # unknown passes through


def test_shootouts_to_meta_records_2026_winners(tmp_path):
    shootouts = tmp_path / "shootouts.csv"
    shootouts.write_text(
        "date,home_team,away_team,winner,first_shooter\n"
        "2018-07-03,Croatia,Denmark,Croatia,Denmark\n"      # pre-2026, ignored
        "2026-06-29,Germany,Paraguay,Paraguay,Germany\n"
        "2026-06-29,Netherlands,Morocco,Morocco,Netherlands\n",
        encoding="utf-8")
    meta_path = tmp_path / "knockout_meta.json"
    n = shootouts_to_meta(shootouts, meta_path)
    assert n == 2                                  # only the two 2026 ties
    meta = load_knockout_meta(meta_path)
    assert meta["Germany|Paraguay"]["winner"] == "Paraguay"
    assert meta["Morocco|Netherlands"]["winner"] == "Morocco"
    assert all(v["shootout"] for v in meta.values())


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


def test_merge_dedups_across_date_and_name_skew():
    # Mirror has the game one day off and with an alternate name; the live
    # result must replace it (no duplicate), matching on the team pair.
    mirror = pd.DataFrame({
        "date": ["2026-06-21"],
        "home_team": ["Uruguay"], "away_team": ["Cape Verde"],
        "home_score": [2], "away_score": [2],
        "tournament": ["FIFA World Cup"],
        "city": [None], "country": [None], "neutral": [True],
    })
    live = pd.DataFrame({
        "date": ["2026-06-22"],            # one day off
        "home_team": ["Uruguay"], "away_team": ["Cape Verde"],
        "home_score": [2], "away_score": [2],
        "tournament": ["FIFA World Cup"],
        "city": [None], "country": [None], "neutral": [True],
    })
    merged = merge_live_into_mirror(mirror, live)
    uru = merged[merged["home_team"] == "Uruguay"]
    assert len(uru) == 1                   # not duplicated
