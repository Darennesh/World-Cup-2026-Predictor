"""Tests for 2026 group resolution."""
import pandas as pd

from src.simulation.groups_2026 import _infer_groups, _snake_draft
from src.ratings.elo import EloModel


def _wc_full_roundrobin():
    """Two groups of 4 with complete round-robins played in 2026."""
    rows = []
    groups = {"G1": ["A", "B", "C", "D"], "G2": ["E", "F", "G", "H"]}
    from itertools import combinations
    for teams in groups.values():
        for h, a in combinations(teams, 2):
            rows.append({"date": "2026-06-12", "home_team": h, "away_team": a,
                         "home_goals": 1, "away_goals": 0, "neutral": True,
                         "importance": 1.0, "tournament": "FIFA World Cup"})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_infer_groups_from_complete_roundrobin():
    df = _wc_full_roundrobin()
    # Only 2 groups here, so the 12-group acceptance check returns None.
    assert _infer_groups(df) is None


def test_infer_groups_requires_twelve():
    # Build 12 complete groups -> inference should succeed.
    from itertools import combinations
    rows = []
    for gi in range(12):
        teams = [f"T{gi}_{k}" for k in range(4)]
        for h, a in combinations(teams, 2):
            rows.append({"date": "2026-06-12", "home_team": h, "away_team": a,
                         "home_goals": 1, "away_goals": 0, "neutral": True,
                         "importance": 1.0, "tournament": "FIFA World Cup"})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    groups = _infer_groups(df)
    assert groups is not None
    assert len(groups) == 12
    assert all(len(t) == 4 for t in groups.values())


def test_snake_draft_distributes_all_teams():
    # 48 teams, each appearing in a 2026 WC match.
    rows = []
    teams = [f"N{i}" for i in range(48)]
    for i in range(0, 48, 2):
        rows.append({"date": "2026-06-12", "home_team": teams[i],
                     "away_team": teams[i + 1], "home_goals": 1,
                     "away_goals": 0, "neutral": True, "importance": 1.0,
                     "tournament": "FIFA World Cup"})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    elo = EloModel(ratings={t: 1500 + i for i, t in enumerate(teams)})
    groups = _snake_draft(df, elo)
    assert len(groups) == 12
    flat = [t for ts in groups.values() for t in ts]
    assert len(flat) == 48 and len(set(flat)) == 48   # all unique, all placed
