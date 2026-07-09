"""Tests for 2026 group resolution."""
import pandas as pd

from src.simulation.groups_2026 import (_infer_groups, _snake_draft,
                                        _reconcile_r32_with_actual)
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


def test_reconcile_r32_corrects_misslotted_third():
    # Winners W_A/W_B face thirds from OTHER groups. The seeding slots T_D to
    # face W_A, but the actual R32 game was W_A vs T_C (third mis-slotted).
    groups = {"A": ["W_A", "R_A", "a3", "a4"], "B": ["W_B", "R_B", "b3", "b4"],
              "C": ["W_C", "R_C", "T_C", "c4"], "D": ["W_D", "R_D", "T_D", "d4"]}
    seeded = ["W_A", "T_D", "W_B", "T_C"]        # W_A-T_D is WRONG
    played = pd.DataFrame([
        {"date": "2026-07-01", "home_team": "W_A", "away_team": "T_C",
         "home_goals": 2, "away_goals": 0, "neutral": True,
         "importance": 1.0, "tournament": "FIFA World Cup"},
        {"date": "2026-07-01", "home_team": "W_B", "away_team": "T_D",
         "home_goals": 1, "away_goals": 0, "neutral": True,
         "importance": 1.0, "tournament": "FIFA World Cup"},
    ])
    played["date"] = pd.to_datetime(played["date"])
    fixed = _reconcile_r32_with_actual(played, groups, seeded, {"W_A", "W_B"})
    # After reconciliation W_A faces its real opponent T_C, W_B faces T_D.
    assert {fixed[0], fixed[1]} == {"W_A", "T_C"}
    assert {fixed[2], fixed[3]} == {"W_B", "T_D"}
