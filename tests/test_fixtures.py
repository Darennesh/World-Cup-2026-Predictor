"""Tests for upcoming-fixtures derivation, EAT scheduling, and predictions."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.simulation.fixtures import (remaining_fixtures, upcoming_fixtures,
                                     load_schedule, EAT, _parse_utc,
                                     _fixture_key)


class FixedModel:
    def __init__(self, teams):
        self.strength = {t: i for i, t in enumerate(teams)}
        self._teams = list(teams)

    @property
    def teams(self):
        return self._teams

    def score_matrix(self, home, away, neutral=True):
        from scipy.special import gammaln
        lam_h = 0.8 + 0.2 * self.strength[home]
        lam_a = 0.8 + 0.2 * self.strength[away]
        g = np.arange(8)
        ph = np.exp(g * np.log(lam_h) - lam_h - gammaln(g + 1))
        pa = np.exp(g * np.log(lam_a) - lam_a - gammaln(g + 1))
        m = np.outer(ph, pa)
        return m / m.sum()


def _matches_with_one_played():
    # Group of 4; one fixture (A vs B) already played in the 2026 WC.
    rows = [{
        "date": pd.Timestamp("2026-06-18"), "home_team": "A", "away_team": "B",
        "home_goals": 1, "away_goals": 0, "neutral": True,
        "importance": 1.0, "tournament": "FIFA World Cup", "outcome": 0,
    }]
    return pd.DataFrame(rows)


GROUPS = {"A": ["A", "B", "C", "D"]}


def test_remaining_excludes_played():
    rem = remaining_fixtures(_matches_with_one_played(), GROUPS)
    pairs = {frozenset((h, a)) for _, h, a in rem}
    assert frozenset(("A", "B")) not in pairs      # already played
    assert frozenset(("C", "D")) in pairs          # still to play
    assert len(rem) == 5                            # 6 total - 1 played


def test_parse_utc_handles_z_and_naive():
    assert _parse_utc("2026-06-24T19:00:00Z").hour == 19
    naive = _parse_utc("2026-06-24T19:00:00")
    assert naive.tzinfo == timezone.utc


def test_eat_offset_is_plus_three():
    dt = _parse_utc("2026-06-24T19:00:00Z")
    assert dt.astimezone(EAT).hour == 22          # +3h


def test_upcoming_sorted_timed_first():
    model = FixedModel(GROUPS["A"])
    schedule = {_fixture_key("C", "D"): _parse_utc("2026-06-24T16:00:00Z")}
    up = upcoming_fixtures(model, _matches_with_one_played(), GROUPS, schedule)
    # The scheduled fixture must come first; its EAT label is concrete.
    assert up[0].home == "C" and up[0].away == "D"
    assert "EAT" in up[0].eat_label and "TBD" not in up[0].eat_label
    # Probabilities form a valid distribution.
    assert abs(up[0].p_home + up[0].p_draw + up[0].p_away - 1.0) < 1e-6


def test_unscheduled_fixtures_show_tbd():
    model = FixedModel(GROUPS["A"])
    up = upcoming_fixtures(model, _matches_with_one_played(), GROUPS, {})
    assert all(f.kickoff_utc is None for f in up)
    assert all(f.eat_label == "Time TBD (EAT)" for f in up)


def test_favourite_is_highest_probability():
    model = FixedModel(GROUPS["A"])
    up = upcoming_fixtures(model, _matches_with_one_played(), GROUPS, {})
    for f in up:
        best = max((f.p_home, f.home), (f.p_away, f.away),
                   (f.p_draw, "Draw"), key=lambda x: x[0])[1]
        assert f.favourite == best


# ---- Knockout fixtures ----------------------------------------------------
def _ko_groups():
    return {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}


def _ko_matches():
    # One cross-group knockout tie already played (A1 beat B2 2-0); A3 vs B4
    # is an unplayed knockout tie.
    rows = [{
        "date": pd.Timestamp("2026-06-29"), "home_team": "A1", "away_team": "B2",
        "home_goals": 2, "away_goals": 0, "neutral": True,
        "importance": 1.0, "tournament": "FIFA World Cup", "outcome": 0,
    }]
    return pd.DataFrame(rows)


def test_played_knockout_detects_cross_group():
    from src.simulation.fixtures import played_knockout_results
    played = played_knockout_results(_ko_matches(), _ko_groups())
    assert _fixture_key("A1", "B2") in played
    # An intra-group pair must NOT be treated as knockout.
    assert _fixture_key("A1", "A2") not in played


def test_upcoming_knockout_marks_round_and_skips_played():
    from src.simulation.fixtures import upcoming_knockout_fixtures
    model = FixedModel(["A1", "B2", "A3", "B4"])
    bracket = ["A1", "B2", "A3", "B4"]      # 4-team mini-knockout
    up = upcoming_knockout_fixtures(model, _ko_matches(), _ko_groups(), bracket, {})
    pairs = {frozenset((f.home, f.away)) for f in up}
    # A1 vs B2 already played -> excluded; A3 vs B4 is upcoming.
    assert frozenset(("A1", "B2")) not in pairs
    assert frozenset(("A3", "B4")) in pairs
    assert all(f.is_knockout for f in up)
    assert all(f.favourite != "Draw" for f in up)   # knockouts have a winner


def test_knockout_winner_resolves_goals_and_pens():
    from src.simulation.fixtures import knockout_winner
    # Decisive result -> team with more goals.
    assert knockout_winner("Spain", "Morocco", 2, 1) == "Spain"
    # Level result, no meta -> unknown.
    assert knockout_winner("Spain", "Morocco", 1, 1) is None
    # Level result + shootout meta -> the shootout winner.
    meta = {"Morocco|Spain": {"home": "Spain", "away": "Morocco",
                              "pens_home": 2, "pens_away": 3, "winner": "Morocco"}}
    assert knockout_winner("Spain", "Morocco", 1, 1, meta) == "Morocco"


def test_round_schedule_prestages_semifinal_times():
    from src.simulation.fixtures import upcoming_knockout_fixtures, _parse_utc
    # 4-team bracket -> one Semi-final tie (teams known, unplayed).
    model = FixedModel(["A", "B", "C", "D"])
    bracket = ["A", "B", "C", "D"]
    # No games played; pre-staged Semi-final time supplies the kickoff.
    round_sched = {"Semi-final": [_parse_utc("2026-07-14T19:00:00Z")]}
    matches = pd.DataFrame(columns=["date", "home_team", "away_team",
                                    "home_goals", "away_goals", "neutral",
                                    "importance", "tournament"])
    up = upcoming_knockout_fixtures(model, matches, {"A": ["A", "B", "C", "D"]},
                                    bracket, schedule={},
                                    round_schedule=round_sched)
    sf = [f for f in up if f.group == "Semi-final"]
    assert sf and sf[0].kickoff_utc is not None      # time was pre-staged
    assert "TBD" not in sf[0].eat_label


