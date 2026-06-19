"""Snapshot of each team's latest state (Elo + recent form) after walking the
full match history once.

The GBM match model is trained on pre-match features, but to predict a *future*
fixture (e.g. a 2026 knockout tie that has not been played) we need each team's
current Elo rating and recent-form averages. This module replays history in
chronological order and returns the final per-team state, which the model then
turns into a feature vector for any hypothetical matchup.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import pandas as pd

from src.config import ELO_START
from src.data.features import FORM_WINDOW, _weighted_form
from src.ratings.elo import EloModel


@dataclass
class TeamState:
    elo: float
    gf: float        # recency-weighted goals for
    ga: float        # recency-weighted goals against


def build_team_states(matches: pd.DataFrame) -> tuple[dict[str, TeamState], EloModel]:
    """Replay all matches chronologically; return latest state per team and the
    fitted Elo model (so callers can reuse its home-advantage constant)."""
    matches = matches.sort_values("date").reset_index(drop=True)
    elo = EloModel()
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))

    for m in matches.itertuples(index=False):
        elo.update(m.home_team, m.away_team, m.home_goals, m.away_goals,
                   neutral=bool(m.neutral), importance=float(m.importance))
        form[m.home_team].append((m.home_goals, m.away_goals))
        form[m.away_team].append((m.away_goals, m.home_goals))

    states: dict[str, TeamState] = {}
    teams = set(matches["home_team"]) | set(matches["away_team"])
    for t in teams:
        gf, ga = _weighted_form(form[t])
        states[t] = TeamState(elo=elo.rating(t), gf=gf, ga=ga)
    return states, elo
