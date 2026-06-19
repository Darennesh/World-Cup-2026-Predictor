"""Elo ratings with World-Cup-style importance weighting and incremental
updating from live 2026 group-stage results.

Elo is our backbone team-strength feature and the mechanism by which the model
'learns the patterns of this World Cup': we start every team from a prior
(history-trained Elo) and nudge it after each 2026 group game. Because the
update is incremental and shrinks toward evidence, it behaves like a lightweight
Bayesian posterior — strong priors are not overturned by a single result, but a
run of results moves a team's rating meaningfully.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import ELO_START, ELO_K, ELO_HOME_ADV


def expected_score(rating_a: float, rating_b: float, home_adv: float = 0.0) -> float:
    """Probability that A beats B (draw counted as half), Elo logistic curve."""
    return 1.0 / (1.0 + 10 ** (-((rating_a + home_adv) - rating_b) / 400.0))


def _result_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for < goals_against:
        return 0.0
    return 0.5


def _margin_multiplier(goal_diff: int) -> float:
    """Goal-difference multiplier (FIFA/World-Football-Elo style): bigger wins
    move ratings more, with diminishing returns."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


@dataclass
class EloModel:
    """Holds a rating per team and updates them match by match."""

    ratings: dict[str, float] = field(default_factory=dict)
    k: float = ELO_K
    home_adv: float = ELO_HOME_ADV

    def rating(self, team: str) -> float:
        return self.ratings.get(team, ELO_START)

    def win_probability(self, home: str, away: str, neutral: bool = False) -> float:
        adv = 0.0 if neutral else self.home_adv
        return expected_score(self.rating(home), self.rating(away), adv)

    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        *,
        neutral: bool = False,
        importance: float = 1.0,
    ) -> None:
        """Apply one match result, mutating both teams' ratings in place."""
        adv = 0.0 if neutral else self.home_adv
        exp_home = expected_score(self.rating(home), self.rating(away), adv)
        score_home = _result_score(home_goals, away_goals)

        k_eff = self.k * importance * _margin_multiplier(home_goals - away_goals)
        delta = k_eff * (score_home - exp_home)

        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta

    def update_from_results(self, results, *, neutral: bool = True, importance: float = 1.0):
        """Bulk update from an iterable of (home, away, hg, ag) tuples.

        Use this to feed the 2026 group-stage results in chronological order
        before simulating the knockouts.
        """
        for home, away, hg, ag in results:
            self.update(home, away, hg, ag, neutral=neutral, importance=importance)
        return self
