"""Bayesian-style rating update from the live 2026 World Cup group stage.

This is the mechanism that makes the model "learn the patterns of *this* World
Cup". We treat each team's pre-tournament Elo (trained on all history up to the
first 2026 match) as a **prior**, then update it with the 2026 group-stage
results actually played to obtain a **posterior** rating.

Why this is Bayesian in spirit: the Elo update is an incremental, evidence-
weighted shrinkage of the prior toward what we observe. A single surprising
result nudges -- but does not overturn -- a strong prior; a consistent run of
results moves the rating substantially. We deliberately *de-emphasise* over-
reacting to a 3-game sample by using a moderate update rate, because a strong
team can stumble in one group game and still be strong (e.g. Argentina lost
their 2022 opener and won the tournament).

The posterior ratings are what the simulation should seed its knockout forecasts
with, blending long-run quality (prior) and current-tournament form (group
results).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ratings.elo import EloModel


# A World Cup group game is high-importance evidence, but we keep the update
# rate moderate so three games don't swamp years of prior signal.
GROUP_STAGE_IMPORTANCE = 1.0


def _is_2026_wc(df: pd.DataFrame) -> pd.Series:
    return (
        (df["date"] >= "2026-01-01")
        & df["tournament"].str.contains("World Cup", case=False, na=False)
        & ~df["tournament"].str.contains("qual", case=False, na=False)
    )


@dataclass
class RatingUpdate:
    prior: dict[str, float]
    posterior: dict[str, float]
    n_group_matches: int

    def shifts(self) -> pd.DataFrame:
        """Per-team prior -> posterior change, biggest movers first."""
        rows = [{"team": t, "prior": self.prior[t],
                 "posterior": self.posterior.get(t, self.prior[t]),
                 "delta": self.posterior.get(t, self.prior[t]) - self.prior[t]}
                for t in self.prior]
        return (pd.DataFrame(rows)
                .sort_values("delta", key=lambda s: s.abs(), ascending=False)
                .reset_index(drop=True))


def fit_prior_then_update(matches: pd.DataFrame) -> tuple[EloModel, RatingUpdate]:
    """Build prior Elo from pre-2026 history, then apply 2026 group results.

    Returns the updated EloModel (posterior) and a RatingUpdate record capturing
    both rating snapshots for inspection.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    is_wc = _is_2026_wc(matches)
    history = matches[~is_wc]
    group = matches[is_wc]

    # --- Prior: train Elo on everything before the 2026 tournament ---
    elo = EloModel()
    for m in history.itertuples(index=False):
        elo.update(m.home_team, m.away_team, m.home_goals, m.away_goals,
                   neutral=bool(m.neutral), importance=float(m.importance))
    prior = dict(elo.ratings)

    # --- Posterior: update with the 2026 group games (neutral venues) ---
    for m in group.itertuples(index=False):
        elo.update(m.home_team, m.away_team, m.home_goals, m.away_goals,
                   neutral=True, importance=GROUP_STAGE_IMPORTANCE)
    posterior = dict(elo.ratings)

    # Ensure every team that appears anywhere has a prior entry for diffing.
    for t in set(group["home_team"]) | set(group["away_team"]):
        prior.setdefault(t, posterior.get(t, elo.rating(t)))

    return elo, RatingUpdate(prior=prior, posterior=posterior,
                             n_group_matches=len(group))


def group_results_2026(matches: pd.DataFrame) -> list[tuple[str, str, int, int]]:
    """Return the played 2026 group-stage results as (home, away, hg, ag)."""
    g = matches[_is_2026_wc(matches)].sort_values("date")
    return [(m.home_team, m.away_team, int(m.home_goals), int(m.away_goals))
            for m in g.itertuples(index=False)]
