"""Live forecasting of the remaining group-stage matches.

As the group stage unfolds, this produces -- for every group -- the current
standings from games already played, a prediction (win/draw/loss + expected
score) for each *remaining* fixture, and each team's probability of finishing
1st / 2nd / 3rd and of advancing to the knockout stage.

These predictions fluctuate with performance because the underlying model blends
two signals: a team's long-run strength prior (Elo trained on all history) and
its recent form. That blend is exactly why a strong side that stumbles in its
opener -- Spain or Portugal drawing first up -- is still forecast to bounce back:
the prior keeps their rating high while the new result nudges it. Team-news
effects (injuries, suspensions) enter via optional rating adjustments
(see ratings.adjustments), which shift a team's strength before these forecasts
are computed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from src.simulation.standings import Group


@dataclass
class MatchForecast:
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    exp_home: float       # expected goals, home
    exp_away: float       # expected goals, away


@dataclass
class GroupForecast:
    name: str
    remaining: list[MatchForecast] = field(default_factory=list)
    advance_prob: dict[str, float] = field(default_factory=dict)
    finish_first: dict[str, float] = field(default_factory=dict)
    played: list[tuple[str, str, int, int]] = field(default_factory=list)


def _match_probs(model, home: str, away: str):
    mat = model.score_matrix(home, away, neutral=True)
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    g = np.arange(mat.shape[0])
    exp_home = float((mat.sum(axis=1) * g).sum())
    exp_away = float((mat.sum(axis=0) * g).sum())
    return p_home, p_draw, p_away, exp_home, exp_away


def forecast_group(model, name: str, teams: list[str],
                   played: list[tuple[str, str, int, int]],
                   *, n_sims: int = 5000, advance: int = 2,
                   rng: np.random.Generator | None = None) -> GroupForecast:
    """Forecast one group: remaining-match predictions + advance probabilities.

    `played` holds already-decided results (home, away, hg, ag) among `teams`.
    `advance` is how many automatically advance (top 2); third place may also
    advance as a best-third but that depends on other groups, so here we report
    finishing position probabilities and top-2 advancement.
    """
    rng = rng or np.random.default_rng(0)
    played_pairs = {frozenset((h, a)) for h, a, _, _ in played}

    # Remaining fixtures = full round-robin minus those already played.
    remaining_pairs = [(h, a) for h, a in combinations(teams, 2)
                       if frozenset((h, a)) not in played_pairs]

    forecasts = []
    for h, a in remaining_pairs:
        ph, pd, pa, eh, ea = _match_probs(model, h, a)
        forecasts.append(MatchForecast(h, a, ph, pd, pa, eh, ea))

    # Monte Carlo the remaining games to get finishing-position probabilities.
    first = {t: 0 for t in teams}
    top2 = {t: 0 for t in teams}
    # Pre-cache score matrices for sampling.
    sample_mats = {}
    for fc in forecasts:
        m = model.score_matrix(fc.home, fc.away, neutral=True)
        flat = (m / m.sum()).ravel()
        cum = np.cumsum(flat)
        cum[-1] = 1.0
        sample_mats[(fc.home, fc.away)] = (cum, m.shape[1])

    for _ in range(n_sims):
        g = Group(name, teams)
        for h, a, hg, ag in played:
            g.play(h, a, hg, ag)
        for fc in forecasts:
            cum, ncols = sample_mats[(fc.home, fc.away)]
            idx = int(np.searchsorted(cum, rng.random()))
            g.play(fc.home, fc.away, idx // ncols, idx % ncols)
        table = g.standings(rng)
        first[table[0].team] += 1
        for r in table[:advance]:
            top2[r.team] += 1

    return GroupForecast(
        name=name,
        remaining=forecasts,
        advance_prob={t: top2[t] / n_sims for t in teams},
        finish_first={t: first[t] / n_sims for t in teams},
        played=list(played),
    )


def played_results_for(teams: list[str],
                       all_2026: list[tuple[str, str, int, int]]
                       ) -> list[tuple[str, str, int, int]]:
    """Filter the played 2026 results to those between members of `teams`."""
    s = set(teams)
    return [(h, a, hg, ag) for h, a, hg, ag in all_2026 if h in s and a in s]
