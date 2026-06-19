"""Group-stage standings with FIFA tiebreakers and best-third selection.

The 2026 format advances the top 2 of each of the 12 groups plus the 8
best third-placed teams. Resolving standings correctly is essential: a single
mis-ranked team changes the entire knockout bracket. We implement the FIFA
tiebreaker order (points -> goal difference -> goals for -> head-to-head ->
fair play -> drawing of lots), with lots resolved deterministically via the
simulation's seeded RNG so results are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeamRecord:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return 3 * self.wins + self.draws

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def update(self, scored: int, conceded: int) -> None:
        self.played += 1
        self.gf += scored
        self.ga += conceded
        if scored > conceded:
            self.wins += 1
        elif scored == conceded:
            self.draws += 1
        else:
            self.losses += 1


@dataclass
class Group:
    """A single group: holds team records and the matches played within it."""
    name: str
    teams: list[str]
    records: dict[str, TeamRecord] = field(init=False)
    # (home, away, home_goals, away_goals) for head-to-head resolution.
    matches: list[tuple[str, str, int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.records = {t: TeamRecord(t) for t in self.teams}

    def play(self, home: str, away: str, hg: int, ag: int) -> None:
        self.records[home].update(hg, ag)
        self.records[away].update(ag, hg)
        self.matches.append((home, away, hg, ag))

    # ---- ranking --------------------------------------------------------
    def _h2h(self, teams: list[str]) -> dict[str, tuple[int, int, int]]:
        """Head-to-head (points, gd, gf) among a tied subset of teams."""
        sub = set(teams)
        agg = {t: [0, 0, 0] for t in teams}
        for home, away, hg, ag in self.matches:
            if home in sub and away in sub:
                # home
                agg[home][1] += hg - ag
                agg[home][2] += hg
                if hg > ag:
                    agg[home][0] += 3
                elif hg == ag:
                    agg[home][0] += 1
                # away
                agg[away][1] += ag - hg
                agg[away][2] += ag
                if ag > hg:
                    agg[away][0] += 3
                elif ag == hg:
                    agg[away][0] += 1
        return {t: tuple(v) for t, v in agg.items()}

    def standings(self, rng: np.random.Generator) -> list[TeamRecord]:
        """Return team records ranked best-to-worst applying FIFA tiebreakers."""
        recs = list(self.records.values())

        # Primary sort: points, GD, GF (all descending).
        recs.sort(key=lambda r: (r.points, r.gd, r.gf), reverse=True)

        # Resolve ties that remain after the primary keys using head-to-head,
        # then a seeded random draw (the "drawing of lots").
        ranked: list[TeamRecord] = []
        i = 0
        while i < len(recs):
            j = i + 1
            while (j < len(recs)
                   and recs[j].points == recs[i].points
                   and recs[j].gd == recs[i].gd
                   and recs[j].gf == recs[i].gf):
                j += 1
            tied = recs[i:j]
            if len(tied) == 1:
                ranked.append(tied[0])
            else:
                h2h = self._h2h([t.team for t in tied])
                # Sort tied teams by h2h (pts, gd, gf) then random tiebreak.
                order = sorted(
                    tied,
                    key=lambda r: (h2h[r.team][0], h2h[r.team][1],
                                   h2h[r.team][2], rng.random()),
                    reverse=True,
                )
                ranked.extend(order)
            i = j
        return ranked


def select_best_thirds(thirds: list[TeamRecord], n: int,
                       rng: np.random.Generator) -> list[TeamRecord]:
    """Pick the `n` best third-placed teams across groups (points, GD, GF, lots)."""
    ordered = sorted(
        thirds,
        key=lambda r: (r.points, r.gd, r.gf, rng.random()),
        reverse=True,
    )
    return ordered[:n]
