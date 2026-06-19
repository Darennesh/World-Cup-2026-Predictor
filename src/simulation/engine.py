"""Monte Carlo tournament simulation engine for the 2026 World Cup.

Knockout tournaments are *path-dependent*: who a team meets in the Round of 32
depends on group results, so the only sound way to estimate "Team X reaches the
quarterfinal" is to simulate the whole bracket many thousands of times and count
how often each team survives to each round.

Pipeline per simulation:
  1. Play every group's round-robin (sample scorelines from the model).
  2. Rank each group (FIFA tiebreakers) -> winners, runners-up, thirds.
  3. Take the 8 best third-placed teams -> 32 qualifiers.
  4. Seed the Round of 32 and play single-elimination to the Final, resolving
     draws via a win-probability-weighted shootout.
  5. Record how far each team advanced.

Aggregated over N simulations, this yields calibrated probabilities for every
team reaching R32 / R16 / QF / SF / Final / Champion.

NOTE on bracket seeding: the official 2026 R32 slotting table (which third-place
groups feed which slots) is complex and combination-dependent. We use a
deterministic, balanced default below; replace `seed_round_of_32` with the
official table once finalised for exact realism.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from src.simulation.sampler import ScoreSampler
from src.simulation.standings import Group, TeamRecord, select_best_thirds

# Standard round labels keyed by the number of teams still in the bracket.
_ROUND_BY_SIZE = {64: "R64", 32: "R32", 16: "R16", 8: "QF", 4: "SF",
                  2: "Final", 1: "Champion"}
# Default labels for the full 2026 format (32-team knockout).
ROUNDS = ["R32", "R16", "QF", "SF", "Final", "Champion"]


def round_labels(bracket_size: int) -> list[str]:
    """Round labels from R<bracket_size> down to 'Champion' for any 2**k size."""
    labels, size = [], bracket_size
    while size >= 1:
        labels.append(_ROUND_BY_SIZE.get(size, f"R{size}"))
        if size == 1:
            break
        size //= 2
    return labels


class Tournament:
    """Holds the group definitions and runs Monte Carlo simulations."""

    def __init__(self, groups: dict[str, list[str]], sampler: ScoreSampler,
                 best_thirds: int = 8):
        self.group_defs = groups
        self.sampler = sampler
        self.all_teams = [t for teams in groups.values() for t in teams]
        # Adjust the number of qualifying thirds so the knockout bracket size
        # (2*groups + thirds) is a power of two. For the full 2026 format
        # (12 groups, 8 thirds) this leaves 8 unchanged -> a 32-team bracket.
        self.best_thirds = self._fit_thirds(len(groups), best_thirds)

    @staticmethod
    def _fit_thirds(n_groups: int, want: int) -> int:
        base = 2 * n_groups
        total = base + min(want, n_groups)
        p = 1 << (total.bit_length() - 1)   # largest power of two <= total
        return max(p - base, 0)

    # ---- group stage ----------------------------------------------------
    def _play_group(self, name: str, teams: list[str],
                    rng: np.random.Generator) -> Group:
        g = Group(name, teams)
        for home, away in combinations(teams, 2):  # single round-robin
            hg, ag = self.sampler.sample(home, away, rng, neutral=True)
            g.play(home, away, hg, ag)
        return g

    # ---- knockout -------------------------------------------------------
    def _knockout_match(self, a: str, b: str, rng: np.random.Generator) -> str:
        """Return the winner; resolve draws by a weighted shootout."""
        hg, ag = self.sampler.sample(a, b, rng, neutral=True)
        if hg > ag:
            return a
        if ag > hg:
            return b
        p = self.sampler.win_probability(a, b, neutral=True)
        return a if rng.random() < p else b

    @staticmethod
    def seed_round_of_32(winners: dict[str, str], runners: dict[str, str],
                         thirds: list[str]) -> list[str]:
        """Build the knockout bracket order (deterministic, balanced default).

        Works for any number of groups G as long as the total qualifiers
        (2*G + len(thirds)) is a power of two. For the full 2026 format that is
        12 winners + 12 runners-up + 8 best thirds = 32. Group winners are the
        higher seeds: each winner is paired with a third-placed team first, then
        with a runner-up; any remaining runners-up form all-runner-up matches.
        Same-group first-round clashes are avoided. Pairs are (0,1), (2,3), ...
        """
        gnames = sorted(winners.keys())
        W = [winners[g] for g in gnames]
        R = [runners[g] for g in gnames]
        T = list(thirds)

        n_runner_opp = len(W) - len(T)            # winners not facing a third
        if n_runner_opp < 0:
            T = T[:len(W)]
            n_runner_opp = 0
        # Offset runner-up opponents by half the field to dodge same-group ties.
        off = len(R) // 2
        runner_opps = [R[(i + off) % len(R)] for i in range(n_runner_opp)]
        winner_opponents = T + runner_opps
        used = set(winner_opponents)
        rest_runners = [r for r in R if r not in used]

        bracket: list[str] = []
        for i in range(len(W)):
            bracket.append(W[i])
            bracket.append(winner_opponents[i])
        bracket.extend(rest_runners)
        return bracket

    def _run_knockout(self, bracket: list[str],
                      rng: np.random.Generator) -> dict[str, str]:
        """Play single-elimination; return {team: furthest round reached}."""
        labels = round_labels(len(bracket))   # e.g. [R32, R16, QF, SF, Final, Champion]
        reached = {t: labels[0] for t in bracket}
        current = bracket
        ri = 1
        while len(current) > 1:
            winners = []
            for k in range(0, len(current), 2):
                w = self._knockout_match(current[k], current[k + 1], rng)
                winners.append(w)
                reached[w] = labels[ri]
            current = winners
            ri += 1
        return reached

    # ---- single simulation ---------------------------------------------
    def simulate_once(self, rng: np.random.Generator) -> dict[str, str]:
        winners, runners, thirds_records = {}, {}, []
        for name, teams in self.group_defs.items():
            g = self._play_group(name, teams, rng)
            table = g.standings(rng)
            winners[name] = table[0].team
            runners[name] = table[1].team
            if len(table) >= 3:
                thirds_records.append(table[2])

        best = select_best_thirds(thirds_records, self.best_thirds, rng)
        bracket = self.seed_round_of_32(winners, runners,
                                        [r.team for r in best])
        return self._run_knockout(bracket, rng)

    # ---- Monte Carlo ----------------------------------------------------
    def run(self, n_sims: int = 50_000, seed: int = 2026):
        """Run N simulations; return a DataFrame of per-team round probabilities."""
        import pandas as pd

        # Determine bracket size / round labels once from a dry structural pass.
        n_groups = len(self.group_defs)
        n_thirds = min(self.best_thirds, n_groups)
        bracket_size = 2 * n_groups + n_thirds
        labels = round_labels(bracket_size)
        rank = {r: i for i, r in enumerate(labels)}

        counts = {t: defaultdict(int) for t in self.all_teams}
        rng = np.random.default_rng(seed)

        for _ in range(n_sims):
            reached = self.simulate_once(rng)
            for team, rnd in reached.items():
                # Count this team as having reached `rnd` AND every earlier round.
                for r in labels[:rank[rnd] + 1]:
                    counts[team][r] += 1

        rows = []
        for team in self.all_teams:
            row = {"team": team}
            for r in labels:
                row[f"P_{r}"] = counts[team][r] / n_sims
            rows.append(row)
        champ_col = f"P_{labels[-1]}"
        df = pd.DataFrame(rows).sort_values(champ_col, ascending=False)
        return df.reset_index(drop=True)
