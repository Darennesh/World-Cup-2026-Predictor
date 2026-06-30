"""Build a single, concrete "expected" knockout bracket with per-game win
probabilities for visualization.

The Monte Carlo engine answers "how often does each team reach round R" by
averaging over thousands of random brackets. For a *picture* of the tournament
we also want one representative bracket: who is expected to qualify, who plays
whom, and the model's predicted win probability for each specific tie, all the
way to the Final.

Approach:
  1. Estimate each group's expected standings analytically (expected points and
     goal difference from pairwise model probabilities) -> pick winners,
     runners-up, and the best thirds deterministically.
  2. Seed the bracket with the engine's own seeding function (so it matches the
     simulation's structure).
  3. Walk the bracket round by round. At each tie, record both teams and the
     model's P(win); advance the more likely team. This yields the single most
     likely path to the title plus a probability annotation on every game.

The advancing team is the favourite, so this is the modal (most-likely-at-each-
step) bracket, not a claim of certainty -- the Monte Carlo probabilities remain
the headline, and this bracket is their visual companion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from src.simulation.engine import Tournament, round_labels


@dataclass
class Game:
    round: str
    home: str
    away: str
    p_home: float        # model probability the home/first team wins the tie
    winner: str
    played: bool = False           # True once the real result is in the data
    actual: tuple | None = None    # (home_goals, away_goals) when played
    pens: tuple | None = None      # (home_pens, away_pens) if a shootout

    @property
    def p_winner(self) -> float:
        return self.p_home if self.winner == self.home else 1.0 - self.p_home


@dataclass
class ExpectedBracket:
    games_by_round: dict[str, list[Game]] = field(default_factory=dict)
    champion: str = ""
    group_standings: dict[str, list[str]] = field(default_factory=dict)

    def all_games(self) -> list[Game]:
        out = []
        for games in self.games_by_round.values():
            out.extend(games)
        return out


def _expected_group_order(tt: Tournament, teams: list[str]) -> list[str]:
    """Rank a group's teams by expected points then expected goal difference,
    derived from the model's pairwise score matrices (no sampling)."""
    pts = {t: 0.0 for t in teams}
    gd = {t: 0.0 for t in teams}
    for a, b in combinations(teams, 2):
        mat = tt.sampler.model.score_matrix(a, b, neutral=True)
        p_a = float(np.tril(mat, -1).sum())
        p_draw = float(np.trace(mat))
        p_b = float(np.triu(mat, 1).sum())
        pts[a] += 3 * p_a + p_draw
        pts[b] += 3 * p_b + p_draw
        # Expected goal difference contribution.
        gi = np.arange(mat.shape[0])
        exp_a = float((mat.sum(axis=1) * gi).sum())
        exp_b = float((mat.sum(axis=0) * gi).sum())
        gd[a] += exp_a - exp_b
        gd[b] += exp_b - exp_a
    return sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)


def _tie_probability(tt: Tournament, a: str, b: str) -> float:
    """Model probability that `a` wins a knockout tie vs `b` (draws split by the
    shootout weighting already encoded in the sampler)."""
    return tt.sampler.win_probability(a, b, neutral=True)


def build_expected_bracket(tt: Tournament, played: dict | None = None,
                           meta: dict | None = None) -> ExpectedBracket:
    """Compute the modal bracket and per-game probabilities for a Tournament.

    `played` maps an unordered team-pair frozenset to (home, away, hg, ag) for
    knockout ties already decided; `meta` carries shootout records. Where a tie
    is played, the real winner advances (not the model favourite), so the
    bracket tracks the actual tournament.
    """
    # Group stage complete: use the locked, real Round-of-32 bracket directly.
    if getattr(tt, "fixed_bracket", None) is not None:
        return _bracket_from_seed(tt, list(tt.fixed_bracket), played=played,
                                  meta=meta)

    winners, runners, thirds = {}, {}, []
    third_group: dict[str, str] = {}     # team -> group letter
    standings: dict[str, list[str]] = {}
    for name, teams in tt.group_defs.items():
        order = _expected_group_order(tt, teams)
        standings[name] = order
        winners[name] = order[0]
        runners[name] = order[1]
        if len(order) >= 3:
            thirds.append(order[2])
            third_group[order[2]] = name

    # Best thirds by the same expected-quality proxy: use posterior win prob vs
    # an average opponent is overkill; rank thirds by their group order strength
    # via expected points recomputed quickly.
    third_strength = {}
    for name, teams in tt.group_defs.items():
        order = standings[name]
        if len(order) >= 3:
            t = order[2]
            # crude strength: average model win prob vs the other 47 is costly;
            # use Elo-free proxy = win prob vs the group winner (lower = weaker).
            third_strength[t] = _tie_probability(tt, t, winners[name])
    best_thirds = sorted(thirds, key=lambda t: third_strength.get(t, 0.0),
                         reverse=True)[: tt.best_thirds]

    if tt.use_official_bracket:
        from src.simulation.bracket_2026 import build_bracket as _official
        bracket = _official(winners, runners,
                            [(third_group[t], t) for t in best_thirds])
    else:
        bracket = tt.seed_round_of_32(winners, runners, best_thirds)
    return _bracket_from_seed(tt, bracket, standings)


def _bracket_from_seed(tt: Tournament, bracket: list[str],
                       standings: dict[str, list[str]] | None = None,
                       played: dict | None = None,
                       meta: dict | None = None) -> ExpectedBracket:
    """Walk a fixed R32 seed order, recording per-tie win probabilities and
    advancing the favourite -- or the *actual* winner where a tie has been
    played -- to produce the bracket through to the Final."""
    from src.simulation.fixtures import knockout_winner
    played = played or {}
    labels = round_labels(len(bracket))   # e.g. [R32, R16, QF, SF, Final, Champion]

    games_by_round: dict[str, list[Game]] = {}
    current = bracket
    ri = 0
    # Each iteration plays the round whose name is labels[ri] (the round of
    # len(current) teams) and produces the winners that advance.
    while len(current) > 1:
        rnd = labels[ri]
        games = []
        nxt = []
        for k in range(0, len(current), 2):
            a, b = current[k], current[k + 1]
            p_a = _tie_probability(tt, a, b)
            res = played.get(frozenset((a, b)))
            if res is not None:
                hh, aa, hg, ag = res
                real = knockout_winner(hh, aa, hg, ag, meta)
                winner = real if real is not None else (a if p_a >= 0.5 else b)
                rec = (meta or {}).get("|".join(sorted([a, b])))
                pens = ((rec["pens_home"], rec["pens_away"])
                        if rec and "pens_home" in rec else None)
                # Orient the actual score to the (a, b) display order.
                actual = (hg, ag) if hh == a else (ag, hg)
                if pens and hh != a:
                    pens = (pens[1], pens[0])
                games.append(Game(round=rnd, home=a, away=b, p_home=p_a,
                                  winner=winner, played=True, actual=actual,
                                  pens=pens))
            else:
                winner = a if p_a >= 0.5 else b
                games.append(Game(round=rnd, home=a, away=b, p_home=p_a,
                                  winner=winner))
            nxt.append(winner)
        games_by_round[rnd] = games
        current = nxt
        ri += 1

    return ExpectedBracket(games_by_round=games_by_round,
                           champion=current[0],
                           group_standings=standings or {})
