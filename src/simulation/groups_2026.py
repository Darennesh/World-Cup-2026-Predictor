"""Build the 2026 group layout for simulation.

Resolution order (most authoritative first):
  1. Real groups from config/tournament_2026.yaml, if the draw has been filled
     in (no TBD placeholders).
  2. Groups *inferred* from the data: once each group's full round-robin has
     been played, the teams that have all faced each other form connected
     components of size 4. This makes predictions exact as the tournament
     progresses, with zero manual input.
  3. Fallback: an Elo-seeded "snake draft" of all 48 qualified teams into 12
     balanced groups. This guarantees a complete, sensible 48-team bracket even
     when only partial group data exists (as in mid-group-stage), and is
     clearly labelled as a reconstructed draw.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
import yaml

from src.config import CONFIG_DIR
from src.ratings.bayesian import _is_2026_wc


def _config_groups() -> dict[str, list[str]] | None:
    cfg = yaml.safe_load((CONFIG_DIR / "tournament_2026.yaml").read_text(encoding="utf-8"))
    groups = {g: v["teams"] for g, v in cfg["groups"].items()}
    if any(any(str(t).startswith("TBD") for t in teams)
           for teams in groups.values()):
        return None
    return groups


def _infer_groups(matches: pd.DataFrame) -> dict[str, list[str]] | None:
    """Infer groups from completed round-robins via connected components.

    Two teams are linked if they met in the 2026 group stage. A fully played
    group of 4 yields a clique of 4 mutually linked teams. We only accept the
    inference when it produces exactly 12 components of size 4.
    """
    g = matches[_is_2026_wc(matches)]
    adj: dict[str, set[str]] = defaultdict(set)
    for m in g.itertuples(index=False):
        adj[m.home_team].add(m.away_team)
        adj[m.away_team].add(m.home_team)

    seen: set[str] = set()
    comps: list[list[str]] = []
    for team in adj:
        if team in seen:
            continue
        stack, comp = [team], []
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            comp.append(t)
            stack.extend(adj[t] - seen)
        comps.append(comp)

    if len(comps) == 12 and all(len(c) == 4 for c in comps):
        return {chr(65 + i): sorted(c) for i, c in enumerate(comps)}
    return None


def _snake_draft(matches: pd.DataFrame, elo) -> dict[str, list[str]]:
    """Distribute the 48 qualified teams into 12 groups by Elo snake draft."""
    teams = sorted(set(matches[_is_2026_wc(matches)]["home_team"])
                   | set(matches[_is_2026_wc(matches)]["away_team"]))
    ranked = sorted(teams, key=lambda t: elo.rating(t), reverse=True)

    n_groups = 12
    groups: dict[str, list[str]] = {chr(65 + i): [] for i in range(n_groups)}
    names = list(groups)
    # Snake order: 0..11, 11..0, 0..11, 11..0 -> balanced pots.
    for i, team in enumerate(ranked[: n_groups * 4]):
        rnd = i // n_groups
        pos = i % n_groups
        idx = pos if rnd % 2 == 0 else (n_groups - 1 - pos)
        groups[names[idx]].append(team)
    return groups


def resolve_groups(matches: pd.DataFrame, elo) -> tuple[dict[str, list[str]], str]:
    """Return (groups, source_label) using the best available method."""
    cfg = _config_groups()
    if cfg is not None:
        return cfg, "config draw"
    inferred = _infer_groups(matches)
    if inferred is not None:
        return inferred, "inferred from completed round-robin"
    return _snake_draft(matches, elo), "Elo-seeded snake draft (reconstructed)"


def group_stage_complete(matches: pd.DataFrame,
                         groups: dict[str, list[str]]) -> bool:
    """True if every group has played its full 6-game round-robin."""
    wc = matches[_is_2026_wc(matches)]
    played = {frozenset((h, a))
              for h, a in zip(wc["home_team"], wc["away_team"])}
    from itertools import combinations
    for teams in groups.values():
        for a, b in combinations(teams, 2):
            if frozenset((a, b)) not in played:
                return False
    return True


def actual_knockout_bracket(matches: pd.DataFrame,
                            groups: dict[str, list[str]],
                            seed: int = 2026) -> list[str] | None:
    """Build the real Round-of-32 bracket from completed group results.

    Returns the 32-team bracket order (using the official FIFA structure) once
    the group stage is complete, else None. The bracket is then locked into the
    simulation so knockout forecasts are conditioned on the actual qualifiers.
    """
    import numpy as np
    from src.simulation.standings import Group, select_best_thirds
    from src.simulation.bracket_2026 import build_bracket as build_bracket_2026
    from src.simulation.bracket_2026 import is_official_layout

    if not group_stage_complete(matches, groups):
        return None
    if not (is_official_layout(groups.keys()) and len(groups) == 12):
        return None

    wc = matches[_is_2026_wc(matches)]
    rng = np.random.default_rng(seed)
    winners, runners, third_group, thirds = {}, {}, {}, []
    for gn, teams in groups.items():
        g = Group(gn, teams)
        for m in wc.itertuples(index=False):
            if m.home_team in teams and m.away_team in teams:
                g.play(m.home_team, m.away_team,
                       int(m.home_goals), int(m.away_goals))
        table = g.standings(rng)
        winners[gn] = table[0].team
        runners[gn] = table[1].team
        third_group[table[2].team] = gn
        thirds.append(table[2])

    best = select_best_thirds(thirds, 8, rng)
    third_pairs = [(third_group[r.team], r.team) for r in best]
    bracket = build_bracket_2026(winners, runners, third_pairs)

    # FIFA's assignment of the 8 best thirds to R32 slots is combination-
    # dependent and can differ from our seeding. Once R32 is played, reconcile
    # each pairing with the actual game so the bracket matches reality (the
    # fixed winner/runner-up slots stay; only mis-slotted thirds are corrected).
    anchors = set(winners.values()) | set(runners.values())
    bracket = _reconcile_r32_with_actual(matches, groups, bracket, anchors)
    return bracket


def _reconcile_r32_with_actual(matches: pd.DataFrame,
                               groups: dict[str, list[str]],
                               bracket: list[str],
                               anchors: set[str]) -> list[str]:
    """Correct seeded R32 pairings to match the games actually played.

    Each team's real R32 opponent is its earliest knockout game. Where a seeded
    pair was never played (a third slotted differently than FIFA did), the
    anchor (group winner/runner-up, whose slot is fixed) keeps its place and the
    opposing slot is set to whoever the anchor actually faced.
    """
    tg = {t: g for g, teams in groups.items() for t in teams}
    wc = matches[_is_2026_wc(matches)].sort_values("date")

    # First knockout (cross-group) opponent for each bracket team = R32 rival.
    r32_opp: dict[str, str] = {}
    for m in wc.itertuples(index=False):
        gh, ga = tg.get(m.home_team), tg.get(m.away_team)
        if gh is None or ga is None or gh == ga:
            continue                                  # group-stage game
        for x, y in ((m.home_team, m.away_team), (m.away_team, m.home_team)):
            if x in bracket and x not in r32_opp:
                r32_opp[x] = y

    played_pairs = {frozenset((a, b))
                    for a, b in [(m.home_team, m.away_team)
                                 for m in wc.itertuples(index=False)]}

    result = list(bracket)
    for k in range(0, len(bracket), 2):
        a, b = result[k], result[k + 1]
        if frozenset((a, b)) in played_pairs:
            continue                                  # pairing already correct
        # Repair using the anchor's real opponent.
        if a in anchors and r32_opp.get(a):
            result[k + 1] = r32_opp[a]
        elif b in anchors and r32_opp.get(b):
            result[k] = r32_opp[b]
    return result


