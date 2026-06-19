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
