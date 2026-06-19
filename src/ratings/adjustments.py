"""Optional team-news rating adjustments (injuries, suspensions, recoveries).

Our datasets contain results, not squad availability, so injury/team-news
effects cannot be inferred automatically. Instead this module reads a small,
human-curated YAML file of Elo-point deltas per team and applies them to the
model's team-state before forecasting. This keeps the adjustment transparent and
auditable: a key player ruled out might be worth, say, -40 Elo; a star returning
from injury +30.

config/adjustments.yaml format:
    adjustments:
      Spain: -35        # key midfielder injured
      Portugal: 20      # talisman returns to full fitness
      France: -15
    note: free-text explaining the current adjustments

Deltas default to empty (no adjustment) when the file is absent.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.config import CONFIG_DIR


def load_adjustments(path: Path | None = None) -> dict[str, float]:
    """Load {team: elo_delta} from config/adjustments.yaml (empty if missing)."""
    path = path or (CONFIG_DIR / "adjustments.yaml")
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    adj = data.get("adjustments", {}) or {}
    return {str(k): float(v) for k, v in adj.items()}


def apply_to_states(states: dict, adjustments: dict[str, float]) -> dict:
    """Return a shallow-copied team-state dict with Elo deltas applied.

    `states` maps team -> object with an `.elo` attribute (TeamState). Only the
    Elo is shifted; recent-form fields are left untouched, so adjustments express
    "this squad is stronger/weaker than its results suggest right now".
    """
    import copy
    out = {}
    for team, st in states.items():
        s = copy.copy(st)
        if team in adjustments:
            s.elo = s.elo + adjustments[team]
        out[team] = s
    return out
