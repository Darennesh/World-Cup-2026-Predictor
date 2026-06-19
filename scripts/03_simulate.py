"""Step 3: run a full Monte Carlo tournament simulation.

Fits the Dixon-Coles model on the processed matches, wraps it in a score
sampler, loads the 2026 group layout from config (falling back to a demo set of
real nations if the official draw is still TBD), and prints each team's
probability of reaching every knockout round.

Usage:
    python scripts/03_simulate.py                 # 50k sims, demo/real groups
    python scripts/03_simulate.py --sims 10000
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, CONFIG_DIR  # noqa: E402
from src.models.dixon_coles import DixonColes  # noqa: E402
from src.simulation.sampler import ScoreSampler  # noqa: E402
from src.simulation.engine import Tournament  # noqa: E402

import pandas as pd  # noqa: E402

# Fallback groups using real nations present in typical datasets, so the demo
# produces meaningful names until the official 2026 draw is filled into config.
# 16 unique teams -> 4 groups of 4 -> a clean 8-team knockout (QF onward).
DEMO_GROUPS = {
    "A": ["Brazil", "Croatia", "Japan", "Senegal"],
    "B": ["Canada", "Spain", "Morocco", "Uruguay"],
    "C": ["USA", "England", "Netherlands", "Germany"],
    "D": ["France", "Portugal", "Argentina", "Mexico"],
}


def load_groups() -> dict[str, list[str]]:
    cfg = yaml.safe_load((CONFIG_DIR / "tournament_2026.yaml").read_text(encoding="utf-8"))
    groups = {g: v["teams"] for g, v in cfg["groups"].items()}
    # If any group still has TBD placeholders, use the demo set instead.
    if any(any(str(t).startswith("TBD") for t in teams)
           for teams in groups.values()):
        print("Config groups contain TBD placeholders -> using DEMO groups.\n")
        return DEMO_GROUPS
    return groups


def main() -> None:
    n_sims = 50_000
    if "--sims" in sys.argv:
        n_sims = int(sys.argv[sys.argv.index("--sims") + 1])

    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    print(f"Fitting Dixon-Coles on {len(matches):,} matches ...")
    model = DixonColes().fit(matches)

    groups = load_groups()
    # Keep only teams the model has ratings for (others can't be simulated).
    known = set(model.teams)
    groups = {g: [t for t in teams if t in known] for g, teams in groups.items()}
    groups = {g: teams for g, teams in groups.items() if len(teams) == 4}
    if not groups:
        print("No fully-known groups to simulate. Add real results data first.")
        return
    print(f"Simulating {len(groups)} group(s) x {n_sims:,} runs ...\n")

    sampler = ScoreSampler(model)
    tt = Tournament(groups, sampler)
    df = tt.run(n_sims=n_sims)

    pd.set_option("display.float_format", lambda v: f"{v:.1%}")
    print(df.to_string(index=False))

    out = PROCESSED_DIR / "simulation_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved probabilities to {out}")


if __name__ == "__main__":
    main()
