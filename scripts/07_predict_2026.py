"""Step 7: generate the final 2026 World Cup knockout-stage predictions.

Pipeline:
  1. Fit the validated LightGBM match model on ALL data through the latest
     available date -- so the 2026 group results already played are baked into
     every team's state (this is the Bayesian "learn from the group stage"
     signal in action).
  2. Resolve the 12 groups (config draw / inferred / Elo snake-draft fallback).
  3. Run a full 48-team Monte Carlo simulation of the knockout bracket.
  4. Report each team's probability of reaching R32 -> R16 -> QF -> SF ->
     Final -> Champion.

Usage:
    python scripts/07_predict_2026.py                # 50k sims
    python scripts/07_predict_2026.py --sims 100000
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.models.gbm import GBMModel  # noqa: E402
from src.ratings.bayesian import fit_prior_then_update  # noqa: E402
from src.simulation.sampler import ScoreSampler  # noqa: E402
from src.simulation.engine import Tournament  # noqa: E402
from src.simulation.groups_2026 import resolve_groups  # noqa: E402


def main() -> None:
    n_sims = 50_000
    if "--sims" in sys.argv:
        n_sims = int(sys.argv[sys.argv.index("--sims") + 1])

    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    # Posterior Elo (prior + 2026 group games) -> used for group seeding.
    elo, update = fit_prior_then_update(matches)
    print(f"Posterior ratings include {update.n_group_matches} played "
          f"2026 group matches.")

    print("Fitting LightGBM match model on all data through "
          f"{matches['date'].max().date()} ...")
    model = GBMModel().fit(features, matches)

    groups, source = resolve_groups(matches, elo)
    print(f"Group layout: {source}\n")
    # Keep only teams the model can rate.
    known = set(model.teams)
    groups = {g: [t for t in teams if t in known] for g, teams in groups.items()}
    groups = {g: teams for g, teams in groups.items() if len(teams) == 4}

    print("Groups:")
    for g, teams in groups.items():
        print(f"  {g}: {', '.join(teams)}")
    print()

    sampler = ScoreSampler(model)
    tt = Tournament(groups, sampler)
    print(f"Running {n_sims:,} tournament simulations ...\n")
    df = tt.run(n_sims=n_sims)

    pred_cols = [c for c in df.columns if c.startswith("P_")]
    pd.set_option("display.float_format", lambda v: f"{v:.1%}")
    print("=== 2026 World Cup knockout-stage probabilities ===")
    print(df[["team"] + pred_cols].head(24).to_string(index=False))

    out = PROCESSED_DIR / "predictions_2026.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved full predictions ({len(df)} teams) to {out}")

    champ = df.iloc[0]
    print(f"\nMost likely champion: {champ['team']} "
          f"({champ[pred_cols[-1]]:.1%})")


if __name__ == "__main__":
    main()
