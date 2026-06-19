"""Step 9: live forecast of the remaining group-stage matches.

For each group, prints the predicted result (win/draw/loss + expected score) of
every fixture not yet played, and each team's probability of finishing top and
of advancing (top two). Predictions reflect the model's blend of long-run
strength and current form, plus any team-news adjustments in
config/adjustments.yaml.

Usage:
    python scripts/09_group_forecast.py
    python scripts/09_group_forecast.py --sims 10000
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.models.gbm import GBMModel  # noqa: E402
from src.ratings.bayesian import fit_prior_then_update, group_results_2026  # noqa: E402
from src.ratings.adjustments import load_adjustments  # noqa: E402
from src.simulation.groups_2026 import resolve_groups  # noqa: E402
from src.simulation.group_forecast import (forecast_group,  # noqa: E402
                                           played_results_for)


def main() -> None:
    n_sims = 5000
    if "--sims" in sys.argv:
        n_sims = int(sys.argv[sys.argv.index("--sims") + 1])

    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    elo, _ = fit_prior_then_update(matches)
    model = GBMModel().fit(features, matches)

    adjustments = load_adjustments()
    if adjustments:
        model.apply_adjustments(adjustments)
        print(f"Applied team-news adjustments: {adjustments}\n")

    groups, source = resolve_groups(matches, elo)
    known = set(model.teams)
    groups = {g: [t for t in teams if t in known] for g, teams in groups.items()}
    groups = {g: teams for g, teams in groups.items() if len(teams) == 4}
    print(f"Group layout: {source}\n")

    all_2026 = group_results_2026(matches)

    for name, teams in groups.items():
        played = played_results_for(teams, all_2026)
        fc = forecast_group(model, name, teams, played, n_sims=n_sims)

        print(f"=== Group {name} ===")
        if fc.remaining:
            print("  Remaining fixtures (model prediction):")
            for m in fc.remaining:
                print(f"    {m.home:>16} {m.exp_home:.1f}-{m.exp_away:.1f} "
                      f"{m.away:<16}  W/D/L: {m.p_home:.0%}/{m.p_draw:.0%}/{m.p_away:.0%}")
        adv = sorted(fc.advance_prob.items(), key=lambda kv: kv[1], reverse=True)
        print("  Advance (top-2) probability:")
        for team, p in adv:
            win = fc.finish_first[team]
            print(f"    {team:<18} advance {p:5.0%}   |  win group {win:4.0%}")
        print()


if __name__ == "__main__":
    main()
