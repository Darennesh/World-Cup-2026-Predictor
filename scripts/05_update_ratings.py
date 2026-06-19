"""Step 5: apply Bayesian rating updates from the 2026 group stage and report
how each team's strength shifted relative to its pre-tournament prior.

This quantifies what the live group results tell us: which teams have over- or
under-performed expectations so far, and therefore enter the knockout
simulation stronger or weaker than their long-run reputation alone would imply.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.ratings.bayesian import fit_prior_then_update  # noqa: E402


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    elo, update = fit_prior_then_update(matches)

    print(f"Applied {update.n_group_matches} played 2026 group-stage matches.\n")

    shifts = update.shifts()
    # Restrict to teams actually in the 2026 tournament for a clean view.
    wc_teams = set(shifts["team"])  # already only updated teams move
    movers = shifts[shifts["delta"].abs() > 1e-6].head(20)

    print("Biggest rating movers (prior -> posterior):")
    with pd.option_context("display.float_format", lambda v: f"{v:8.1f}"):
        print(movers.to_string(index=False))

    out = PROCESSED_DIR / "ratings_posterior.csv"
    pd.DataFrame({"team": list(update.posterior.keys()),
                  "rating": list(update.posterior.values())}) \
        .sort_values("rating", ascending=False) \
        .to_csv(out, index=False)
    print(f"\nSaved posterior ratings to {out}")


if __name__ == "__main__":
    main()
