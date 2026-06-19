"""Step 2: fit the Dixon-Coles baseline and report its RPS on a temporal
hold-out, compared against a naive base-rate forecast.

Temporal validation (train on the past, test on the most recent slice) is the
only honest way to evaluate a forecaster -- random shuffling would leak future
information. The base-rate comparison tells us whether the model adds value over
simply predicting the historical home/draw/away frequencies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.models.dixon_coles import DixonColes  # noqa: E402
from src.evaluation.metrics import ranked_probability_score, log_loss_3way  # noqa: E402

import pandas as pd  # noqa: E402

HOLDOUT_FRAC = 0.15  # most recent 15% of matches = test set


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet").sort_values("date")
    split = int(len(matches) * (1 - HOLDOUT_FRAC))
    train, test = matches.iloc[:split], matches.iloc[split:]
    print(f"Train: {len(train):,} matches | Test: {len(test):,} matches")

    model = DixonColes().fit(train)

    # Only evaluate test matches whose teams were seen in training.
    known = set(model.teams)
    test = test[test["home_team"].isin(known) & test["away_team"].isin(known)]

    probs, outcomes = [], []
    for m in test.itertuples(index=False):
        probs.append(model.predict_match(m.home_team, m.away_team, neutral=bool(m.neutral)))
        outcomes.append(int(m.outcome))
    probs = np.array(probs)
    outcomes = np.array(outcomes)

    # Naive base-rate forecast from the training distribution.
    base = np.bincount(train["outcome"], minlength=3) / len(train)
    base_probs = np.tile(base, (len(outcomes), 1))

    dc_rps = ranked_probability_score(probs, outcomes)
    base_rps = ranked_probability_score(base_probs, outcomes)

    print("\n=== RPS (lower is better) ===")
    print(f"  Base-rate : {base_rps:.4f}")
    print(f"  DixonColes: {dc_rps:.4f}   ({(base_rps - dc_rps) / base_rps:+.1%} vs base)")
    print(f"\n  Log loss  : DC={log_loss_3way(probs, outcomes):.4f}  "
          f"base={log_loss_3way(base_probs, outcomes):.4f}")

    print("\nTop teams by attack rating:")
    print(model.ratings_table().head(8).to_string(index=False))


if __name__ == "__main__":
    main()
