"""Step 9: tune Elo (K, home advantage, draw model) against historical RPS.

Grid-searches the Elo parameters on a chronological hold-out (the most recent
years) and reports the best configuration versus the current hand-set defaults,
so any change is justified by a measured RPS improvement rather than a guess.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, ELO_K, ELO_HOME_ADV  # noqa: E402
from src.ratings.elo_tuning import EloParams, evaluate_elo, tune_elo  # noqa: E402


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet").sort_values("date")
    # Evaluate on the most recent ~4 years of matches.
    eval_start = matches["date"].max() - pd.Timedelta(days=365 * 4)
    n_eval = int((matches["date"] >= eval_start).sum())
    print(f"Tuning Elo on {n_eval:,} hold-out matches since {eval_start.date()} ...\n")

    # Current defaults as a baseline (with reasonable draw-model values).
    base = EloParams(k=ELO_K, home_adv=ELO_HOME_ADV, d0=0.26, sigma=250)
    base_rps, _ = evaluate_elo(matches, base, eval_start)

    best, best_rps, results = tune_elo(matches, eval_start)

    print("=== Elo tuning (RPS, lower is better) ===")
    print(f"  Current defaults (K={base.k:.0f}, home={base.home_adv:.0f}): "
          f"{base_rps:.4f}")
    print(f"  Best found      (K={best.k:.0f}, home={best.home_adv:.0f}, "
          f"d0={best.d0:.2f}, sigma={best.sigma:.0f}): {best_rps:.4f}")
    gain = (base_rps - best_rps) / base_rps
    print(f"  Improvement: {gain:+.1%}\n")

    print("Top 5 configurations:")
    for p, rps in results[:5]:
        print(f"  RPS {rps:.4f} | K={p.k:.0f} home={p.home_adv:.0f} "
              f"d0={p.d0:.2f} sigma={p.sigma:.0f}")


if __name__ == "__main__":
    main()
