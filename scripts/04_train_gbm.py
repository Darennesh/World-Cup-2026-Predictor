"""Step 4: train the LightGBM challenger and compare it head-to-head against the
Dixon-Coles baseline on an identical temporal hold-out, then print SHAP-based
feature importances so the black box stays explainable.

The comparison is the whole point: more model flexibility only matters if it
lowers RPS out-of-sample. We evaluate both models on the same most-recent slice
of matches (no shuffling -> no leakage) and report the winner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.models.dixon_coles import DixonColes  # noqa: E402
from src.models.gbm import GBMModel, FEATURES  # noqa: E402
from src.evaluation.metrics import ranked_probability_score, log_loss_3way  # noqa: E402

HOLDOUT_FRAC = 0.15


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet").sort_values("date")
    feats = pd.read_parquet(PROCESSED_DIR / "features.parquet").sort_values("date")

    split = int(len(matches) * (1 - HOLDOUT_FRAC))
    train_m, test_m = matches.iloc[:split], matches.iloc[split:]
    train_f = feats.iloc[:split]

    # --- Baseline ---
    print("Fitting Dixon-Coles baseline ...")
    dc = DixonColes().fit(train_m)
    dc_known = set(dc.teams)

    # --- Challenger ---
    print("Fitting LightGBM challenger ...")
    gbm = GBMModel().fit(train_f, train_m)
    gbm_known = set(gbm.teams)

    # Evaluate on test matches both models can predict.
    known = dc_known & gbm_known
    test = test_m[test_m["home_team"].isin(known) & test_m["away_team"].isin(known)]
    outcomes = test["outcome"].to_numpy()

    # Dixon-Coles: per-match (cheap closed form). GBM: batched fast path.
    dc_probs = np.array([dc.predict_match(m.home_team, m.away_team, neutral=bool(m.neutral))
                         for m in test.itertuples(index=False)])
    gbm_probs = gbm.predict_matches(test[["home_team", "away_team", "neutral"]])

    base = np.bincount(train_m["outcome"], minlength=3) / len(train_m)
    base_probs = np.tile(base, (len(outcomes), 1))

    dc_rps = ranked_probability_score(dc_probs, outcomes)
    gbm_rps = ranked_probability_score(gbm_probs, outcomes)
    base_rps = ranked_probability_score(base_probs, outcomes)

    print(f"\nEvaluated on {len(outcomes):,} hold-out matches\n")
    print("=== RPS (lower is better) ===")
    print(f"  Base-rate  : {base_rps:.4f}")
    print(f"  DixonColes : {dc_rps:.4f}   ({(base_rps-dc_rps)/base_rps:+.1%} vs base)")
    print(f"  LightGBM   : {gbm_rps:.4f}   ({(base_rps-gbm_rps)/base_rps:+.1%} vs base)")
    winner = "LightGBM" if gbm_rps < dc_rps else "DixonColes"
    print(f"\n  >> Winner on RPS: {winner} "
          f"(gap {abs(gbm_rps-dc_rps):.4f})")

    print(f"\n  Log loss   : DC={log_loss_3way(dc_probs, outcomes):.4f}  "
          f"GBM={log_loss_3way(gbm_probs, outcomes):.4f}")

    # --- SHAP interpretability ---
    try:
        import shap
        print("\nComputing SHAP feature importances (home-goals model) ...")
        sample = train_f[FEATURES].sample(min(2000, len(train_f)), random_state=0)
        explainer = shap.TreeExplainer(gbm.home_model)
        sv = explainer.shap_values(sample)
        imp = np.abs(sv).mean(axis=0)
        order = np.argsort(imp)[::-1]
        print("  mean |SHAP|  feature")
        for i in order:
            print(f"  {imp[i]:9.4f}  {FEATURES[i]}")
    except Exception as exc:
        print(f"\n(SHAP skipped: {exc})")


if __name__ == "__main__":
    main()
