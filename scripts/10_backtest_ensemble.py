"""Step 10: back-test the Dixon-Coles + LightGBM ensemble against each base
model, with the blend weight tuned out-of-sample.

To keep the comparison honest we use a three-way temporal split:
  * train      -- fit both base models (oldest data),
  * validation -- tune the ensemble weight to minimise RPS,
  * test       -- final hold-out, never seen during fitting or tuning.

Tuning the weight on a separate slice (not the test set) avoids the subtle
leakage of picking the blend that happens to look best on the evaluation data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.models.dixon_coles import DixonColes  # noqa: E402
from src.models.gbm import GBMModel  # noqa: E402
from src.models.ensemble import EnsembleModel, tune_weight  # noqa: E402
from src.evaluation.metrics import (ranked_probability_score,  # noqa: E402
                                    log_loss_3way)

VAL_FRAC = 0.15      # middle slice for weight tuning
TEST_FRAC = 0.15     # most-recent slice for final evaluation


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet").sort_values("date")
    feats = pd.read_parquet(PROCESSED_DIR / "features.parquet").sort_values("date")

    n = len(matches)
    test_start = int(n * (1 - TEST_FRAC))
    val_start = int(n * (1 - TEST_FRAC - VAL_FRAC))

    train_m, val_m, test_m = (matches.iloc[:val_start],
                              matches.iloc[val_start:test_start],
                              matches.iloc[test_start:])
    train_f = feats.iloc[:val_start]
    print(f"Train {len(train_m):,} | Val {len(val_m):,} | Test {len(test_m):,}")

    print("Fitting Dixon-Coles and LightGBM on the training slice ...")
    dc = DixonColes().fit(train_m)
    gbm = GBMModel().fit(train_f, train_m)

    print("Tuning ensemble weight on the validation slice ...")
    w, val_rps = tune_weight(dc, gbm, val_m)
    print(f"  Best weight (mass on LightGBM): {w:.2f}  (val RPS {val_rps:.4f})")
    ens = EnsembleModel(dc=dc, gbm=gbm, weight=w)

    # Evaluate all three on the untouched test slice.
    known = set(dc.teams) & set(gbm.teams)
    test = test_m[test_m["home_team"].isin(known) & test_m["away_team"].isin(known)]
    outcomes = test["outcome"].to_numpy(dtype=int)

    dc_p, gbm_p, ens_p = [], [], []
    for m in test.itertuples(index=False):
        nb = bool(m.neutral)
        dc_p.append(dc.predict_match(m.home_team, m.away_team, neutral=nb))
        gbm_p.append(gbm.predict_match(m.home_team, m.away_team, neutral=nb))
        ens_p.append(ens.predict_match(m.home_team, m.away_team, neutral=nb))
    dc_p, gbm_p, ens_p = np.array(dc_p), np.array(gbm_p), np.array(ens_p)

    base = np.bincount(train_m["outcome"], minlength=3) / len(train_m)
    base_p = np.tile(base, (len(outcomes), 1))

    def line(name, p):
        rps = ranked_probability_score(p, outcomes)
        ll = log_loss_3way(p, outcomes)
        skill = (ranked_probability_score(base_p, outcomes) - rps) \
            / ranked_probability_score(base_p, outcomes)
        return f"  {name:<12} RPS {rps:.4f}  | log loss {ll:.4f}  | skill {skill:+.1%}"

    print(f"\nEvaluated on {len(outcomes):,} hold-out matches\n")
    print("=== RPS (lower is better) ===")
    print(line("Base rate", base_p))
    print(line("DixonColes", dc_p))
    print(line("LightGBM", gbm_p))
    print(line("Ensemble", ens_p))

    rps = {n: ranked_probability_score(p, outcomes)
           for n, p in [("DixonColes", dc_p), ("LightGBM", gbm_p),
                        ("Ensemble", ens_p)]}
    winner = min(rps, key=rps.get)
    print(f"\n  >> Best model on RPS: {winner} ({rps[winner]:.4f})")


if __name__ == "__main__":
    main()
