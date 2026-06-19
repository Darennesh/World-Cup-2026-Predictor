"""Step 6: back-test the models on the 2018 and 2022 World Cups and fit
probability calibration.

For each past edition we train only on prior data, predict the tournament's
matches, and report RPS (baseline vs challenger vs base rate). We then fit a
temperature calibrator on 2018 and confirm on 2022 that it improves -- or at
least does not harm -- calibration on an unseen edition. The chosen temperature
is saved for use when generating the 2026 predictions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.evaluation.backtest import backtest_year  # noqa: E402
from src.evaluation.calibration import TemperatureCalibrator  # noqa: E402


def _fmt(r):
    return (f"  {r.year}: {r.n_matches:>2} matches | "
            f"base {r.base_rps:.4f} | DC {r.dc_rps:.4f} | GBM {r.gbm_rps:.4f}")


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    print("Back-testing past World Cups (train on prior data only) ...\n")
    results = {}
    for year in (2018, 2022):
        try:
            results[year] = backtest_year(matches, features, year)
            print(_fmt(results[year]))
        except ValueError as exc:
            print(f"  {year}: skipped ({exc})")

    if 2018 not in results or 2022 not in results:
        print("\nNeed both 2018 and 2022 editions to fit/validate calibration.")
        return

    # --- RPS summary ---
    print("\n=== RPS summary (lower is better) ===")
    avg_dc = np.mean([results[y].dc_rps for y in results])
    avg_gbm = np.mean([results[y].gbm_rps for y in results])
    avg_base = np.mean([results[y].base_rps for y in results])
    print(f"  Avg base : {avg_base:.4f}")
    print(f"  Avg DC   : {avg_dc:.4f}   ({(avg_base-avg_dc)/avg_base:+.1%})")
    print(f"  Avg GBM  : {avg_gbm:.4f}   ({(avg_base-avg_gbm)/avg_base:+.1%})")
    print(f"  Best model: {'GBM' if avg_gbm < avg_dc else 'DixonColes'}")

    # --- Calibration: fit on 2018 (GBM), validate on 2022 ---
    print("\n=== Calibration (temperature scaling, GBM) ===")
    cal = TemperatureCalibrator().fit(results[2018].gbm_probs,
                                      results[2018].outcomes)
    val = cal.report(results[2022].gbm_probs, results[2022].outcomes)
    print(f"  Fitted T on 2018: {val['T']:.3f}")
    print(f"  2022 RPS     : {val['rps_before']:.4f} -> {val['rps_after']:.4f}")
    print(f"  2022 log loss: {val['logloss_before']:.4f} -> {val['logloss_after']:.4f}")

    # Only adopt calibration if it actually improves the *unseen* edition.
    # Otherwise the model is already well-calibrated and T=1.0 is safer.
    if val["rps_after"] < val["rps_before"]:
        chosen_T = cal.T
        print(f"  -> Calibration helps on 2022; adopting T={chosen_T:.3f}.")
    else:
        chosen_T = 1.0
        print("  -> Calibration did not improve unseen 2022; keeping T=1.0 "
              "(model already well-calibrated).")

    out = PROCESSED_DIR / "calibration.json"
    out.write_text(json.dumps({"temperature": chosen_T}, indent=2))
    print(f"\nSaved calibration temperature to {out}")


if __name__ == "__main__":
    main()
