"""Step 11: back-test the Negative Binomial goal model against the Poisson-based
Dixon-Coles baseline on a temporal hold-out.

The NB model shares Dixon-Coles' attack/defence means but adds a dispersion
parameter for over-dispersion (blowouts). This compares whether modelling that
extra variance lowers RPS out-of-sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.models.dixon_coles import DixonColes  # noqa: E402
from src.models.negbinom import NegativeBinomialModel  # noqa: E402
from src.evaluation.metrics import (ranked_probability_score,  # noqa: E402
                                    log_loss_3way)

HOLDOUT_FRAC = 0.15


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet").sort_values("date")
    split = int(len(matches) * (1 - HOLDOUT_FRAC))
    train, test = matches.iloc[:split], matches.iloc[split:]
    print(f"Train {len(train):,} | Test {len(test):,}")

    print("Fitting Dixon-Coles (Poisson) and Negative Binomial ...")
    dc = DixonColes().fit(train)
    nb = NegativeBinomialModel().fit(train)
    print(f"  NB dispersion r = {nb.r:.1f}  (large r -> Poisson limit)")

    known = set(dc.teams)
    test = test[test["home_team"].isin(known) & test["away_team"].isin(known)]
    outcomes = test["outcome"].to_numpy(dtype=int)

    dc_p, nb_p = [], []
    for m in test.itertuples(index=False):
        nb_neutral = bool(m.neutral)
        dc_p.append(dc.predict_match(m.home_team, m.away_team, neutral=nb_neutral))
        nb_p.append(nb.predict_match(m.home_team, m.away_team, neutral=nb_neutral))
    dc_p, nb_p = np.array(dc_p), np.array(nb_p)

    base = np.bincount(train["outcome"], minlength=3) / len(train)
    base_p = np.tile(base, (len(outcomes), 1))
    base_rps = ranked_probability_score(base_p, outcomes)

    def line(name, p):
        rps = ranked_probability_score(p, outcomes)
        return (f"  {name:<16} RPS {rps:.4f}  | log loss {log_loss_3way(p, outcomes):.4f}"
                f"  | skill {(base_rps - rps) / base_rps:+.1%}")

    print(f"\nEvaluated on {len(outcomes):,} hold-out matches\n")
    print("=== RPS (lower is better) ===")
    print(line("Base rate", base_p))
    print(line("DixonColes/Poisson", dc_p))
    print(line("NegativeBinomial", nb_p))


if __name__ == "__main__":
    main()
