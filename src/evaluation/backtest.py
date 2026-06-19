"""Back-testing: validate the models against *past* World Cups.

The honest test of a tournament forecaster is to pretend we are standing just
before a past World Cup, train only on data available then, predict that
tournament's matches, and score the predictions against what actually happened.
We do this for 2018 and 2022 (and any other edition in the data), reporting RPS
for the baseline and the challenger versus a naive base rate.

A held-out tournament is also the right place to *fit calibration*: we use one
edition to estimate the temperature and another to confirm it generalises,
avoiding fitting and evaluating calibration on the same matches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColes
from src.models.gbm import GBMModel
from src.evaluation.metrics import ranked_probability_score, log_loss_3way


def wc_edition(matches: pd.DataFrame, year: int) -> pd.DataFrame:
    """All FIFA World Cup (finals, not qualifiers) matches in a given year."""
    m = matches
    mask = (
        (m["date"].dt.year == year)
        & m["tournament"].str.contains("World Cup", case=False, na=False)
        & ~m["tournament"].str.contains("qual", case=False, na=False)
    )
    return m[mask].sort_values("date")


@dataclass
class BacktestResult:
    year: int
    n_matches: int
    base_rps: float
    dc_rps: float
    gbm_rps: float
    dc_probs: np.ndarray
    gbm_probs: np.ndarray
    outcomes: np.ndarray


def backtest_year(matches: pd.DataFrame, features: pd.DataFrame,
                  year: int) -> BacktestResult:
    """Train on everything before `year`'s World Cup; predict that edition."""
    edition = wc_edition(matches, year)
    if edition.empty:
        raise ValueError(f"No World Cup matches found for {year}.")
    start = edition["date"].min()

    train_m = matches[matches["date"] < start]
    train_f = features[features["date"] < start]

    dc = DixonColes().fit(train_m)
    gbm = GBMModel().fit(train_f, train_m)

    known = set(dc.teams) & set(gbm.teams)
    test = edition[edition["home_team"].isin(known) & edition["away_team"].isin(known)]
    outcomes = test["outcome"].to_numpy()

    dc_probs = np.array([dc.predict_match(t.home_team, t.away_team, neutral=bool(t.neutral))
                         for t in test.itertuples(index=False)])
    gbm_probs = gbm.predict_matches(test[["home_team", "away_team", "neutral"]])

    base = np.bincount(train_m["outcome"], minlength=3) / len(train_m)
    base_probs = np.tile(base, (len(outcomes), 1))

    return BacktestResult(
        year=year, n_matches=len(test),
        base_rps=ranked_probability_score(base_probs, outcomes),
        dc_rps=ranked_probability_score(dc_probs, outcomes),
        gbm_rps=ranked_probability_score(gbm_probs, outcomes),
        dc_probs=dc_probs, gbm_probs=gbm_probs, outcomes=outcomes,
    )
