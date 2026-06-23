"""Empirically tune the Elo K-factor, home-advantage, and draw model to minimise
historical Ranked Probability Score (RPS).

The Elo constants (K, home advantage) were originally hand-set. This module
turns Elo into a proper 3-way (win/draw/loss) forecaster and grid-searches its
parameters against real historical outcomes, so the team-strength signal feeding
the rest of the system is optimised rather than guessed.

Converting Elo to a 3-way probability
-------------------------------------
Elo's expected score `e = 1 / (1 + 10**(-dr/400))` already equals
`p_home + 0.5 * p_draw` (a win counts 1, a draw 0.5). We add a draw model whose
mass peaks when teams are evenly matched and decays as the rating gap widens:

    p_draw = d0 * exp(-(dr / sigma)**2)
    p_home = e        - 0.5 * p_draw
    p_away = (1 - e)  - 0.5 * p_draw

This is internally consistent (probabilities sum to 1 and respect the Elo
expectation) and has just two extra tunable parameters, d0 and sigma, which is
robust to fit on limited data. We grid-search (K, home_adv, d0, sigma) to
minimise RPS on a chronological hold-out -- never peeking ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ratings.elo import EloModel, expected_score
from src.evaluation.metrics import ranked_probability_score


def elo_3way(dr: float, d0: float, sigma: float) -> tuple[float, float, float]:
    """Convert an effective rating difference to (p_home, p_draw, p_away)."""
    e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    p_draw = d0 * np.exp(-((dr / sigma) ** 2))
    p_home = e - 0.5 * p_draw
    p_away = (1.0 - e) - 0.5 * p_draw
    # Guard against tiny negatives in lopsided games, then renormalise.
    p = np.clip(np.array([p_home, p_draw, p_away]), 1e-6, None)
    p = p / p.sum()
    return float(p[0]), float(p[1]), float(p[2])


@dataclass
class EloParams:
    k: float
    home_adv: float
    d0: float
    sigma: float


def evaluate_elo(matches: pd.DataFrame, params: EloParams,
                 eval_start: pd.Timestamp) -> tuple[float, int]:
    """Walk-forward Elo: build ratings match by match, scoring every match on or
    after `eval_start`. Returns (mean RPS, n_eval)."""
    matches = matches.sort_values("date").reset_index(drop=True)
    elo = EloModel(k=params.k, home_adv=params.home_adv)

    probs, outcomes = [], []
    for m in matches.itertuples(index=False):
        if m.date >= eval_start:
            adv = 0.0 if bool(m.neutral) else params.home_adv
            dr = (elo.rating(m.home_team) + adv) - elo.rating(m.away_team)
            probs.append(elo_3way(dr, params.d0, params.sigma))
            outcomes.append(int(m.outcome))
        elo.update(m.home_team, m.away_team, m.home_goals, m.away_goals,
                   neutral=bool(m.neutral), importance=float(m.importance))

    if not outcomes:
        return float("inf"), 0
    return ranked_probability_score(np.array(probs), np.array(outcomes)), len(outcomes)


def tune_elo(matches: pd.DataFrame, eval_start: pd.Timestamp,
             k_grid=(20, 30, 40, 50, 60),
             home_grid=(40, 55, 65, 80, 100),
             d0_grid=(0.22, 0.26, 0.30),
             sigma_grid=(150, 250, 400)) -> tuple[EloParams, float, list]:
    """Grid-search Elo parameters to minimise hold-out RPS.

    Returns (best_params, best_rps, all_results) where all_results is a list of
    (EloParams, rps) sorted ascending by RPS.
    """
    results = []
    for k in k_grid:
        for h in home_grid:
            for d0 in d0_grid:
                for sigma in sigma_grid:
                    p = EloParams(k=k, home_adv=h, d0=d0, sigma=sigma)
                    rps, n = evaluate_elo(matches, p, eval_start)
                    results.append((p, rps))
    results.sort(key=lambda r: r[1])
    best, best_rps = results[0]
    return best, best_rps, results
