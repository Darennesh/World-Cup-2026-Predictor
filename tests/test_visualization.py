"""Tests for the expected-bracket builder and visualization plumbing."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.sampler import ScoreSampler
from src.simulation.engine import Tournament
from src.simulation.bracket import build_expected_bracket
from src.visualization.plots import (plot_bracket, plot_champion_bar,
                                     plot_round_heatmap)


class FixedModel:
    def __init__(self, order):
        self.strength = {t: len(order) - i for i, t in enumerate(order)}

    def score_matrix(self, home, away, neutral=True):
        from scipy.special import gammaln
        lam_h = 0.8 + 0.25 * self.strength[home]
        lam_a = 0.8 + 0.25 * self.strength[away]
        g = np.arange(8)
        ph = np.exp(g * np.log(lam_h) - lam_h - gammaln(g + 1))
        pa = np.exp(g * np.log(lam_a) - lam_a - gammaln(g + 1))
        m = np.outer(ph, pa)
        return m / m.sum()


def _tt():
    groups = {chr(65 + i): [f"{chr(65+i)}{k}" for k in range(4)] for i in range(12)}
    order = [t for ts in groups.values() for t in ts]
    return Tournament(groups, ScoreSampler(FixedModel(order)))


def test_bracket_has_expected_rounds_and_champion():
    bracket = build_expected_bracket(_tt())
    assert bracket.champion != ""
    # 32-team knockout -> rounds R32, R16, QF, SF, Final.
    assert list(bracket.games_by_round.keys()) == ["R32", "R16", "QF", "SF", "Final"]
    # R32 has 16 games, halving each round down to the Final.
    assert len(bracket.games_by_round["R32"]) == 16
    assert len(bracket.games_by_round["Final"]) == 1


def test_every_game_has_valid_probability():
    bracket = build_expected_bracket(_tt())
    for g in bracket.all_games():
        assert 0.0 <= g.p_home <= 1.0
        assert g.winner in (g.home, g.away)
        # Winner must be the side with >= 50% (favourite advances).
        assert g.p_winner >= 0.5 - 1e-9


def test_plots_write_files(tmp_path: Path):
    tt = _tt()
    bracket = build_expected_bracket(tt)
    pred = tt.run(n_sims=150, seed=1)

    b = plot_bracket(bracket, tmp_path / "bracket.png")
    c = plot_champion_bar(pred, tmp_path / "bar.png")
    h = plot_round_heatmap(pred, tmp_path / "heat.png")
    assert b.exists() and b.stat().st_size > 0
    assert c.exists() and c.stat().st_size > 0
    assert h.exists() and h.stat().st_size > 0
