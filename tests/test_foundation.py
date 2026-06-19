"""Smoke test: verifies the foundation (Elo + RPS metric) works end to end.

Run with:  pytest tests/test_foundation.py
"""
import numpy as np

from src.ratings.elo import EloModel, expected_score
from src.evaluation.metrics import ranked_probability_score, log_loss_3way


def test_expected_score_symmetry():
    # Equal ratings, neutral venue -> 50/50.
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-9


def test_stronger_team_favoured():
    elo = EloModel(ratings={"Brazil": 2000, "Tonga": 1300})
    p = elo.win_probability("Brazil", "Tonga", neutral=True)
    assert p > 0.9


def test_elo_update_moves_ratings_correctly():
    elo = EloModel(ratings={"A": 1500, "B": 1500})
    elo.update("A", "B", 3, 0, neutral=True)  # A wins big
    assert elo.rating("A") > 1500
    assert elo.rating("B") < 1500
    # Zero-sum update.
    assert abs((elo.rating("A") - 1500) + (elo.rating("B") - 1500)) < 1e-9


def test_rps_perfect_vs_wrong():
    perfect = np.array([[1.0, 0.0, 0.0]])
    wrong = np.array([[0.0, 0.0, 1.0]])
    outcome = np.array([0])  # home win
    assert ranked_probability_score(perfect, outcome) == 0.0
    assert ranked_probability_score(wrong, outcome) > ranked_probability_score(
        np.array([[0.4, 0.3, 0.3]]), outcome
    )


def test_log_loss_runs():
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    outcomes = np.array([0, 2])
    assert log_loss_3way(probs, outcomes) > 0
