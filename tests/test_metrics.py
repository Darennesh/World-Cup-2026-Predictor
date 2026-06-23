"""Tests for the added probabilistic-quality metrics."""
import numpy as np

from src.evaluation.metrics import (brier_score_multiclass,
                                     expected_calibration_error,
                                     classification_report_3way,
                                     confusion_matrix_3way, skill_score)


def test_brier_perfect_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 2])
    assert brier_score_multiclass(probs, outcomes) == 0.0


def test_brier_worst_is_two():
    probs = np.array([[0.0, 0.0, 1.0]])
    outcomes = np.array([0])
    assert abs(brier_score_multiclass(probs, outcomes) - 2.0) < 1e-9


def test_ece_perfect_calibration_low():
    # Predictions that exactly match frequencies -> low ECE.
    rng = np.random.default_rng(0)
    n = 2000
    outcomes = rng.integers(0, 3, n)
    probs = np.full((n, 3), 1 / 3)        # uniform; observed freq ~1/3 each
    ece = expected_calibration_error(probs, outcomes)
    assert ece < 0.1


def test_classification_report_keys_and_accuracy():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]])
    outcomes = np.array([0, 2, 1])        # all correct
    rep = classification_report_3way(probs, outcomes)
    assert rep["accuracy"] == 1.0
    assert set(rep["classes"]) == {"Home", "Draw", "Away"}
    assert rep["classes"]["Home"]["precision"] == 1.0


def test_confusion_matrix_shape_and_counts():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    outcomes = np.array([0, 0])           # second is predicted Away, actual Home
    cm = confusion_matrix_3way(probs, outcomes)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 2] == 1


def test_skill_score_positive_when_better():
    assert skill_score(0.18, 0.24) > 0
    assert skill_score(0.24, 0.18) < 0
