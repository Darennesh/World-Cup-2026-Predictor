"""Tests for live group-stage forecasting and team-news adjustments."""
import numpy as np

from src.simulation.group_forecast import (forecast_group, played_results_for,
                                           MatchForecast)
from src.ratings.adjustments import apply_to_states


class FixedModel:
    """Stronger team (earlier in order) scores more."""
    def __init__(self, order):
        self.strength = {t: len(order) - i for i, t in enumerate(order)}

    def score_matrix(self, home, away, neutral=True):
        from scipy.special import gammaln
        lam_h = 0.7 + 0.3 * self.strength[home]
        lam_a = 0.7 + 0.3 * self.strength[away]
        g = np.arange(8)
        ph = np.exp(g * np.log(lam_h) - lam_h - gammaln(g + 1))
        pa = np.exp(g * np.log(lam_a) - lam_a - gammaln(g + 1))
        m = np.outer(ph, pa)
        return m / m.sum()


def test_remaining_excludes_played():
    teams = ["A", "B", "C", "D"]
    model = FixedModel(teams)
    played = [("A", "B", 2, 0)]      # one of six fixtures already played
    fc = forecast_group(model, "G", teams, played, n_sims=300)
    # 6 round-robin fixtures, 1 played -> 5 remaining.
    assert len(fc.remaining) == 5
    assert all(isinstance(m, MatchForecast) for m in fc.remaining)
    assert frozenset(("A", "B")) not in {frozenset((m.home, m.away)) for m in fc.remaining}


def test_match_probabilities_sum_to_one():
    teams = ["A", "B", "C", "D"]
    fc = forecast_group(FixedModel(teams), "G", teams, [], n_sims=200)
    for m in fc.remaining:
        assert abs(m.p_home + m.p_draw + m.p_away - 1.0) < 1e-6


def test_stronger_team_more_likely_to_advance():
    teams = ["Strong", "Good", "Weak", "Weakest"]
    fc = forecast_group(FixedModel(teams), "G", teams, [], n_sims=1500)
    assert fc.advance_prob["Strong"] > fc.advance_prob["Weakest"]
    # Advance probs: 2 of 4 advance -> total ~2.0.
    assert abs(sum(fc.advance_prob.values()) - 2.0) < 0.05


def test_played_results_filter():
    teams = ["A", "B"]
    allres = [("A", "B", 1, 0), ("A", "X", 3, 3), ("Y", "Z", 0, 0)]
    out = played_results_for(teams, allres)
    assert out == [("A", "B", 1, 0)]


def test_adjustments_shift_elo_only():
    class S:
        def __init__(self, elo, gf, ga):
            self.elo, self.gf, self.ga = elo, gf, ga
    states = {"Spain": S(2000, 2.0, 0.8), "Tonga": S(1300, 0.5, 2.5)}
    out = apply_to_states(states, {"Spain": -50})
    assert out["Spain"].elo == 1950
    assert out["Spain"].gf == 2.0          # form untouched
    assert out["Tonga"].elo == 1300        # unaffected team unchanged
    # Original not mutated.
    assert states["Spain"].elo == 2000
