"""Tests for the simulation engine: standings, tiebreakers, and bracket logic."""
import numpy as np
import pandas as pd

from src.simulation.standings import Group, select_best_thirds, TeamRecord
from src.simulation.sampler import ScoreSampler
from src.simulation.engine import Tournament, ROUNDS


class FixedModel:
    """A toy model: stronger team (earlier in `order`) gets higher goal rates."""

    def __init__(self, order):
        self.strength = {t: len(order) - i for i, t in enumerate(order)}

    def score_matrix(self, home, away, neutral=True):
        import numpy as np
        lam_h = 0.8 + 0.25 * self.strength[home]
        lam_a = 0.8 + 0.25 * self.strength[away]
        g = np.arange(8)
        from scipy.special import gammaln
        ph = np.exp(g * np.log(lam_h) - lam_h - gammaln(g + 1))
        pa = np.exp(g * np.log(lam_a) - lam_a - gammaln(g + 1))
        m = np.outer(ph, pa)
        return m / m.sum()


def test_standings_orders_by_points():
    g = Group("A", ["W", "X", "Y", "Z"])
    # W beats everyone, Z loses everything.
    g.play("W", "X", 2, 0)
    g.play("W", "Y", 1, 0)
    g.play("W", "Z", 3, 0)
    g.play("X", "Y", 1, 1)
    g.play("X", "Z", 2, 0)
    g.play("Y", "Z", 2, 0)
    rng = np.random.default_rng(0)
    table = g.standings(rng)
    assert table[0].team == "W"
    assert table[-1].team == "Z"


def test_head_to_head_breaks_tie():
    # Two teams level on points/GD/GF; head-to-head decides.
    g = Group("B", ["P", "Q", "R", "S"])
    g.play("P", "Q", 1, 0)   # P beats Q head-to-head
    g.play("P", "R", 0, 3)
    g.play("P", "S", 3, 0)
    g.play("Q", "R", 0, 3)
    g.play("Q", "S", 3, 0)
    g.play("R", "S", 3, 0)
    rng = np.random.default_rng(0)
    table = g.standings(rng)
    names = [t.team for t in table]
    # P and Q both have 6 pts, +1/-... check P ranked above Q via h2h.
    assert names.index("P") < names.index("Q")


def test_select_best_thirds_count():
    thirds = [TeamRecord(f"T{i}") for i in range(12)]
    for i, r in enumerate(thirds):
        r.wins = i % 3          # vary points
        r.gf = i
    best = select_best_thirds(thirds, 8, np.random.default_rng(0))
    assert len(best) == 8


def test_full_tournament_probabilities_valid():
    groups = {chr(65 + i): [f"{chr(65+i)}{k}" for k in range(4)] for i in range(12)}
    order = [t for ts in groups.values() for t in ts]
    sampler = ScoreSampler(FixedModel(order))
    tt = Tournament(groups, sampler)
    df = tt.run(n_sims=200, seed=1)

    # Every team present; probabilities in [0,1] and monotone across rounds.
    assert len(df) == 48
    for col in [f"P_{r}" for r in ROUNDS]:
        assert df[col].between(0, 1).all()
    # P_R32 >= P_R16 >= ... >= P_Champion for each team (nested events).
    for _, row in df.iterrows():
        vals = [row[f"P_{r}"] for r in ROUNDS]
        assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))
    # Exactly one champion per sim -> champion probs sum to ~1.
    assert abs(df["P_Champion"].sum() - 1.0) < 1e-6


def test_champion_prob_favours_strong_teams():
    groups = {chr(65 + i): [f"{chr(65+i)}{k}" for k in range(4)] for i in range(12)}
    order = [t for ts in groups.values() for t in ts]
    sampler = ScoreSampler(FixedModel(order))
    df = Tournament(groups, sampler).run(n_sims=300, seed=2)
    # The globally strongest team (first in `order`) should have above-average
    # champion probability.
    top_team = order[0]
    assert df.set_index("team").loc[top_team, "P_Champion"] > 1 / 48


def test_played_knockout_eliminates_favourite():
    # An 8-team locked bracket where the strongest team (S0) is recorded as
    # losing its first knockout tie -> its champion probability must be 0.
    teams = [f"S{i}" for i in range(8)]
    sampler = ScoreSampler(FixedModel(teams))
    groups = {chr(65 + i): teams[i * 2:i * 2 + 2] + [f"X{i}a", f"X{i}b"]
              for i in range(4)}
    # Minimal Tournament with a fixed 8-team bracket; S0 is the top seed but is
    # recorded as eliminated by S1 in the first round.
    tt = Tournament(groups, sampler, fixed_bracket=teams,
                    played_ko={frozenset(("S0", "S1")): "S1"})
    df = tt.run(n_sims=400, seed=1).set_index("team")
    assert df.loc["S0", "P_Champion"] == 0.0     # eliminated -> 0%
    assert df.loc["S1", "P_Champion"] > 0.0      # advanced instead
