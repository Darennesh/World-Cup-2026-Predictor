"""Tests for Bayesian rating updating from group-stage results."""
import pandas as pd

from src.ratings.bayesian import fit_prior_then_update, group_results_2026


def _matches():
    rows = [
        # Pre-2026 history: A is clearly strong, B weak.
        {"date": "2024-01-01", "home_team": "A", "away_team": "B",
         "home_goals": 3, "away_goals": 0, "neutral": False,
         "importance": 1.0, "tournament": "Friendly"},
        {"date": "2024-02-01", "home_team": "A", "away_team": "B",
         "home_goals": 2, "away_goals": 0, "neutral": False,
         "importance": 1.0, "tournament": "Friendly"},
        # 2026 World Cup group game: B upsets A.
        {"date": "2026-06-12", "home_team": "A", "away_team": "B",
         "home_goals": 0, "away_goals": 2, "neutral": True,
         "importance": 1.0, "tournament": "FIFA World Cup"},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_group_results_extracted():
    res = group_results_2026(_matches())
    assert res == [("A", "B", 0, 2)]


def test_qualifiers_excluded():
    df = _matches()
    df.loc[len(df)] = {"date": pd.Timestamp("2026-03-01"), "home_team": "A",
                       "away_team": "B", "home_goals": 5, "away_goals": 0,
                       "neutral": False, "importance": 0.7,
                       "tournament": "FIFA World Cup qualification"}
    res = group_results_2026(df)
    # The qualification match must NOT count as a 2026 group-stage game.
    assert ("A", "B", 5, 0) not in res


def test_posterior_moves_toward_result():
    _, update = fit_prior_then_update(_matches())
    s = update.shifts().set_index("team")
    # B upset A, so B's posterior rises and A's falls vs their priors.
    assert s.loc["B", "delta"] > 0
    assert s.loc["A", "delta"] < 0


def test_prior_unchanged_without_group_games():
    df = _matches()
    df = df[df["tournament"] != "FIFA World Cup"]   # drop the WC game
    _, update = fit_prior_then_update(df)
    assert update.n_group_matches == 0
    assert all(abs(d) < 1e-9 for d in update.shifts()["delta"])
