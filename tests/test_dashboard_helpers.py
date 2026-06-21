"""Tests for the dashboard flag and recent-form helpers."""
import pandas as pd

from src.visualization.flags import flag, with_flag, FLAGS
from src.visualization.form import recent_form


def test_known_team_has_flag():
    assert flag("Spain") == "🇪🇸"
    assert with_flag("Brazil").startswith("🇧🇷")


def test_unknown_team_falls_back():
    assert flag("Atlantis") == "🏳️"
    assert "Atlantis" in with_flag("Atlantis")


def test_all_flags_nonempty():
    assert all(v for v in FLAGS.values())


def _matches():
    rows = [
        {"date": "2026-01-01", "home_team": "A", "away_team": "B",
         "home_goals": 2, "away_goals": 0},
        {"date": "2026-01-05", "home_team": "A", "away_team": "C",
         "home_goals": 1, "away_goals": 1},
        {"date": "2026-01-09", "home_team": "B", "away_team": "A",
         "home_goals": 3, "away_goals": 0},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_recent_form_orders_oldest_to_newest():
    form = recent_form(_matches(), n=5)
    # A: won (2-0), drew (1-1), lost (0-3) -> ['W','D','L'] most recent last.
    assert form["A"] == ["W", "D", "L"]
    assert form["B"][-1] == "W"      # B's most recent is the 3-0 win


def test_recent_form_window_limits_length():
    form = recent_form(_matches(), n=2)
    assert len(form["A"]) == 2
    assert form["A"] == ["D", "L"]   # only the two most recent
