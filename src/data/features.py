"""Leak-free feature engineering.

The cardinal rule for tournament forecasting is **no lookahead**: every feature
attached to a match must be computable from information available strictly
*before* kickoff. We guarantee this structurally by walking matches in
chronological order and, for each match, (1) reading the current state to emit
features, then (2) updating the state with the result. State that has not yet
"seen" a match cannot leak into that match's features.

Features produced per match (all pre-match):
    elo_home, elo_away, elo_diff       - team strength (incl. home edge in diff)
    home_gf, home_ga, away_gf, away_ga - recency-weighted goals for/against
    form_diff                          - net attacking edge (home vs away)
    home_rest, away_rest               - days since each team last played
    neutral                            - neutral venue flag (int)
    importance                         - match-importance weight
Targets carried through: home_goals, away_goals, outcome.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.ratings.elo import EloModel

# Recency weighting for form: most recent games count most.
FORM_WINDOW = 10
FORM_DECAY = 0.85       # weight of game k-back = DECAY**k


def _weighted_form(history: deque) -> tuple[float, float]:
    """Recency-weighted average goals (for, against) from a team's recent games.

    `history` holds (goals_for, goals_against) for prior matches, most recent
    last. Returns (0.0, 0.0) priors when the team has no history.
    """
    if not history:
        return 0.0, 0.0
    gf = np.array([h[0] for h in history], dtype=float)
    ga = np.array([h[1] for h in history], dtype=float)
    # Most-recent-last -> weights increasing toward the end.
    k = np.arange(len(history))[::-1]
    w = FORM_DECAY ** k
    w_sum = w.sum()
    return float((gf * w).sum() / w_sum), float((ga * w).sum() / w_sum)


def build_features(matches: pd.DataFrame, *, save: bool = True) -> pd.DataFrame:
    """Compute pre-match features for every match via one chronological pass."""
    matches = matches.sort_values("date").reset_index(drop=True)

    elo = EloModel()
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    last_played: dict[str, pd.Timestamp] = {}

    rows = []
    for m in matches.itertuples(index=False):
        home, away = m.home_team, m.away_team
        neutral = bool(m.neutral)

        # --- (1) READ STATE -> emit pre-match features ---
        elo_home, elo_away = elo.rating(home), elo.rating(away)
        home_adv = 0.0 if neutral else elo.home_adv
        h_gf, h_ga = _weighted_form(form[home])
        a_gf, a_ga = _weighted_form(form[away])

        home_rest = (m.date - last_played[home]).days if home in last_played else 30
        away_rest = (m.date - last_played[away]).days if away in last_played else 30

        rows.append({
            "date": m.date,
            "home_team": home, "away_team": away,
            "elo_home": elo_home, "elo_away": elo_away,
            "elo_diff": (elo_home + home_adv) - elo_away,
            "home_gf": h_gf, "home_ga": h_ga,
            "away_gf": a_gf, "away_ga": a_ga,
            "form_diff": (h_gf - h_ga) - (a_gf - a_ga),
            "home_rest": min(home_rest, 365), "away_rest": min(away_rest, 365),
            "neutral": int(neutral),
            "importance": float(m.importance),
            # Targets:
            "home_goals": int(m.home_goals), "away_goals": int(m.away_goals),
            "outcome": int(m.outcome),
        })

        # --- (2) UPDATE STATE with the realised result ---
        elo.update(home, away, m.home_goals, m.away_goals,
                   neutral=neutral, importance=m.importance)
        form[home].append((m.home_goals, m.away_goals))
        form[away].append((m.away_goals, m.home_goals))
        last_played[home] = m.date
        last_played[away] = m.date

    feats = pd.DataFrame(rows)
    if save:
        out = PROCESSED_DIR / "features.parquet"
        feats.to_parquet(out, index=False)
        print(f"Wrote {len(feats):,} feature rows to {out}")
    return feats


def load_matches() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "matches.parquet")


if __name__ == "__main__":
    build_features(load_matches())
