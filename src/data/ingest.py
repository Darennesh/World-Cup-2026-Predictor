"""Load and clean the historical international match dataset into a single
standardized, chronologically sorted table saved as Parquet.

Expected raw input: the widely used Kaggle dataset
"martj42/international-football-results-from-1872-to-2017" (file `results.csv`)
with columns: date, home_team, away_team, home_score, away_score, tournament,
city, country, neutral. The loader is tolerant of minor column-name variants.

Output schema (data/processed/matches.parquet):
    date            datetime64  - match date (sorted ascending)
    home_team       str
    away_team       str
    home_goals      int
    away_goals      int
    tournament      str         - raw competition name
    neutral         bool        - True if played at a neutral venue
    outcome         int         - 0 home_win, 1 draw, 2 away_win
    importance      float       - match-importance weight (see config)
    country         str         - host country (for host-advantage features)
"""
from __future__ import annotations

import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR, MATCH_WEIGHTS

# Map a variety of possible source column names -> our canonical names.
_COLUMN_ALIASES = {
    "home_score": "home_goals",
    "away_score": "away_goals",
    "home": "home_team",
    "away": "away_team",
}

REQUIRED = ["date", "home_team", "away_team", "home_goals", "away_goals"]


def _importance(tournament: str) -> float:
    """Map a competition name to a match-importance weight.

    Uses substring matching so e.g. 'FIFA World Cup qualification' is caught
    before the broader 'FIFA World Cup' key.
    """
    if not isinstance(tournament, str):
        return MATCH_WEIGHTS["default"]
    # Check the most specific keys first.
    for key in ("FIFA World Cup qualification", "FIFA World Cup",
                "UEFA Euro", "Copa América", "Friendly"):
        if key.lower() in tournament.lower():
            return MATCH_WEIGHTS[key]
    return MATCH_WEIGHTS["default"]


def _outcome(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def load_raw(csv_name: str = "results.csv") -> pd.DataFrame:
    """Read the raw results CSV from data/raw/ and normalize columns."""
    path = RAW_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run scripts/01_download_data.py "
            "or place results.csv in data/raw/."
        )
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, type, derive columns, and sort chronologically."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team",
                           "home_goals", "away_goals"])

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    for col in ("home_team", "away_team"):
        df[col] = df[col].astype(str).str.strip()

    if "tournament" not in df.columns:
        df["tournament"] = "Unknown"
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].astype(bool)
    if "country" not in df.columns:
        df["country"] = pd.NA

    df["outcome"] = [_outcome(h, a) for h, a in zip(df["home_goals"], df["away_goals"])]
    df["importance"] = df["tournament"].map(_importance)

    keep = ["date", "home_team", "away_team", "home_goals", "away_goals",
            "tournament", "neutral", "outcome", "importance", "country"]
    df = df[keep].sort_values("date").reset_index(drop=True)
    return df


def build(csv_name: str = "results.csv", save: bool = True) -> pd.DataFrame:
    """End-to-end: load raw -> clean -> (optionally) save Parquet."""
    df = clean(load_raw(csv_name))
    if save:
        out = PROCESSED_DIR / "matches.parquet"
        df.to_parquet(out, index=False)
        print(f"Wrote {len(df):,} matches to {out}")
    return df


if __name__ == "__main__":
    build()
