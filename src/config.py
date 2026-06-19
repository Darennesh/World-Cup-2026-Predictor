"""Central paths and project constants."""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = ROOT / "config"
MODELS_DIR = ROOT / "models_store"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Elo settings ---
ELO_START = 1500.0          # rating for a team with no history
ELO_K = 40.0                # update speed; higher = more reactive
ELO_HOME_ADV = 65.0         # rating points added for the home/host side
ELO_WC_WEIGHT = 60.0        # match-importance multiplier for World Cup games

# --- Match-importance weights (used when fitting on historical data) ---
MATCH_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "FIFA World Cup qualification": 0.7,
    "UEFA Euro": 0.9,
    "Copa América": 0.9,
    "Friendly": 0.3,           # de-weight experimental friendlies
    "default": 0.5,
}

# --- Outcome encoding ---
OUTCOMES = ["home_win", "draw", "away_win"]
