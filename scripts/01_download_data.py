"""Step 1 of the pipeline: obtain the raw historical results dataset.

Resolution order:
  1. If `data/raw/results.csv` already exists, do nothing.
  2. Fetch from the public GitHub mirror (no authentication required).
  3. Fall back to the Kaggle API (requires `pip install kaggle` and a
     kaggle.json token at ~/.kaggle/).
  4. Otherwise print manual-download instructions, and -- if run with
     `--sample` -- generate a small synthetic dataset so the rest of the
     pipeline can be exercised immediately.

Usage:
    python scripts/01_download_data.py            # real data (GitHub/Kaggle)
    python scripts/01_download_data.py --sample   # also write a synthetic sample
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR  # noqa: E402

KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"
# Same dataset, maintained by the same author, served raw over HTTPS with no
# auth -- the simplest and safest way to get the real data.
GITHUB_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
TARGET = RAW_DIR / "results.csv"

MANUAL_MSG = f"""
Could not fetch automatically. To get the real dataset:

  Option A - Kaggle API:
    1. pip install kaggle
    2. Create an API token at https://www.kaggle.com/settings -> 'Create New Token'
    3. Save kaggle.json to %USERPROFILE%\\.kaggle\\kaggle.json
    4. Re-run this script.

  Option B - Manual download:
    1. Visit https://www.kaggle.com/datasets/{KAGGLE_DATASET}
    2. Download and unzip 'results.csv'
    3. Place it at: {TARGET}

Or run with --sample to generate a small synthetic dataset for testing.
"""


def try_github() -> bool:
    """Download results.csv straight from the public GitHub mirror."""
    try:
        import requests
    except Exception:
        print("`requests` not installed; skipping GitHub fetch.")
        return False
    try:
        print(f"Downloading results.csv from GitHub mirror ...")
        resp = requests.get(GITHUB_URL, timeout=60)
        resp.raise_for_status()
        # Sanity check: must look like the expected CSV header.
        head = resp.text[:200].lower()
        if "home_team" not in head or "home_score" not in head:
            print("Unexpected file contents from GitHub; skipping.")
            return False
        TARGET.write_bytes(resp.content)
        return TARGET.exists()
    except Exception as exc:
        print(f"GitHub fetch failed: {exc}")
        return False


def try_kaggle() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return False
    try:
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {KAGGLE_DATASET} via Kaggle API ...")
        api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True)
        return TARGET.exists()
    except Exception as exc:  # auth/network errors
        print(f"Kaggle API failed: {exc}")
        return False


def write_sample() -> None:
    """Generate a small, plausible synthetic results.csv for smoke testing."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    teams = ["Brazil", "France", "Argentina", "England", "Spain", "Germany",
             "Portugal", "Netherlands", "Croatia", "Mexico", "USA", "Canada",
             "Japan", "Morocco", "Senegal", "Uruguay"]
    # Rough latent strengths to make goals realistic.
    strength = {t: rng.normal(0, 0.4) for t in teams}

    rows = []
    date = pd.Timestamp("2014-01-01")
    for _ in range(1500):
        date += pd.Timedelta(days=int(rng.integers(2, 9)))
        h, a = rng.choice(teams, size=2, replace=False)
        lam_h = np.exp(0.3 + strength[h] - strength[a] + 0.2)  # +home edge
        lam_a = np.exp(0.3 + strength[a] - strength[h])
        hg, ag = rng.poisson(lam_h), rng.poisson(lam_a)
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "home_team": h, "away_team": a,
            "home_score": int(hg), "away_score": int(ag),
            "tournament": rng.choice(["Friendly", "FIFA World Cup qualification",
                                      "FIFA World Cup"], p=[0.5, 0.4, 0.1]),
            "city": "Sample City", "country": h,
            "neutral": bool(rng.random() < 0.15),
        })
    pd.DataFrame(rows).to_csv(TARGET, index=False)
    print(f"Wrote synthetic sample ({len(rows)} matches) to {TARGET}")


def main() -> None:
    want_sample = "--sample" in sys.argv
    force = "--force" in sys.argv

    if TARGET.exists() and not force:
        print(f"Raw dataset already present: {TARGET} (use --force to replace)")
        return

    # Preferred: no-auth GitHub mirror.
    if try_github():
        print(f"Downloaded real dataset to {TARGET}")
        return

    # Fallback: Kaggle API (needs kaggle.json).
    if try_kaggle():
        print(f"Downloaded real dataset to {TARGET}")
        return

    print(MANUAL_MSG)
    if want_sample:
        write_sample()


if __name__ == "__main__":
    main()
