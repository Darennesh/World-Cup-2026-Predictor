"""Live 2026 World Cup results from football-data.org.

The historical GitHub mirror is reliable but batch-updated, so a finished match
can take hours to appear. This module fetches *finished* 2026 World Cup matches
directly from the football-data.org REST API (free tier), which posts final
scores within minutes of full-time. The fetched results are merged into the raw
results table ahead of the mirror (see data.ingest), so the track record and
forecasts reflect a game almost as soon as it ends.

Auth: set the FOOTBALL_DATA_API_KEY environment variable (locally or as a
GitHub Actions secret). If the key is missing or the request fails, the fetch
returns an empty frame and the pipeline silently falls back to the mirror -- so
the build never breaks on a transient API issue.

API: https://www.football-data.org/  (competition code "WC", free tier).
"""
from __future__ import annotations

import json
import os

import pandas as pd

from src.config import PROCESSED_DIR

API_URL = "https://api.football-data.org/v4/competitions/WC/matches"
API_KEY_ENV = "FOOTBALL_DATA_API_KEY"
TIMEOUT = 30

# Knockout shootout / extra-time metadata that the mirror CSV cannot hold lives
# in this side-file, keyed by the sorted "TeamA|TeamB" pair.
KNOCKOUT_META_PATH = PROCESSED_DIR / "knockout_meta.json"

# football-data.org names -> names used in our dataset (only where they differ).
TEAM_NAME_MAP = {
    "United States": "United States",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Curaçao": "Curaçao",
    "Curacao": "Curaçao",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
}


def _map_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def fetch_live_results(api_key: str | None = None) -> pd.DataFrame:
    """Return finished 2026 World Cup matches as a results-CSV-shaped frame.

    Columns match the mirror schema: date, home_team, away_team, home_score,
    away_score, tournament, city, country, neutral. Returns an empty frame
    (same columns) if the key is absent or the request fails.
    """
    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "city", "country", "neutral"]
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        print(f"[live] {API_KEY_ENV} not set; skipping live results fetch.")
        return pd.DataFrame(columns=cols)

    try:
        import requests
        resp = requests.get(API_URL, headers={"X-Auth-Token": key},
                            params={"season": 2026}, timeout=TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"[live] fetch failed ({exc}); falling back to mirror.")
        return pd.DataFrame(columns=cols)

    rows = []
    meta_updates = {}
    for m in payload.get("matches", []):
        if m.get("status") != "FINISHED":
            continue
        score = m.get("score", {}) or {}
        full = score.get("fullTime", {}) or {}
        hg, ag = full.get("home"), full.get("away")
        if hg is None or ag is None:
            continue
        home = _map_team((m.get("homeTeam") or {}).get("name", ""))
        away = _map_team((m.get("awayTeam") or {}).get("name", ""))
        if not home or not away:
            continue
        date = str(m.get("utcDate", ""))[:10]
        rows.append({
            "date": date, "home_team": home, "away_team": away,
            "home_score": int(hg), "away_score": int(ag),
            "tournament": "FIFA World Cup", "city": None, "country": None,
            "neutral": True,
        })
        # Capture penalty shootout (knockout draws) and the resolved winner.
        pens = score.get("penalties") or {}
        ph, pa = pens.get("home"), pens.get("away")
        if ph is not None and pa is not None:
            winner = home if ph > pa else away
            meta_updates[_pair_key(home, away)] = {
                "home": home, "away": away,
                "ft_home": int(hg), "ft_away": int(ag),
                "pens_home": int(ph), "pens_away": int(pa),
                "winner": winner,
            }

    if meta_updates:
        _save_knockout_meta(meta_updates)

    df = pd.DataFrame(rows, columns=cols)
    print(f"[live] fetched {len(df)} finished 2026 World Cup match(es); "
          f"{len(meta_updates)} shootout(s) recorded.")
    return df


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted([str(a), str(b)]))


def load_knockout_meta(path=KNOCKOUT_META_PATH) -> dict:
    """Return the persisted shootout metadata, or {} if none recorded."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_knockout_meta(updates: dict, path=KNOCKOUT_META_PATH) -> None:
    """Merge new shootout records into the side-file (keeps past results)."""
    meta = load_knockout_meta(path)
    # Merge per key so a winner-only record (from shootouts.csv) does not wipe
    # out richer pen-score fields already captured from the API, and vice versa.
    for k, v in updates.items():
        existing = meta.get(k, {})
        existing.update({kk: vv for kk, vv in v.items() if vv is not None})
        meta[k] = existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def shootouts_to_meta(path, meta_path=KNOCKOUT_META_PATH) -> int:
    """Fold 2026 World Cup penalty-shootout winners from shootouts.csv into the
    knockout meta side-file. Returns the number of 2026 shootouts recorded.

    shootouts.csv columns: date, home_team, away_team, winner, first_shooter.
    It carries the *winner* but not the pen score, which is enough to advance
    the correct team in the bracket; pen scores (if any) come from the API.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    df = df[pd.to_datetime(df["date"], errors="coerce") >= "2026-06-01"]
    updates = {}
    for r in df.itertuples(index=False):
        home = _map_team(str(r.home_team))
        away = _map_team(str(r.away_team))
        winner = _map_team(str(r.winner))
        updates[_pair_key(home, away)] = {
            "home": home, "away": away, "winner": winner, "shootout": True,
        }
    if updates:
        _save_knockout_meta(updates, meta_path)
    return len(updates)


def merge_live_into_mirror(mirror: pd.DataFrame,
                           live: pd.DataFrame) -> pd.DataFrame:
    """Merge live results into the mirror frame, preferring live for any
    matching fixture and adding games the mirror hasn't published.

    Fixtures are matched on the *unordered* team pair within a +/-1 day window,
    so harmless date/name discrepancies between the two sources don't create
    duplicate rows (e.g. mirror dates a late-night game one day off the API).
    The live row supersedes the mirror row entirely.
    """
    if live is None or live.empty:
        return mirror

    mirror = mirror.copy()
    live = live.copy()
    mirror["_d"] = pd.to_datetime(mirror["date"], errors="coerce")
    live["_d"] = pd.to_datetime(live["date"], errors="coerce")

    def pair(df):
        a = df["home_team"].astype(str)
        b = df["away_team"].astype(str)
        lo = a.where(a < b, b)
        hi = a.where(a >= b, b)
        return lo + "|" + hi

    mirror["_pair"] = pair(mirror)
    live["_pair"] = pair(live)

    # For each live fixture, drop any mirror row with the same team pair within
    # one day of the live date.
    drop_idx = set()
    for lp, ld in zip(live["_pair"], live["_d"]):
        m = mirror[(mirror["_pair"] == lp)
                   & (mirror["_d"] - ld).abs().le(pd.Timedelta(days=1))]
        drop_idx.update(m.index.tolist())

    mirror = mirror.drop(index=drop_idx)
    merged = pd.concat([mirror, live], ignore_index=True)
    return merged.drop(columns=["_d", "_pair"])
