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

import os

import pandas as pd

API_URL = "https://api.football-data.org/v4/competitions/WC/matches"
API_KEY_ENV = "FOOTBALL_DATA_API_KEY"
TIMEOUT = 30

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
    for m in payload.get("matches", []):
        if m.get("status") != "FINISHED":
            continue
        score = m.get("score", {}).get("fullTime", {})
        hg, ag = score.get("home"), score.get("away")
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

    df = pd.DataFrame(rows, columns=cols)
    print(f"[live] fetched {len(df)} finished 2026 World Cup match(es).")
    return df


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
