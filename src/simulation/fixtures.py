"""Upcoming group-stage fixtures with live predictions and EAT kickoff times.

This powers the dashboard's "upcoming fixtures" panel. It:
  1. Derives the *remaining* group-stage fixtures automatically -- the full
     round-robin of each group minus the matches already played (so it needs no
     manual fixture list and stays correct as results arrive).
  2. Attaches a live model prediction to each upcoming fixture (win/draw/loss
     probabilities and an expected scoreline) using the current model, so the
     numbers move with form and team-news adjustments.
  3. Looks up each fixture's kickoff time from an editable schedule file
     (config/schedule_2026.yaml, stored in UTC) and converts it to East Africa
     Time (EAT = UTC+3, no daylight saving) for display.

Once a fixture is played it drops out of "remaining" automatically and is picked
up by the track-record logger (see evaluation.tracker) -- so games flow from
"upcoming" to "scored" with no manual step.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import CONFIG_DIR
from src.ratings.bayesian import _is_2026_wc

# East Africa Time is a fixed UTC+3 offset (no DST).
EAT = timezone(timedelta(hours=3))
SCHEDULE_PATH = CONFIG_DIR / "schedule_2026.yaml"


@dataclass
class UpcomingFixture:
    group: str
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    exp_home: float
    exp_away: float
    kickoff_utc: datetime | None = None   # None when not yet scheduled

    @property
    def kickoff_eat(self) -> datetime | None:
        return self.kickoff_utc.astimezone(EAT) if self.kickoff_utc else None

    @property
    def eat_label(self) -> str:
        k = self.kickoff_eat
        return k.strftime("%a %d %b, %H:%M EAT") if k else "Time TBD (EAT)"

    @property
    def favourite(self) -> str:
        if self.p_home >= self.p_away and self.p_home >= self.p_draw:
            return self.home
        if self.p_away >= self.p_home and self.p_away >= self.p_draw:
            return self.away
        return "Draw"


def _fixture_key(home: str, away: str) -> frozenset:
    return frozenset((home, away))


def load_schedule(path: Path = SCHEDULE_PATH) -> dict[frozenset, datetime]:
    """Load kickoff times from the schedule YAML.

    Expected format (UTC, ISO-8601), keyed by 'Home vs Away':

        fixtures:
          "Spain vs Uruguay": 2026-06-24T19:00:00Z
          "France vs Norway": 2026-06-24T16:00:00Z

    Missing or unparseable entries are skipped (fixture shown as Time TBD).
    """
    if not Path(path).exists():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[frozenset, datetime] = {}
    for label, value in (data.get("fixtures") or {}).items():
        if " vs " not in str(label):
            continue
        home, away = [s.strip() for s in str(label).split(" vs ", 1)]
        dt = _parse_utc(value)
        if dt is not None:
            out[_fixture_key(home, away)] = dt
    return out


def _parse_utc(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def remaining_fixtures(matches: pd.DataFrame,
                       groups: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """Return (group, home, away) for every group fixture not yet played."""
    wc = matches[_is_2026_wc(matches)]
    played = {_fixture_key(h, a)
              for h, a in zip(wc["home_team"], wc["away_team"])}

    out = []
    for gname, teams in groups.items():
        for h, a in combinations(teams, 2):
            if _fixture_key(h, a) not in played:
                out.append((gname, h, a))
    return out


def _predict(model, home: str, away: str):
    mat = model.score_matrix(home, away, neutral=True)
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    g = np.arange(mat.shape[0])
    exp_home = float((mat.sum(axis=1) * g).sum())
    exp_away = float((mat.sum(axis=0) * g).sum())
    return p_home, p_draw, p_away, exp_home, exp_away


def upcoming_fixtures(model, matches: pd.DataFrame,
                      groups: dict[str, list[str]],
                      schedule: dict[frozenset, datetime] | None = None
                      ) -> list[UpcomingFixture]:
    """Predict all remaining group fixtures, ordered by kickoff time.

    Fixtures with a known kickoff come first (chronologically); undated fixtures
    follow, grouped by group letter for a stable display order.
    """
    schedule = schedule if schedule is not None else load_schedule()
    known = set(model.teams)

    fixtures: list[UpcomingFixture] = []
    for gname, home, away in remaining_fixtures(matches, groups):
        if home not in known or away not in known:
            continue
        ph, pd_, pa, eh, ea = _predict(model, home, away)
        fixtures.append(UpcomingFixture(
            group=gname, home=home, away=away,
            p_home=ph, p_draw=pd_, p_away=pa, exp_home=eh, exp_away=ea,
            kickoff_utc=schedule.get(_fixture_key(home, away)),
        ))

    far_future = datetime.max.replace(tzinfo=timezone.utc)
    fixtures.sort(key=lambda f: (f.kickoff_utc or far_future, f.group,
                                 f.home))
    return fixtures
