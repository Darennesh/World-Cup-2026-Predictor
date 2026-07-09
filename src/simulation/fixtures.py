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
    is_knockout: bool = False             # True for R32/R16/.../Final ties

    @property
    def kickoff_eat(self) -> datetime | None:
        return self.kickoff_utc.astimezone(EAT) if self.kickoff_utc else None

    @property
    def eat_label(self) -> str:
        k = self.kickoff_eat
        return k.strftime("%a %d %b, %H:%M EAT") if k else "Time TBD (EAT)"

    @property
    def favourite(self) -> str:
        # Knockout ties always produce a winner (extra time / penalties), so the
        # pick is the side more likely to win the tie, never "Draw".
        if self.is_knockout:
            return self.home if self.p_home >= self.p_away else self.away
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


def load_round_schedule(path: Path = SCHEDULE_PATH) -> dict[str, list[datetime]]:
    """Load round-level kickoff times for ties whose teams aren't known yet.

    Lets the Semi-finals and Final be *pre-staged* before the participants are
    decided. YAML shape (times UTC, in bracket order within a round):

        rounds:
          Semi-final: [2026-07-14T19:00:00Z, 2026-07-15T19:00:00Z]
          Final: [2026-07-19T19:00:00Z]

    Used as a fallback when a fixture has no exact "Home vs Away" entry: the
    n-th tie of a round takes the n-th time listed for that round.
    """
    if not Path(path).exists():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, list[datetime]] = {}
    for rnd, times in (data.get("rounds") or {}).items():
        parsed = [_parse_utc(t) for t in (times or [])]
        out[str(rnd)] = [t for t in parsed if t is not None]
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


def _team_groups(groups: dict[str, list[str]]) -> dict[str, str]:
    return {t: g for g, teams in groups.items() for t in teams}


def knockout_winner(home: str, away: str, hg: int, ag: int,
                    meta: dict | None = None) -> str | None:
    """Resolve who advanced from a played knockout tie.

    Decided by goals when not level; a level score went to penalties, so the
    winner is read from the shootout meta (football-data.org). Returns None when
    the tie is level and no shootout record is available.
    """
    if hg > ag:
        return home
    if ag > hg:
        return away
    if meta:
        rec = meta.get("|".join(sorted([home, away])))
        if rec and rec.get("winner"):
            return rec["winner"]
    return None


def played_knockout_results(matches: pd.DataFrame,
                            groups: dict[str, list[str]]):
    """Return played knockout ties as a dict keyed by the unordered team pair.

    A knockout match is a 2026 World Cup game between teams from *different*
    groups (group-stage games are always within a single group), so this cleanly
    separates the two phases without a fixtures list.
    """
    wc = matches[_is_2026_wc(matches)]
    tg = _team_groups(groups)
    out = {}
    for m in wc.itertuples(index=False):
        gh, ga = tg.get(m.home_team), tg.get(m.away_team)
        if gh is not None and ga is not None and gh != ga:
            out[_fixture_key(m.home_team, m.away_team)] = (
                m.home_team, m.away_team, int(m.home_goals), int(m.away_goals))
    return out


def _round_labels_for(n: int) -> list[str]:
    names = {32: "Round of 32", 16: "Round of 16", 8: "Quarter-final",
             4: "Semi-final", 2: "Final"}
    labels, size = [], n
    while size >= 2:
        labels.append(names.get(size, f"Last {size}"))
        size //= 2
    return labels


def upcoming_knockout_fixtures(model, matches: pd.DataFrame,
                               groups: dict[str, list[str]],
                               bracket: list[str],
                               schedule: dict[frozenset, datetime] | None = None,
                               round_schedule: dict[str, list[datetime]] | None = None
                               ) -> list[UpcomingFixture]:
    """Derive the next knockout ties whose participants are already decided.

    Walks the locked Round-of-32 `bracket` round by round, advancing the actual
    winner where a result exists. A tie is 'upcoming' when both its teams are
    known (from earlier results or the locked R32 seeding) but it has not yet
    been played. Returns those ties with live predictions, ordered by kickoff.

    `round_schedule` supplies pre-staged round-level times (Semi-final, Final)
    used when a tie has no exact team-pair kickoff yet -- so those ties still
    show a date and sort correctly before their teams are known.
    """
    schedule = schedule if schedule is not None else load_schedule()
    round_schedule = (round_schedule if round_schedule is not None
                      else load_round_schedule())
    known = set(model.teams)
    played = played_knockout_results(matches, groups)
    from src.data.live_results import load_knockout_meta
    meta = load_knockout_meta()
    labels = _round_labels_for(len(bracket))

    fixtures: list[UpcomingFixture] = []
    current = list(bracket)
    ri = 0
    while len(current) > 1:
        rnd = labels[ri] if ri < len(labels) else f"Round of {len(current)}"
        round_idx = 0        # position of this tie within the round
        nxt = []
        for k in range(0, len(current), 2):
            a, b = current[k], current[k + 1]
            if a is None or b is None:
                nxt.append(None)
                continue
            res = played.get(_fixture_key(a, b))
            if res is not None:
                hh, aa, hg, ag = res
                winner = knockout_winner(hh, aa, hg, ag, meta)
                if winner is None:
                    # Level tie, shootout result not yet available -> advance
                    # the model's favourite as a placeholder until it lands.
                    ph, _, pa, _, _ = _predict(model, a, b)
                    winner = a if ph >= pa else b
                nxt.append(winner)
            else:
                nxt.append(None)
                # Both teams known but unplayed -> an upcoming fixture.
                if a in known and b in known:
                    ph, pd_, pa, eh, ea = _predict(model, a, b)
                    # Exact team-pair time if scheduled, else the round-level
                    # pre-staged time for this tie's position in the round.
                    kickoff = schedule.get(_fixture_key(a, b))
                    if kickoff is None:
                        times = round_schedule.get(rnd, [])
                        if round_idx < len(times):
                            kickoff = times[round_idx]
                    fixtures.append(UpcomingFixture(
                        group=rnd, home=a, away=b,
                        p_home=ph, p_draw=pd_, p_away=pa,
                        exp_home=eh, exp_away=ea,
                        kickoff_utc=kickoff, is_knockout=True))
            round_idx += 1
        current = nxt
        ri += 1

    far_future = datetime.max.replace(tzinfo=timezone.utc)
    fixtures.sort(key=lambda f: (f.kickoff_utc or far_future, f.home))
    return fixtures
