"""Recent-form helper: last-N results per team for the form-guide strip.

Returns each team's most recent results as a list of 'W'/'D'/'L' (most recent
last), used to render the small coloured dots in the group cards. Looks at all
international matches in the dataset, not just the World Cup, so every team has
context even before the tournament.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd


def recent_form(matches: pd.DataFrame, n: int = 5) -> dict[str, list[str]]:
    """Map each team -> list of its last `n` results as 'W'/'D'/'L'.

    Results are ordered oldest-to-newest within the window (so the last entry is
    the most recent match). Draws count as 'D'.
    """
    df = matches.sort_values("date")
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=n))
    for m in df.itertuples(index=False):
        if m.home_goals > m.away_goals:
            form[m.home_team].append("W")
            form[m.away_team].append("L")
        elif m.home_goals < m.away_goals:
            form[m.home_team].append("L")
            form[m.away_team].append("W")
        else:
            form[m.home_team].append("D")
            form[m.away_team].append("D")
    return {t: list(d) for t, d in form.items()}
