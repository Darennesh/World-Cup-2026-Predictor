"""Flag emoji lookup for national teams.

Pure Unicode regional-indicator emojis -- no image files, no external requests,
no copyright concerns. Used across the dashboard (tables, group cards, bracket)
to make team names instantly recognisable. Falls back to a neutral globe for any
team not in the map so rendering never breaks.
"""
from __future__ import annotations

# Team name (as used in the dataset) -> flag emoji.
FLAGS: dict[str, str] = {
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Canada": "🇨🇦", "Cape Verde": "🇨🇻", "Colombia": "🇨🇴", "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼", "Czech Republic": "🇨🇿", "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷",
    "Germany": "🇩🇪", "Ghana": "🇬🇭", "Haiti": "🇭🇹", "Iran": "🇮🇷",
    "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮", "Japan": "🇯🇵", "Jordan": "🇯🇴",
    "Mexico": "🇲🇽", "Morocco": "🇲🇦", "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿", "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹", "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳", "South Africa": "🇿🇦",
    "South Korea": "🇰🇷", "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳", "Turkey": "🇹🇷", "United States": "🇺🇸", "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
}

_FALLBACK = "🏳️"


def flag(team: str) -> str:
    """Return the flag emoji for a team, or a neutral fallback."""
    return FLAGS.get(team, _FALLBACK)


def with_flag(team: str) -> str:
    """'🇪🇸 Spain' — flag prefixed to the team name."""
    return f"{flag(team)} {team}"
