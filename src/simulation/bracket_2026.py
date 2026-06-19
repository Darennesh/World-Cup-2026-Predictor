"""Official 2026 FIFA World Cup Round-of-32 bracket structure.

FIFA fixes exactly which group's winner / runner-up occupies each Round-of-32
slot, and constrains which groups' third-placed teams may fill the eight
"winner vs 3rd" slots (Annex C of the regulations). This module encodes that
official structure so the simulation reproduces the real bracket and the correct
path to the Final -- instead of a generic seeding.

Source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage

Round-of-32 matches (FIFA match numbers):
  73: RU A vs RU B          81: W D  vs 3rd(B/E/F/I/J)
  74: W E  vs 3rd(A/B/C/D/F) 82: W G  vs 3rd(A/E/H/I/J)
  75: W F  vs RU C          83: RU K vs RU L
  76: W C  vs RU F          84: W H  vs RU J
  77: W I  vs 3rd(C/D/F/G/H) 85: W B  vs 3rd(E/F/G/I/J)
  78: RU E vs RU I          86: W J  vs RU H
  79: W A  vs 3rd(C/E/F/H/I) 87: W K  vs 3rd(D/E/I/J/L)
  80: W L  vs 3rd(E/H/I/J/K) 88: RU D vs RU G

The single-elimination tree (so consecutive bracket pairs reproduce R16->Final):
  R16: 89=(74,77) 90=(73,75) 91=(76,78) 92=(79,80)
       93=(83,84) 94=(81,82) 95=(86,88) 96=(85,87)
  QF : 97=(89,90) 98=(93,94) 99=(91,92) 100=(95,96)
  SF : 101=(97,98) 102=(99,100)
  F  : 104=(101,102)
"""
from __future__ import annotations

# Each R32 slot participant is one of:
#   ("W", group)         -> winner of group
#   ("RU", group)        -> runner-up of group
#   ("3", frozenset(...)) -> a third-placed team from one of the allowed groups
_W = lambda g: ("W", g)
_RU = lambda g: ("RU", g)
_3 = lambda s: ("3", frozenset(s))

# R32 matches keyed by FIFA match number -> (home_slot, away_slot).
R32_MATCHES: dict[int, tuple] = {
    73: (_RU("A"), _RU("B")),
    74: (_W("E"), _3("ABCDF")),
    75: (_W("F"), _RU("C")),
    76: (_W("C"), _RU("F")),
    77: (_W("I"), _3("CDFGH")),
    78: (_RU("E"), _RU("I")),
    79: (_W("A"), _3("CEFHI")),
    80: (_W("L"), _3("EHIJK")),
    81: (_W("D"), _3("BEFIJ")),
    82: (_W("G"), _3("AEHIJ")),
    83: (_RU("K"), _RU("L")),
    84: (_W("H"), _RU("J")),
    85: (_W("B"), _3("EFGIJ")),
    86: (_W("J"), _RU("H")),
    87: (_W("K"), _3("DEIJL")),
    88: (_RU("D"), _RU("G")),
}

# Order of R32 matches such that consecutive pairs feed the official R16->Final
# tree (derived from the match dependency tree in the module docstring).
R32_ORDER: list[int] = [74, 77, 73, 75, 83, 84, 81, 82,
                        76, 78, 79, 80, 86, 88, 85, 87]

# The eight "winner vs 3rd" slots, in R32_ORDER, with their allowed third groups.
THIRD_SLOTS: list[tuple[int, frozenset]] = [
    (m, R32_MATCHES[m][1][1])
    for m in R32_ORDER
    if R32_MATCHES[m][1][0] == "3"
]


def assign_thirds(third_groups: list[str]) -> dict[int, str]:
    """Assign the 8 qualifying third-placed groups to the 8 third-slots.

    `third_groups` is the list of group letters whose third-placed team
    qualified (length 8). Returns {match_number: group_letter}. Uses a
    constraint-respecting bipartite matching against each slot's allowed set
    (Annex C guarantees at least one valid assignment exists for every
    combination of eight groups).
    """
    slots = THIRD_SLOTS
    groups = list(third_groups)
    assignment: dict[int, str] = {}

    # Backtracking matching: assign hardest-constrained slots first.
    order = sorted(range(len(slots)),
                   key=lambda i: len(slots[i][1] & set(groups)))

    used = [False] * len(groups)

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        match_no, allowed = slots[order[k]]
        for gi, g in enumerate(groups):
            if not used[gi] and g in allowed:
                used[gi] = True
                assignment[match_no] = g
                if backtrack(k + 1):
                    return True
                used[gi] = False
                del assignment[match_no]
        return False

    if not backtrack(0):
        # Fallback: assign in given order ignoring constraints (should not happen
        # for the 495 official combinations, but keeps the sim robust).
        for (match_no, _), g in zip(slots, groups):
            assignment[match_no] = g
    return assignment


def build_bracket(winners: dict[str, str], runners: dict[str, str],
                  thirds: list[tuple[str, str]]) -> list[str]:
    """Return the 32-team bracket in official R32 order.

    Parameters
    ----------
    winners, runners : {group_letter: team_name}
    thirds : list of (group_letter, team_name) for the 8 best third-placed teams

    The returned list pairs consecutively (0,1), (2,3), ... and, played as a
    standard single-elimination, reproduces the official R16/QF/SF/Final.
    """
    third_team_by_group = {g: t for g, t in thirds}
    third_assignment = assign_thirds([g for g, _ in thirds])  # match_no -> group
    # Invert to group -> match for filling.
    group_for_match = third_assignment

    bracket: list[str] = []
    for m in R32_ORDER:
        home, away = R32_MATCHES[m]
        bracket.append(_resolve(home, winners, runners, third_team_by_group,
                                group_for_match, m))
        bracket.append(_resolve(away, winners, runners, third_team_by_group,
                                group_for_match, m))
    return bracket


def _resolve(slot, winners, runners, third_team_by_group, group_for_match, m):
    kind = slot[0]
    if kind == "W":
        return winners[slot[1]]
    if kind == "RU":
        return runners[slot[1]]
    # Third-placed slot: look up which group was assigned to this match.
    g = group_for_match.get(m)
    return third_team_by_group[g]


# Group letters used by the official structure.
OFFICIAL_GROUPS = list("ABCDEFGHIJKL")


def is_official_layout(group_names) -> bool:
    """True if the tournament uses the official 12-group A..L labelling."""
    return sorted(group_names) == OFFICIAL_GROUPS
