"""Tests for the official 2026 FIFA Round-of-32 bracket structure."""
from src.simulation.bracket_2026 import (R32_MATCHES, R32_ORDER, THIRD_SLOTS,
                                         assign_thirds, build_bracket,
                                         is_official_layout, OFFICIAL_GROUPS)


def test_sixteen_r32_matches():
    assert len(R32_MATCHES) == 16
    assert len(R32_ORDER) == 16
    assert sorted(R32_ORDER) == sorted(R32_MATCHES.keys())


def test_eight_third_slots():
    # Exactly eight winner-vs-third slots, per FIFA structure.
    assert len(THIRD_SLOTS) == 8


def test_assign_thirds_respects_constraints():
    # All twelve groups but only the 8 in this set qualify thirds.
    third_groups = ["A", "C", "E", "G", "H", "I", "J", "L"]
    assignment = assign_thirds(third_groups)
    assert len(assignment) == 8
    # Each assigned group must be allowed in its slot.
    allowed = {m: s for m, s in THIRD_SLOTS}
    for match_no, g in assignment.items():
        assert g in allowed[match_no]
    # All eight groups used exactly once.
    assert sorted(assignment.values()) == sorted(third_groups)


def test_build_bracket_full_32():
    winners = {g: f"W{g}" for g in OFFICIAL_GROUPS}
    runners = {g: f"R{g}" for g in OFFICIAL_GROUPS}
    third_groups = ["A", "C", "E", "G", "H", "I", "J", "L"]
    thirds = [(g, f"T{g}") for g in third_groups]

    bracket = build_bracket(winners, runners, thirds)
    assert len(bracket) == 32
    assert len(set(bracket)) == 32     # no duplicates

    # Match 73 must be RU A vs RU B (it sits at a known position in R32_ORDER).
    pos = R32_ORDER.index(73) * 2
    assert {bracket[pos], bracket[pos + 1]} == {"RA", "RB"}


def test_official_layout_detection():
    assert is_official_layout(list("ABCDEFGHIJKL"))
    assert not is_official_layout(list("ABCD"))


def test_assign_thirds_many_combinations():
    # A handful of representative third-group sets must all yield valid matchings.
    import itertools
    allowed = {m: s for m, s in THIRD_SLOTS}
    combos = list(itertools.combinations(OFFICIAL_GROUPS, 8))
    # Test a deterministic sample for speed.
    for combo in combos[::40]:
        assignment = assign_thirds(list(combo))
        assert len(assignment) == 8
        for match_no, g in assignment.items():
            assert g in allowed[match_no]
