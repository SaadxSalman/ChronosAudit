"""Unit tests for the temporal interval algebra."""

from datetime import date

from chronos.models import TemporalValidity
from chronos.temporal.interval import Interval, diff_snapshots, edge_valid_within, snapshots


def test_overlap_open_ended():
    a = Interval(date(2024, 1, 1), None)  # open-ended
    b = Interval(date(2024, 7, 1), date(2024, 9, 30))
    assert a.overlaps(b)
    assert a.contains(date(2025, 6, 1))
    assert not b.contains(date(2025, 6, 1))


def test_overlap_disjoint():
    a = Interval(date(2024, 1, 1), date(2024, 3, 31))
    b = Interval(date(2024, 4, 1), date(2024, 6, 30))
    assert not a.overlaps(b)


def test_intersection():
    a = Interval(date(2024, 1, 1), date(2024, 12, 31))
    b = Interval(date(2024, 7, 1), date(2025, 6, 30))
    inter = a.intersection(b)
    assert inter.start == date(2024, 7, 1)
    assert inter.end == date(2024, 12, 31)


def test_edge_valid_within():
    edge = {"valid_from": date(2024, 1, 1), "valid_until": date(2024, 3, 31)}
    assert edge_valid_within(edge, Interval(date(2024, 2, 1), date(2024, 2, 15)))
    assert not edge_valid_within(edge, Interval(date(2024, 4, 1), date(2024, 5, 1)))
    open_edge = {"valid_from": date(2024, 1, 1), "valid_until": None}
    assert edge_valid_within(open_edge, Interval(date(2030, 1, 1), date(2030, 12, 31)))


def test_model_validity_overlap():
    va = TemporalValidity(start=date(2024, 1, 1), end=date(2024, 3, 31))
    vb = TemporalValidity(start=date(2024, 3, 15), end=None)
    assert va.overlaps(vb)
    assert vb.contains(date(2030, 5, 5))


def test_diff_snapshots():
    edges_a = [
        {"subject": "Policy X", "predicate": "requires", "object": "A", "valid_from": date(2023, 1, 1), "valid_until": date(2025, 12, 31), "evidence": "same"},
        {"subject": "Policy X", "predicate": "permits", "object": "B", "valid_from": date(2023, 1, 1), "valid_until": None, "evidence": "same"},
    ]
    edges_b = [
        # same factual triple, but with a changed validity
        {"subject": "Policy X", "predicate": "requires", "object": "A", "valid_from": date(2026, 1, 1), "valid_until": None, "evidence": "updated"},
        # brand-new fact in B
        {"subject": "Policy X", "predicate": "prohibits", "object": "C", "valid_from": date(2026, 1, 1), "valid_until": None, "evidence": "new"},
        # open-ended fact still in force in B → carried forward unchanged
        {"subject": "Policy X", "predicate": "permits", "object": "B", "valid_from": date(2023, 1, 1), "valid_until": None, "evidence": "same"},
    ]
    wa = Interval(date(2023, 1, 1), date(2025, 12, 31))
    wb = Interval(date(2026, 1, 1), date(2026, 12, 31))
    result = diff_snapshots(edges_a, edges_b, wa, wb)
    assert any(e["object"] == "C" for e in result["added"])
    # "permits B" was open-ended, so it carries forward and is not removed
    assert any(e["object"] == "B" for e in result["unchanged"])
    # "requires A" exists on both sides → changed (validity/evidence differ)
    assert any(c["after"]["object"] == "A" for c in result["changed"])


def test_snapshots_segments():
    edges = [
        {"valid_from": date(2024, 1, 1), "valid_until": date(2024, 6, 30)},
        {"valid_from": date(2024, 3, 1), "valid_until": date(2024, 9, 30)},
    ]
    segs = snapshots(edges)
    assert segs, "should produce at least one active segment"
    assert segs[0][2] == 1  # only the first edge active initially