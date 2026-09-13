"""Unit tests for the natural-language temporal window parser."""

from datetime import date

from chronos.temporal.parser import parse_window


def test_quarter():
    w = parse_window("Q3 2024")
    assert w.start == date(2024, 7, 1)
    assert w.end == date(2024, 9, 30)


def test_quarter_range():
    w = parse_window("Q1-Q2 2025")
    assert w.start == date(2025, 1, 1)
    assert w.end == date(2025, 6, 30)


def test_year():
    w = parse_window("2024")
    assert w.start == date(2024, 1, 1)
    assert w.end == date(2024, 12, 31)


def test_iso_range():
    w = parse_window("2024-03-01 to 2024-09-30")
    assert w.start == date(2024, 3, 1)
    assert w.end == date(2024, 9, 30)


def test_month_to_month():
    w = parse_window("January 2024 to March 2024")
    assert w.start == date(2024, 1, 1)
    assert w.end == date(2024, 3, 31)


def test_as_of():
    w = parse_window("as of Dec 31, 2023")
    assert w.start == date(2023, 12, 31)
    assert w.end == date(2023, 12, 31)


def test_relative_current_quarter():
    w = parse_window("current quarter", as_of=date(2024, 5, 15))
    assert w.start == date(2024, 4, 1)
    assert w.end == date(2024, 6, 30)


def test_relative_last_90_days():
    w = parse_window("last 90 days", as_of=date(2024, 10, 31))
    assert w.start == date(2024, 8, 2)
    assert w.end == date(2024, 10, 31)


def test_untemporal():
    w = parse_window("what changed?")
    assert w.is_empty


def test_half_year():
    w = parse_window("H1 2024")
    assert w.start == date(2024, 1, 1)
    assert w.end == date(2024, 6, 30)