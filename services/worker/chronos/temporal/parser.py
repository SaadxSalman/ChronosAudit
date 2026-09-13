"""Natural-language date-window parser.

Turns audit-friendly expressions into explicit ``[start, end]`` intervals:

* ``"Q3 2024"``, ``"Q1-Q2 2025"``, ``"H1 2024"``
* ``"2024"``, ``"2024-03-01 to 2024-09-30"``, ``"Jan 2024 through Mar 2024"`
* ``"as of Dec 31, 2023"`` (point-in-time)
* ``"current"`` / ``"today"`` (resolved to the real current date)
* ``"last tax year"`` approachable rules

Unparseable input returns ``ParsedWindow(None, None, "")`` so callers can fall
back to an open (untemporal) retrieval.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_QUARTER_BY_MONTH = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}


@dataclass(frozen=True)
class ParsedWindow:
    start: Optional[date]
    end: Optional[date]
    label: str

    @property
    def is_empty(self) -> bool:
        return self.start is None and self.end is None

    def as_interval(self):
        from .interval import Interval

        return Interval(self.start, self.end)


def _quarter_bounds(q: int, year: int) -> tuple[date, date]:
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    _, last_day = calendar.monthrange(year, end_month)
    return date(year, start_month, 1), date(year, end_month, last_day)


def _half_bounds(h: int, year: int) -> tuple[date, date]:
    if h == 1:
        return date(year, 1, 1), date(year, 6, 30)
    return date(year, 7, 1), date(year, 12, 31)


def _month_year_bounds(month: int, year: int) -> tuple[date, date]:
    _, last_day = calendar.monthrange(year, month)
    return date(year, month, 1), date(year, month, last_day)
def parse_window(text: Optional[str], as_of: Optional[date] = None) -> ParsedWindow:
    """Parse a date-window expression into an explicit interval.

    ``as_of`` defaults to today and is used for relative expressions like
    "current quarter" and "last 90 days".
    """
    as_of = as_of or date.today()
    if not text:
        return ParsedWindow(None, None, "")
    t = text.strip()
    low = " " + t.lower() + " "

    # --- Explicit ISO / US date range -------------------------------------
    m = re.search(
        r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\s*(?:to|through|until|-|–|→)\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})",
        t,
    )
    if m:
        y1, m1, d1, y2, m2, d2 = (int(x) for x in m.groups())
        return ParsedWindow(date(y1, m1, d1), date(y2, m2, d2), t)

    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        if re.search(r"(as of|on|at)\s+", low):
            return ParsedWindow(date(y, mo, d), date(y, mo, d), t)
        return ParsedWindow(date(y, mo, d), date(y, mo, d) + timedelta(days=1), t)

    # --- "as of <Month> <d>, <year>" point-in-time --------------------------
    m = re.search(
        r"\b(as of|at|on)\s+(jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)\s+(\d{1,2}),?\s*(\d{4})\b",
        low,
    )
    if m:
        mo, d, y = _MONTHS[m.group(2)], int(m.group(3)), int(m.group(4))
        d0 = date(y, mo, d)
        return ParsedWindow(d0, d0, t)

    # --- "Q3 2024" / "Q1-Q2 2024" ------------------------------------------
    m = re.search(r"\bq([1-4])\s*(?:-|to|through)?\s*(?:q)?([1-4])?\s*,?\s*(\d{4})\b", low)
    if m:
        q1 = int(m.group(1))
        y = int(m.group(3))
        q2 = int(m.group(2)) if m.group(2) else q1
        lo, _ = _quarter_bounds(min(q1, q2), y)
        _, hi = _quarter_bounds(max(q1, q2), y)
        return ParsedWindow(lo, hi, t)

    # --- "H1 2024" / "H2 2024" --------------------------------------------
    m = re.search(r"\bh([12])\s*,?\s*(\d{4})\b", low)
    if m:
        lo, hi = _half_bounds(int(m.group(1)), int(m.group(2)))
        return ParsedWindow(lo, hi, t)

    # --- "January 2024 to March 2024" --------------------------------------
    m = re.search(
        r"\b(jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)\s+(\d{4})\s*(?:to|through|until|-|–|→)\s*(jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)\s+(\d{4})\b",
        low,
    )
    if m:
        m1, y1 = _MONTHS[m.group(1)], int(m.group(2))
        m2, y2 = _MONTHS[m.group(3)], int(m.group(4))
        lo, _ = _month_year_bounds(m1, y1)
        _, hi = _month_year_bounds(m2, y2)
        return ParsedWindow(lo, hi, t)

    # --- "January 2024" / "Jan 2024" ---------------------------------------
    m = re.search(r"\b(jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)\s+(\d{4})\b", low)
    if m:
        mo, y = _MONTHS[m.group(1)], int(m.group(2))
        lo, hi = _month_year_bounds(mo, y)
        return ParsedWindow(lo, hi, t)

    # --- "2024" year --------------------------------------------------------
    m = re.search(r"\b(\d{4})\b", t)
    if m:
        y = int(m.group(1))
        if y < 1900 or y > 2200:
            return ParsedWindow(None, None, "")
        return ParsedWindow(date(y, 1, 1), date(y, 12, 31), t)

    # --- Relative expressions ----------------------------------------------
    if "current quarter" in low:
        lo, hi = _quarter_bounds(_QUARTER_BY_MONTH[as_of.month], as_of.year)
        return ParsedWindow(lo, hi, "current quarter")
    if "previous quarter" in low or "last quarter" in low:
        y = as_of.year if as_of.month > 3 else as_of.year - 1
        m_ref = as_of.month - 3 if as_of.month > 3 else as_of.month + 9
        lo, hi = _quarter_bounds(_QUARTER_BY_MONTH[m_ref], y)
        return ParsedWindow(lo, hi, "previous quarter")
    if "current year" in low or "this year" in low:
        return ParsedWindow(date(as_of.year, 1, 1), date(as_of.year, 12, 31), "current year")
    if re.search(r"last \d+ (days|months|years)", low):
        m = re.search(r"last (\d+) (days|months|years)", low)
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "days":
            start = as_of - timedelta(days=n)
        elif unit == "months":
            cursor = as_of.replace(day=1)
            for _ in range(n):
                cursor = (cursor - timedelta(days=1)).replace(day=1)
            start = cursor
        else:
            start = date(as_of.year - n, as_of.month, as_of.day)
        return ParsedWindow(start, as_of, t)
    if "last month" in low:
        end = as_of.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        return ParsedWindow(start, end, t)
    if "last 30 days" in low or "30 days" in low:
        return ParsedWindow(as_of - timedelta(days=30), as_of, t)
    if low.startswith("last tax year"):
        return ParsedWindow(date(as_of.year - 1, 1, 1), date(as_of.year - 1, 12, 31), "last tax year")

    if "current" in low or "today" in low:
        return ParsedWindow(as_of, as_of, "current")

    return ParsedWindow(None, None, "")