"""Temporal knowledge-graph primitives for ChronosAudit.

The whole system is built around *interval-validated facts*. This module
implements the interval algebra used by every layer:

* overlap / containment tests for retrieval filtering,
* interval slicing for "as-of" and "during window" queries,
* snapshot differencing to answer "what changed between window A and B?".
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from chronos.models import TemporalValidity

MIN_DATE = date.min
MAX_DATE = date.max


@dataclass(frozen=True)
class Interval:
    """Half-open-ish inclusive interval [start, end]; None = open bound."""

    start: Optional[date] = None
    end: Optional[date] = None

    @property
    def start_eff(self) -> date:
        return self.start or MIN_DATE

    @property
    def end_eff(self) -> date:
        return self.end or MAX_DATE

    def overlaps(self, other: "Interval") -> bool:
        return self.start_eff <= other.end_eff and other.start_eff <= self.end_eff

    def contains(self, d: date) -> bool:
        return self.start_eff <= d <= self.end_eff

    def contains_interval(self, other: "Interval") -> bool:
        return self.start_eff <= other.start_eff and other.end_eff <= self.end_eff

    def intersection(self, other: "Interval") -> "Interval":
        if not self.overlaps(other):
            return Interval()
        start = max(self.start_eff, other.start_eff)
        end = min(self.end_eff, other.end_eff)
        return Interval(start if start != MIN_DATE else None, end if end != MAX_DATE else None)

    def __str__(self) -> str:
        return f"{self.start or '…'} → {self.end or '…'}"


def to_interval(v: Optional[TemporalValidity] = None) -> Interval:
    if v is None:
        return Interval()
    return Interval(v.start, v.end)


def interval_from_edge(edge: dict) -> Interval:
    return Interval(edge.get("valid_from"), edge.get("valid_until"))


def edge_valid_within(edge: dict, iv: Interval) -> bool:
    """True when the edge's validity interval intersects ``iv``."""
    from datetime import datetime

    vf = edge.get("valid_from")
    vu = edge.get("valid_until")
    if isinstance(vf, datetime):
        vf = vf.date()
    if isinstance(vu, datetime):
        vu = vu.date()
    return Interval(vf, vu).overlaps(iv)


def overlapping_pairs(
    left: Iterable[TemporalValidity], right: Iterable[TemporalValidity]
) -> list[tuple[TemporalValidity, TemporalValidity]]:
    """All (a, b) pairs whose intervals overlap."""
    return [(a, b) for a, b in itertools.product(left, right) if a.overlaps(b)]


def snapshots(edges: Iterable[dict]) -> list[tuple[date, date, int]]:
    """Compress a collection of interval-validated edges into active-count
    segments — i.e. ``[(start, end, active_edges), ...]`` — which is handy for
    plotting the *temporal density* of facts."""
    events: list[tuple[date, int]] = []
    for e in edges:
        start = e.get("valid_from")
        end = e.get("valid_until")
        if start:
            events.append((start, +1))
        if end:
            events.append((end, -1))
    events.sort(key=lambda x: x[0])
    segments: list[tuple[date, date, int]] = []
    active = 0
    cursor: Optional[date] = None
    for d, delta in events:
        if cursor is not None and d != cursor and active > 0:
            segments.append((cursor, d, active))
        active += delta
        cursor = d
    return segments


def diff_snapshots(
    edges_a: list[dict], edges_b: list[dict], window_a: Interval, window_b: Interval
) -> dict:
    """Compare the set of facts active in window A versus window B.

    Returns dict with keys ``added``, ``removed``, ``changed``, ``unchanged``.
    An edge identity is ``(subject, predicate, object)``; two identities present
    in BOTH snapshots are *changed* when their validity or evidence differ.
    """
    sig = lambda e: (str(e.get("subject", ""))[:80], str(e.get("predicate", ""))[:80], str(e.get("object", ""))[:80])

    def normalize(edges_: Iterable[dict], window: Interval) -> dict[tuple, dict]:
        out: dict[tuple, dict] = {}
        for e in edges_:
            if not edge_valid_within(e, window):
                continue
            # Snap the edge to the query window so identical facts don't look
            # different just because their stored intervals were wider.
            local = dict(e)
            iv = interval_from_edge(e)
            inter = iv.intersection(window)
            local["valid_from"] = inter.start
            local["valid_until"] = inter.end
            out.setdefault(sig(e), local)
        return out

    na, nb = normalize(edges_a, window_a), normalize(edges_b, window_b)
    keys_a, keys_b = set(na), set(nb)

    added = [nb[k] for k in sorted(keys_b - keys_a)]
    removed = [na[k] for k in sorted(keys_a - keys_b)]
    unchanged: list[dict] = []
    changed: list[dict] = []
    for k in sorted(keys_a & keys_b):
        ea, eb = na[k], nb[k]
        if (
            ea.get("valid_from") == eb.get("valid_from")
            and ea.get("valid_until") == eb.get("valid_until")
            and ea.get("evidence") == eb.get("evidence")
        ):
            unchanged.append(eb)
        else:
            changed.append({"before": ea, "after": eb})
    return {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged}