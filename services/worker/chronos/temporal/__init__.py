"""Temporal primitives for ChronosAudit (interval algebra + window parsing)."""

from .interval import Interval, diff_snapshots, edge_valid_within, interval_from_edge, overlapping_pairs, snapshots, to_interval
from .parser import ParsedWindow, parse_window

__all__ = [
    "Interval",
    "ParsedWindow",
    "diff_snapshots",
    "edge_valid_within",
    "interval_from_edge",
    "overlapping_pairs",
    "parse_window",
    "snapshots",
    "to_interval",
]