"""Hybrid temporal retriever.

Two complementary lenses:

1. **Vector lens** — LanceDB similarity search with the date predicate
   ``(end IS NULL OR end >= W1) AND (start IS NULL OR start <= W2)`` pushed
   down, so only chunks whose validity overlaps the window are candidates.

2. **Graph lens** — starting from entities in the top hits, walk the NetworkX
   graph restricted to edges whose ``[valid_from, valid_until]`` overlaps the
   same window. Surfaces facts that never appear verbatim in any top-K chunk.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from chronos.embeddings import EmbeddingProvider
from chronos.models import GraphEdge, RetrievalContext
from chronos.storage.graph_store import TemporalGraph
from chronos.storage.lancedb_store import LanceStore
from chronos.temporal.interval import Interval, edge_valid_within
from chronos.temporal.parser import parse_window

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2}


def token_overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def _question_matches_node(node_name: str, question: str) -> bool:
    """True when most of a node's tokens appear in the question text."""
    node_tokens = _tokens(node_name)
    q_tokens = _tokens(question)
    if len(node_tokens) < 2:
        return False
    hit = len(node_tokens & q_tokens)
    return hit >= max(2, len(node_tokens) - 1)


class TemporalRetriever:
    def __init__(self, lance: Optional[LanceStore] = None, graph: Optional[TemporalGraph] = None):
        self.lance = lance if lance is not None else LanceStore()
        self.graph = graph if graph is not None else TemporalGraph()
        self.embedder: EmbeddingProvider = self.lance.embedder
# ------------------------------------------------------------------ core
    def retrieve(
        self,
        question: str,
        window_start: Optional[date] = None,
        window_end: Optional[date] = None,
        window_label: str = "",
        top_k: int = 6,
        hops: int = 2,
        max_graph_edges: int = 40,
    ) -> RetrievalContext:
        """Full temporal retrieval for one question."""
        trace: list[str] = []

        # --- 0. Resolve the window (explicit params win over NL parsing) -----
        # Tolerate callers passing the *label* instead of parsed dates.
        if isinstance(window_start, str) and window_end is None:
            parsed = parse_window(window_start)
            window_start, window_end, window_label = parsed.start, parsed.end, parsed.label or window_start
        if window_start is None and window_end is None:
            parsed = parse_window(question)
            window_start, window_end, window_label = parsed.start, parsed.end, parsed.label
            if not parsed.is_empty:
                trace.append(f"parsed window from question: '{window_label}'")

        if window_start or window_end:
            trace.append(f"temporal filter applied: [{window_start or '…'} → {window_end or '…'}]")
        else:
            trace.append("no temporal filter (open window)")

        iv = Interval(window_start, window_end)

        # --- 1. Vector lens ---------------------------------------------------
        qvec = self.embedder.embed(question)
        hits = self.lance.search(qvec, top_k=top_k, window_start=window_start, window_end=window_end)
        extra = 0
        # Chunks are tagged with a coarse *chunk-level* interval, so a document
        # whose coverage ends just before the window may still describe facts
        # that were in force *inside* it. Merge in an unwindowed refusal when the
        # windowed search is thin — the graph lens then still reaches those facts.
        if (window_start or window_end) and len(hits) < max(4, top_k):
            broad_hits = self.lance.search(qvec, top_k=max(8, top_k + 4))
            known = {h.chunk_id for h in hits}
            extra_hits = [h for h in broad_hits if h.chunk_id not in known]
            extra = len(extra_hits)
            hits.extend(extra_hits)
        trace.append(f"vector search: {len(hits)} chunk hits (top_k={top_k}" + (f", +{extra} broad-recall" if extra else "") + ")")

        # --- 2. Seed entities from hit chunks + question entities -------------------
        seed_entities: list[str] = []
        for hit in hits:
            for ent in hit.entities:
                if not any(token_overlap(ent, h) > 0.4 for h in seed_entities):
                    seed_entities.append(ent)
        # Entities *named in the question* anchor the traversal even when the
        # top-K chunks are thin — critical for temporal "as-of" audits.
        for name in self.graph.graph.nodes:
            if _question_matches_node(name, question):
                seed_entities.append(name)
        seed_entities = seed_entities[:12]
        if seed_entities:
            trace.append(f"seed entities: {seed_entities[:8]}")

        # --- 3. Graph lens: time-sliced traversal -----------------------------
        graph_edges: list[GraphEdge] = []
        seen: set[tuple[str, str, str]] = set()
        frontier = set(seed_entities)
        visited: set[str] = set()
        for _ in range(max(1, hops)):
            nxt: set[str] = set()
            for u, v, data in self.graph.graph.edges(data=True):
                if u not in frontier and v not in frontier:
                    continue
                if iv is not None and not edge_valid_within(data, iv):
                    continue
                key = (str(data.get("subject") or u), str(data.get("predicate") or ""), str(data.get("object") or v))
                if key in seen:
                    continue
                seen.add(key)
                graph_edges.append(self.graph._edge_to_model(data))
                nxt.add(u)
                nxt.add(v)
                if len(graph_edges) >= max_graph_edges:
                    break
            visited |= frontier
            frontier = nxt - visited
            if not frontier:
                break
        if graph_edges:
            trace.append(f"graph lens: {len(graph_edges)} interval-validated facts via {len(seed_entities)} seeds")

        return RetrievalContext(
            question=question,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label or self._fallback_label(window_start, window_end),
            hits=hits,
            graph_edges=graph_edges,
            seed_entities=seed_entities,
            trace=trace,
        )

    @staticmethod
    def _fallback_label(start: Optional[date], end: Optional[date]) -> str:
        if start is None and end is None:
            return ""
        if start == end and start:
            return f"{start.isoformat()} (as of)"
        return f"{start or '…'} → {end or '…'}"

    def compare(self, question: str, window_a_label: str, window_b_label: str, top_k: int = 6) -> tuple[RetrievalContext, RetrievalContext]:
        wa = parse_window(window_a_label)
        wb = parse_window(window_b_label)
        ctx_a = self.retrieve(question, wa.start, wa.end, wa.label or window_a_label, top_k=top_k)
        ctx_b = self.retrieve(question, wb.start, wb.end, wb.label or window_b_label, top_k=top_k)
        return ctx_a, ctx_b

    def diff_between_windows(self, window_a_label: str, window_b_label: str) -> dict:
        wa, wb = parse_window(window_a_label), parse_window(window_b_label)
        return self.graph.diff(wa.as_interval(), wb.as_interval())