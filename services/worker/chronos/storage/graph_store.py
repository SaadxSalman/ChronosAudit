"""Temporal knowledge graph built on NetworkX.

The graph stores *entities* as nodes and interval-validated *facts* as edges
(``MultiDiGraph`` so the same pair of entities can carry several predicates,
and the same predicate can carry several temporal vintages).

Edge attributes: ``predicate, subject, object`` (canonical triple),
``valid_from, valid_until`` (date bounds, None = open), ``start_label`` /
``end_label`` (human-readable validity), ``doc_id, chunk_id`` (provenance) and
``evidence`` (verbatim supporting sentence).

Because every edge is interval-bounded the graph answers the canonical
temporal questions an auditor asks:
1. *as-of*     — what was true on date D?         (``snapshot``/``neighbors``)
2. *during*    — what was true in window [A, B]?  (``snapshot(overlap)``)
3. *evolution* — how did fact X change over time? (``evolution``/``diff``)
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Optional

import networkx as nx

from chronos.config import Settings, get_settings
from chronos.models import (
    ExtractionResult,
    GraphEdge,
    GraphSnapshot,
    GraphStats,
    RelationOccurrence,
)
from chronos.temporal.interval import Interval, diff_snapshots, edge_valid_within

log = logging.getLogger("chronos.storage.graph")


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


class TemporalGraph:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.doc_meta: dict[str, dict] = {}
        self.load()

    # ============================================================ persistence
    def save(self) -> None:
        path = Path(self.settings.graph_store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self.graph, protocol=pickle.HIGHEST_PROTOCOL))
        (path.parent / "doc_meta.json").write_text(
            json.dumps(self.doc_meta, default=str, indent=2), encoding="utf-8"
        )

    def load(self) -> None:
        path = Path(self.settings.graph_store_path)
        if path.exists():
            try:
                self.graph = pickle.loads(path.read_bytes())
            except Exception as exc:  # pragma: no cover
                log.warning("Could not load graph pickle (%s); starting fresh.", exc)
                self.graph = nx.MultiDiGraph()
        meta_path = path.parent / "doc_meta.json"
        if meta_path.exists():
            try:
                self.doc_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover
                self.doc_meta = {}

    # ============================================================ mutation
    def upsert_document(self, doc_id: str, filename: str, extractions: Iterable[ExtractionResult]) -> dict:
        """Remove any prior traces of ``doc_id`` and re-insert its facts."""
        self.remove_document(doc_id)
        entity_names: Counter = Counter()
        relation_count = 0

        for er in extractions:
            for ent in er.entities:
                name = _norm(ent.name)
                if not name:
                    continue
                entity_names[name] += 1
                if self.graph.has_node(name):
                    node = self.graph.nodes[name]
                    node["doc_ids"] = sorted(set(node.get("doc_ids", [])) | {doc_id})
                    if node.get("type") in (None, "UNKNOWN", "Other"):
                        node["type"] = ent.type or "Other"
                else:
                    self.graph.add_node(
                        name,
                        type=ent.type or "Other",
                        aliases=[a for a in ent.aliases if a],
                        doc_ids=[doc_id],
                    )

            for rel in er.relations:
                self._upsert_relation(rel, doc_id, er.chunk_id)
                relation_count += 1

        for name, count in entity_names.items():
            if self.graph.has_node(name):
                node = self.graph.nodes[name]
                node["mention_count"] = node.get("mention_count", 0) + count

        self.doc_meta[doc_id] = {
            "filename": filename,
            "entity_count": len(entity_names),
            "relation_count": relation_count,
        }
        self.save()
        return {"doc_id": doc_id, "entities": len(entity_names), "relations": relation_count}

    def _upsert_relation(self, rel: RelationOccurrence, doc_id: str, chunk_id: str) -> None:
        subj = _norm(rel.subject)
        obj = _norm(rel.object)
        if not subj or not obj:
            return
        for n in (subj, obj):
            if not self.graph.has_node(n):
                self.graph.add_node(n, type="Other", doc_ids=[doc_id])
        edge_data = {
            "predicate": rel.predicate,
            "subject": subj,
            "object": obj,
            "valid_from": rel.validity.start,
            "valid_until": rel.validity.end,
            "start_label": rel.validity.start_label,
            "end_label": rel.validity.end_label,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "evidence": rel.sentence,
            "confidence": rel.confidence,
        }
        self.graph.add_edge(subj, obj, key=f"{doc_id}:{chunk_id}:{rel.predicate}", **edge_data)

    def remove_document(self, doc_id: str) -> None:
        to_remove = [
            (u, v, k)
            for u, v, k, d in self.graph.edges(keys=True, data=True)
            if d.get("doc_id") == doc_id
        ]
        self.graph.remove_edges_from(to_remove)
        orphaned = [n for n in self.graph.nodes if self.graph.degree(n) == 0]
        self.graph.remove_nodes_from(orphaned)
        self.doc_meta.pop(doc_id, None)
        self.save()
# ============================================================ queries
    def iter_edges(self, valid_at: Optional[date] = None, window: Optional[Interval] = None) -> Iterator[dict]:
        """Iterate edges, optionally filtered by temporal validity."""
        iv = window
        if iv is None and valid_at is not None:
            iv = Interval(valid_at, valid_at)
        if iv is None:
            for _, _, data in self.graph.edges(data=True):
                yield data
            return
        for _, _, data in self.graph.edges(data=True):
            if edge_valid_within(data, iv):
                yield data

    @staticmethod
    def _edge_to_model(data: dict) -> GraphEdge:
        return GraphEdge(
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object=data.get("object", ""),
            start=data.get("valid_from"),
            end=data.get("valid_until"),
            start_label=data.get("start_label"),
            end_label=data.get("end_label"),
            doc_id=data.get("doc_id", ""),
            chunk_id=data.get("chunk_id", ""),
            evidence=data.get("evidence", ""),
            weight=float(data.get("confidence", 0.5)),
        )

    def snapshot(
        self,
        valid_at: Optional[date] = None,
        window: Optional[Interval] = None,
        node_filter: Optional[set[str]] = None,
        max_edges: int = 1000,
    ) -> GraphSnapshot:
        """A temporal slice of the graph as a JSON-safe snapshot."""
        edges: list[GraphEdge] = []
        involved: set[str] = set()
        for data in self.iter_edges(valid_at, window):
            s, o = data.get("subject", ""), data.get("object", "")
            if node_filter is not None and s not in node_filter and o not in node_filter:
                continue
            edges.append(self._edge_to_model(data))
            involved.add(s)
            involved.add(o)
            if len(edges) >= max_edges:
                break
        nodes = []
        for n in sorted(involved):
            nd = self.graph.nodes.get(n, {})
            nodes.append(
                {
                    "id": n,
                    "name": n,
                    "type": nd.get("type", "Other"),
                    "doc_ids": nd.get("doc_ids", []),
                    "mention_count": nd.get("mention_count", 0),
                    "degree": self.graph.degree(n),
                }
            )
        return GraphSnapshot(
            valid_at=valid_at.isoformat() if valid_at else None,
            start=window.start.isoformat() if window and window.start else None,
            end=window.end.isoformat() if window and window.end else None,
            nodes=nodes,
            edges=edges,
            edge_count=len(edges),
            node_count=len(nodes),
        )

    def neighbors(self, entity: str, valid_at: Optional[date] = None, window: Optional[Interval] = None, hop: int = 1) -> list[GraphEdge]:
        """Facts within ``hop`` hops of an entity, filtered by validity."""
        entity = _norm(entity)
        if not self.graph.has_node(entity):
            return []
        frontier = {entity}
        found_edges: list[GraphEdge] = []
        iv = window if window is not None else (Interval(valid_at, valid_at) if valid_at else None)
        for _ in range(max(1, min(hop, 3))):
            next_frontier: set[str] = set()
            for u, v, data in self.graph.edges(data=True):
                if u not in frontier and v not in frontier:
                    continue
                if iv is not None and not edge_valid_within(data, iv):
                    continue
                found_edges.append(self._edge_to_model(data))
                next_frontier.add(u)
                next_frontier.add(v)
            if not next_frontier - frontier:
                break
            frontier |= next_frontier
        return found_edges

    def evolution(self, entity: str, predicate: Optional[str] = None) -> list[dict]:
        """History of the fact(s) touching ``entity`` with full intervals."""
        entity = _norm(entity)
        out: list[dict] = []
        for data in self.iter_edges():
            if entity not in (data.get("subject"), data.get("object")):
                continue
            if predicate and data.get("predicate") != predicate:
                continue
            out.append(
                {
                    "subject": data.get("subject"),
                    "predicate": data.get("predicate"),
                    "object": data.get("object"),
                    "start": data.get("valid_from"),
                    "end": data.get("valid_until"),
                    "start_label": data.get("start_label"),
                    "end_label": data.get("end_label"),
                    "doc_id": data.get("doc_id"),
                    "evidence": (data.get("evidence") or "")[:240],
                }
            )
        out.sort(key=lambda r: (r["start"] or date.min, r["end"] or date.max))
        return out

    def diff(self, window_a: Interval, window_b: Interval) -> dict:
        edges_a = list(self.iter_edges(window=window_a))
        edges_b = list(self.iter_edges(window=window_b))
        return diff_snapshots(edges_a, edges_b, window_a, window_b)

    def stats(self) -> GraphStats:
        edges = list(self.graph.edges(data=True))
        validated = 0
        open_ended = 0
        starts: list[date] = []
        ends: list[date] = []
        for _, _, d in edges:
            if d.get("valid_from"):
                validated += 1
                starts.append(d["valid_from"])
            if d.get("valid_until"):
                ends.append(d["valid_until"])
            else:
                open_ended += 1
        return GraphStats(
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
            document_count=len(self.doc_meta),
            entity_types=dict(Counter(str(d.get("type") or "Other") for _, d in self.graph.nodes(data=True))),
            predicates=dict(Counter(str(d.get("predicate") or "") for _, _, d in self.graph.edges(data=True))),
            temporally_validated_edges=validated,
            open_ended_edges=open_ended,
            temporal_range_start=min(starts) if starts else None,
            temporal_range_end=max(ends) if ends else None,
        )

    def export_graphml(self, path: str) -> None:
        """Dump the full graph in GraphML for external tools (Gephi etc.)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, path)