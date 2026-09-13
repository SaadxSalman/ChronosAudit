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
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

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
        nx.write_gpickle(self.graph, str(path))
        (path.parent / "doc_meta.json").write_text(
            json.dumps(self.doc_meta, default=str, indent=2), encoding="utf-8"
        )

    def load(self) -> None:
        path = Path(self.settings.graph_store_path)
        if path.exists():
            try:
                self.graph = nx.read_gpickle(str(path))
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