"""LanceDB vector store for temporal document chunks.

Each chunk row carries the chunk text, provenance (doc id, page) and the
*extracted temporal validity interval* as ISO strings so that retrieval can
push date-window predicates all the way down into the vector index.

Note on dimensionality: LanceDB tables are fixed-schema, so switching the
embedding provider (e.g. hashing 512d → ollama 768d) recreates the table.
This is safe — vectors are re-embedded during reindex.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pyarrow as pa

import lancedb

from chronos.config import Settings, get_settings
from chronos.embeddings import EmbeddingProvider
from chronos.models import ChunkHit, TemporalValidity

log = logging.getLogger("chronos.storage.lancedb")

TABLE_NAME = "chunks"


class LanceStore:
    """Thin wrapper around a LanceDB table of embedded chunks."""

    def __init__(self, settings: Optional[Settings] = None, embedder: Optional[EmbeddingProvider] = None):
        self.settings = settings or get_settings()
        self.embedder = embedder or EmbeddingProvider(settings)
        self.db = lancedb.connect(str(self.settings.lance_db_path))
        self.table = self._open_or_create()

    # ------------------------------------------------------------------ utils
    def _dim(self) -> int:
        return self.embedder.dim

    @staticmethod
    def _iso(d: Optional[date]) -> Optional[str]:
        return d.isoformat() if d else None

    def _open_or_create(self):
        exists = self.db.table_names()
        if TABLE_NAME in exists:
            tbl = self.db.open_table(TABLE_NAME)
            schema = tbl.schema
            try:
                vec_field = schema.field("vector")
                dim = vec_field.type.list_size
            except Exception:  # pragma: no cover - schema introspection best effort
                dim = None
            if dim != self._dim():
                log.warning(
                    "Existing table vector dim (%s) != current (%s); recreating table.",
                    dim,
                    self._dim(),
                )
                self.db.drop_table(TABLE_NAME)
                return self._create()
            return tbl
        return self._create()

    def _create(self):
        dim = self._dim()
        schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("doc_name", pa.string()),
                pa.field("text", pa.string()),
                pa.field("page", pa.int32()),
                pa.field("start", pa.string()),
                pa.field("end", pa.string()),
                pa.field("start_label", pa.string()),
                pa.field("end_label", pa.string()),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        empty = pa.table(
            {
                "chunk_id": pa.array([], pa.string()),
                "doc_id": pa.array([], pa.string()),
                "doc_name": pa.array([], pa.string()),
                "text": pa.array([], pa.string()),
                "page": pa.array([], pa.int32()),
                "start": pa.array([], pa.string()),
                "end": pa.array([], pa.string()),
                "start_label": pa.array([], pa.string()),
                "end_label": pa.array([], pa.string()),
                "entities": pa.array([], pa.list_(pa.string())),
                "vector": pa.array([], pa.list_(pa.float32(), dim)),
            }
        )
        return self.db.create_table(TABLE_NAME, schema=schema, data=empty, mode="create")
# ------------------------------------------------------------------ write
    def add_chunks(
        self,
        doc_id: str,
        doc_name: str,
        chunks: list[dict],
        texts: list[str],
        validity: list[TemporalValidity],
        pages: list[int],
        entities: list[list[str]],
    ) -> int:
        """Insert pre-embedded chunks into LanceDB."""
        vecs = self.embedder.embed_batch(texts)
        rows = pa.table(
            {
                "chunk_id": pa.array([c["chunk_id"] for c in chunks], pa.string()),
                "doc_id": pa.array([doc_id] * len(chunks), pa.string()),
                "doc_name": pa.array([doc_name] * len(chunks), pa.string()),
                "text": pa.array(texts, pa.string()),
                "page": pa.array([int(p) for p in pages], pa.int32()),
                "start": pa.array([self._iso(v.start) for v in validity], pa.string()),
                "end": pa.array([self._iso(v.end) for v in validity], pa.string()),
                "start_label": pa.array([v.start_label or "" for v in validity], pa.string()),
                "end_label": pa.array([v.end_label or "" for v in validity], pa.string()),
                "entities": pa.array([list(e) for e in entities], pa.list_(pa.string())),
                "vector": pa.array([list(map(float, v)) for v in vecs], pa.list_(pa.float32(), self._dim())),
            }
        )
        if rows.num_rows > 0:
            self.table.add(rows)
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        self.table.delete(f"doc_id = '{doc_id}'")

    # ------------------------------------------------------------------ read
    def _temporal_filter(self, window_start: Optional[date], window_end: Optional[date]) -> Optional[str]:
        """LanceDB SQL predicate for chunks overlapping [start, end]."""
        if window_start is None and window_end is None:
            return None
        clauses: list[str] = []
        if window_start is not None:
            clauses.append(f"(end IS NULL OR end >= '{window_start.isoformat()}')")
        if window_end is not None:
            clauses.append(f"(start IS NULL OR start <= '{window_end.isoformat()}')")
        return " AND ".join(clauses)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 6,
        window_start: Optional[date] = None,
        window_end: Optional[date] = None,
    ) -> list[ChunkHit]:
        q = self.table.search(query_vec.tolist())
        expr = self._temporal_filter(window_start, window_end)
        if expr:
            q = q.where(expr)
        try:
            results = q.limit(top_k).to_list()
        except Exception as exc:
            log.warning("LanceDB search failed (%s); returning empty hits.", exc)
            return []
        hits: list[ChunkHit] = []
        for r in results:
            hits.append(
                ChunkHit(
                    chunk_id=r.get("chunk_id") or "",
                    doc_id=r.get("doc_id") or "",
                    doc_name=r.get("doc_name") or "",
                    text=r.get("text") or "",
                    page=int(r.get("page") or 0),
                    start=date.fromisoformat(r["start"]) if r.get("start") else None,
                    end=date.fromisoformat(r["end"]) if r.get("end") else None,
                    start_label=r.get("start_label") or None,
                    end_label=r.get("end_label") or None,
                    score=float(r.get("_distance") or 0.0),
                    entities=list(r.get("entities") or []),
                )
            )
        hits.sort(key=lambda h: h.score)
        return hits

    def count(self) -> int:
        try:
            return self.table.count_rows()
        except Exception:  # pragma: no cover
            return 0

    def doc_ids(self) -> list[str]:
        try:
            return sorted({str(r["doc_id"]) for r in self.table.to_list()})
        except Exception:  # pragma: no cover
            return []