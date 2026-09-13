"""Ingestion pipeline: PDF → pages → chunks → extraction → vectors → graph.

This module is the single source of truth for the *stages* of an ingestion
job. Both the Celery task and the synchronous bridge API call
``run_ingestion`` so that sync and async modes behave identically.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

from chronos.config import Settings, get_settings
from chronos.embeddings import EmbeddingProvider
from chronos.extraction.chunker import Page, chunk_document
from chronos.extraction.fallback import infer_chunk_window
from chronos.extraction.slm import ExtractionSLM
from chronos.models import DocumentRecord, DocumentStatus
from chronos.storage.graph_store import TemporalGraph
from chronos.storage.lancedb_store import LanceStore

log = logging.getLogger("chronos.pipeline")


class StatusStore:
    """Thread/process-safe job status registry.

    In ``sync`` mode this is a plain in-memory dict (persisted to disk). In
    ``async`` mode the Celery result backend (Redis) is the source of truth
    and this dict acts as a local cache/fallback.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._lock = RLock()
        self._data: dict[str, dict] = {}
        self._load_disk()

    def _state_path(self, doc_id: str) -> Path:
        return Path(self.settings.doc_state_dir) / f"{doc_id}.json"

    def _load_disk(self) -> None:
        d = Path(self.settings.doc_state_dir)
        if not d.exists():
            return
        for p in d.glob("*.json"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                self._data[rec["id"]] = rec
            except Exception:  # pragma: no cover
                continue

    def get(self, doc_id: str) -> Optional[dict]:
        with self._lock:
            return self._data.get(doc_id)

    def set(self, doc_id: str, record: dict) -> None:
        with self._lock:
            self._data[doc_id] = record
            try:
                self._state_path(doc_id).write_text(
                    json.dumps(record, default=str, indent=2), encoding="utf-8"
                )
            except Exception as exc:  # pragma: no cover
                log.warning("Could not persist status for %s: %s", doc_id, exc)

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(
                self._data.values(),
                key=lambda r: str(r.get("created_at", "")),
                reverse=True,
            )

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._data.pop(doc_id, None)
            try:
                self._state_path(doc_id).unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                pass
class Pipeline:
    """Composes every worker component for a single ingestion job."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.embedder = EmbeddingProvider(self.settings)
        self.slm = ExtractionSLM(self.settings)
        self.lance = LanceStore(self.settings, self.embedder)
        self.graph = TemporalGraph(self.settings)
        self.status = StatusStore(self.settings)

    # -------------------------------------------------------------- pdf util
    def parse_pdf(self, pdf_path: str) -> tuple[list[Page], str]:
        """Extract pages from a PDF via PyMuPDF. Returns (pages, raw_text)."""
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages: list[Page] = []
        all_text: list[str] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            all_text.append(text)
            pages.append(Page(number=i, text=text))
        doc.close()
        return pages, "\n".join(all_text)

    # ------------------------------------------------------------ run
    def run_ingestion(self, doc_id: str, filename: str, pdf_path: str) -> DocumentRecord:
        """Execute a full ingestion job, updating status as it goes."""
        record = DocumentRecord(
            id=doc_id,
            filename=filename,
            path=pdf_path,
            size_bytes=os.path.getsize(pdf_path),
            status=DocumentStatus.QUEUED,
        )
        self.status.set(doc_id, record.model_dump(mode="json"))

        try:
            # --- Stage 1: parse ----------------------------------------------
            self._update(record, DocumentStatus.PARSING, 0.10, "parsing PDF pages")
            pages, raw = self.parse_pdf(pdf_path)
            if not raw.strip():
                raise ValueError("PDF contains no extractable text")

            # --- Stage 2: chunk ----------------------------------------------
            self._update(record, DocumentStatus.CHUNKING, 0.20, f"{len(pages)} pages → chunks")
            chunks = chunk_document(
                pages,
                doc_id,
                chunk_size_tokens=self.settings.chunk_size_tokens,
                overlap_tokens=self.settings.chunk_overlap_tokens,
            )
            if not chunks:
                raise ValueError("No chunks could be produced")

            # --- Stage 3: extract --------------------------------------------
            self._update(record, DocumentStatus.EXTRACTING, 0.35, f"{len(chunks)} chunks through SLM")
            extractions = []
            for i, ch in enumerate(chunks):
                er, engine = self.slm.extract(ch.text, ch.chunk_id, doc_name=filename, page=ch.first_page())
                if er.chunk_temporal.start is None and er.chunk_temporal.end is None:
                    er = er.model_copy(update={"chunk_temporal": infer_chunk_window(ch.text)})
                extractions.append(er)
                if (i + 1) % 5 == 0:
                    self._update(
                        record,
                        DocumentStatus.EXTRACTING,
                        0.35 + 0.25 * ((i + 1) / len(chunks)),
                        f"extracting {i + 1}/{len(chunks)} (engine={engine})",
                    )
# --- Stage 4: embed + store vectors ------------------------------
            self._update(record, DocumentStatus.EMBEDDING, 0.65, "embedding chunks")
            texts = [c.text for c in chunks]
            validity = [er.chunk_temporal for er in extractions]
            pages_idx = [c.first_page() for c in chunks]
            entities = [list({e.name for e in er.entities}) for er in extractions]
            chunk_meta = [{"chunk_id": c.chunk_id} for c in chunks]
            self.lance.add_chunks(doc_id, filename, chunk_meta, texts, validity, pages_idx, entities)

            # --- Stage 5: build graph ----------------------------------------
            self._update(record, DocumentStatus.INDEXING, 0.85, "upserting knowledge graph")
            summary = self.graph.upsert_document(doc_id, filename, extractions)

            record = DocumentRecord(
                id=doc_id,
                filename=filename,
                path=pdf_path,
                pages=len(pages),
                chunk_count=len(chunks),
                entity_count=summary["entities"],
                relation_count=summary["relations"],
                size_bytes=record.size_bytes,
                status=DocumentStatus.INDEXED,
                progress=1.0,
                stage_detail="ingestion complete",
            )
            self.status.set(doc_id, record.model_dump(mode="json"))
            log.info(
                "Ingestion complete: %s (%d chunks, %d entities, %d relations)",
                doc_id, len(chunks), summary["entities"], summary["relations"],
            )
            return record
        except Exception as exc:  # noqa: BLE001
            log.exception("Ingestion failed for %s", doc_id)
            record.status = DocumentStatus.FAILED
            record.stage_detail = "ingestion failed"
            record.error = str(exc)[:500]
            self.status.set(doc_id, record.model_dump(mode="json"))
            return record

    def _update(self, record: DocumentRecord, status: DocumentStatus, progress: float, detail: str) -> None:
        record.status = status
        record.progress = progress
        record.stage_detail = detail
        record.updated_at = datetime.now(timezone.utc)
        self.status.set(doc_id=record.id, record=record.model_dump(mode="json"))


# module-level singleton for convenience in tasks/API
_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline