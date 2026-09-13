"""Celery ingestion task.

Runs the exact same ``Pipeline.run_ingestion`` used by the synchronous bridge,
so results are identical regardless of queueing mode.
"""

from __future__ import annotations

import logging
import uuid

from chronos.celery_app import celery_app
from chronos.pipeline import get_pipeline

log = logging.getLogger("chronos.tasks.ingest")


@celery_app.task(name="chronos.ingest_document", bind=True, max_retries=2)
def ingest_document(self, doc_id: str, filename: str, pdf_path: str) -> dict:
    """Enqueue a full ingestion job. ``doc_id`` may be empty to autogenerate."""
    doc_id = doc_id or f"doc-{uuid.uuid4().hex[:12]}"
    try:
        pipe = get_pipeline()
        record = pipe.run_ingestion(doc_id, filename, pdf_path)
        return {"doc_id": doc_id, "status": record.status.value, "error": record.error}
    except Exception as exc:  # noqa: BLE001
        log.exception("Celery ingestion failed for %s", doc_id)
        raise self.retry(exc=exc, countdown=5)