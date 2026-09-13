"""FastAPI bridge for the ChronosAudit worker.

The TypeScript/Express layer talks to this bridge (never to celery/lancedb
directly). In ``WORKER_MODE=sync`` ingestion runs in-process; in ``async``
mode the bridge enqueues a Celery task.

Docs are available at http://localhost:8100/docs (Swagger UI).
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from chronos.celery_app import celery_app
from chronos.config import get_settings
from chronos.pipeline import get_pipeline
from chronos.retrieval.qa import AnswerEngine
from chronos.retrieval.retriever import TemporalRetriever
from chronos.temporal.interval import Interval
from chronos.temporal.parser import parse_window

app = FastAPI(title="ChronosAudit Worker Bridge", version="1.0.0")
settings = get_settings()


# ================================================================ request DTOs
class IngestRequest(BaseModel):
    doc_id: Optional[str] = None
    filename: str = "document.pdf"
    path: str
    async_: bool = True  # True → Celery, False → run in-process


class QueryRequest(BaseModel):
    question: str
    window: Optional[str] = None
    top_k: int = 6
    hops: int = 2


class CompareRequest(BaseModel):
    question: str
    window_a: str
    window_b: str
    top_k: int = 6


# ================================================================ deps
_retriever: Optional[TemporalRetriever] = None
_answer_engine: Optional[AnswerEngine] = None


def get_retriever() -> TemporalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TemporalRetriever()
    return _retriever


def get_answer_engine() -> AnswerEngine:
    global _answer_engine
    if _answer_engine is None:
        _answer_engine = AnswerEngine()
    return _answer_engine


def _auth_ok(authorization: Optional[str]) -> bool:
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not settings.worker_api_token or settings.worker_api_token == "change-me-dev-token":
        return True  # dev default: token disabled
    return token == settings.worker_api_token


def _require_auth(authorization: Optional[str] = Header(None)) -> None:
    if not _auth_ok(authorization):
        raise HTTPException(status_code=401, detail="Invalid worker API token")


# ================================================================ health
@app.get("/health")
def health():
    pipe = get_pipeline()
    return {
        "ok": True,
        "service": "chronos-worker-bridge",
        "mode": settings.worker_mode,
        "model_provider": settings.model_provider,
        "embedding_provider": pipe.embedder.kind,
        "embedding_dim": pipe.embedder.dim,
        "lancedb_chunks": pipe.lance.count(),
        "graph_nodes": pipe.graph.graph.number_of_nodes(),
        "graph_edges": pipe.graph.graph.number_of_edges(),
        "documents": len(pipe.status.list()),
        "broker": settings.celery_broker_url.split("@")[-1],
    }


# ================================================================ documents
@app.post("/documents/ingest")
def ingest_document(req: IngestRequest, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    from chronos.tasks.ingest import ingest_document as celery_task

    doc_id = req.doc_id or f"doc-{uuid.uuid4().hex[:12]}"
    use_async = req.async_ and settings.worker_mode == "async"
    if use_async:
        celery_task.delay(doc_id, req.filename, req.path)
        return {"doc_id": doc_id, "queued": True, "mode": "celery"}
    pipe = get_pipeline()
    record = pipe.run_ingestion(doc_id, req.filename, req.path)
    return {"doc_id": doc_id, "queued": False, "mode": "sync", "status": record.status.value, "error": record.error}

# ================================================================ query
@app.post("/query")
def query(req: QueryRequest, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question is required")
    retriever = get_retriever()
    parsed = parse_window(req.window)
    ctx = retriever.retrieve(
        req.question,
        parsed.start,
        parsed.end,
        parsed.label or (req.window or ""),
        top_k=req.top_k,
        hops=req.hops,
    )
    result = get_answer_engine().answer(req.question, ctx)
    return result.model_dump(mode="json")


@app.post("/query/compare")
def compare(req: CompareRequest, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    retriever = get_retriever()
    ctx_a, ctx_b = retriever.compare(req.question, req.window_a, req.window_b, top_k=req.top_k)
    result = get_answer_engine().compare(req.question, ctx_a, ctx_b)
    return result.model_dump(mode="json")


@app.post("/graph/diff")
def graph_diff(body: dict, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    a = str(body.get("window_a", ""))
    b = str(body.get("window_b", ""))
    result = get_retriever().diff_between_windows(a, b)
    return result


# ================================================================ graph
@app.get("/graph/stats")
def graph_stats(authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    stats = get_pipeline().graph.stats()
    stats.chunk_vector_count = get_pipeline().lance.count()
    stats.vector_dim = get_pipeline().embedder.dim
    return stats.model_dump(mode="json")


@app.get("/graph/snapshot")
def graph_snapshot(
    valid_at: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    max_edges: int = Query(500, ge=1, le=5000),
    authorization: Optional[str] = Header(None),
):
    _require_auth(authorization)
    pipe = get_pipeline()
    if valid_at:
        snapshot = pipe.graph.snapshot(valid_at=date.fromisoformat(valid_at), max_edges=max_edges)
    else:
        window = Interval(
            date.fromisoformat(start) if start else None,
            date.fromisoformat(end) if end else None,
        )
        snapshot = pipe.graph.snapshot(window=window, max_edges=max_edges)
    return snapshot.model_dump(mode="json")


@app.get("/graph/neighbors")
def graph_neighbors(
    entity: str = Query(...),
    valid_at: Optional[str] = Query(None),
    hop: int = Query(1, ge=1, le=3),
    authorization: Optional[str] = Header(None),
):
    _require_auth(authorization)
    _valid_at = date.fromisoformat(valid_at) if valid_at else None
    edges = get_pipeline().graph.neighbors(entity, valid_at=_valid_at, hop=hop)
    return [e.model_dump(mode="json") for e in edges]


@app.get("/graph/evolution")
def graph_evolution(
    entity: str = Query(...),
    predicate: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    _require_auth(authorization)
    return get_pipeline().graph.evolution(entity, predicate)


@app.get("/graph/export")
def graph_export(gml: bool = Query(False), authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    pipe = get_pipeline()
    if gml:
        path = str(settings.graph_export_path / "chronos.graphml")
        pipe.graph.export_graphml(path)
    else:
        path = str(settings.graph_export_path / "chronos.json")
        snap = pipe.graph.snapshot(max_edges=100_000)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(snap.model_dump_json(indent=2), encoding="utf-8")
    return {"exported": path}


# ================================================================ system
@app.get("/system/info")
def system_info(authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    pipe = get_pipeline()
    return {
        "version": "1.0.0",
        "worker_mode": settings.worker_mode,
        "model_provider": settings.model_provider,
        "extraction_model": settings.model_name,
        "embedding_provider": pipe.embedder.kind,
        "embedding_dim": pipe.embedder.dim,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "lance_db_path": str(settings.lance_db_path),
        "graph_store_path": str(settings.graph_store_path),
    }


# Celery app must be imported for ``celery -A chronos.celery_app`` to work when
# this module is imported by uvicorn (registration of tasks happens via include).
_celery_app = celery_app  # noqa: F841

@app.get("/documents")
def list_documents(authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    return get_pipeline().status.list()


@app.get("/documents/{doc_id}/status")
def document_status(doc_id: str, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    rec = get_pipeline().status.get(doc_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"No such document: {doc_id}")
    return rec


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    pipe = get_pipeline()
    pipe.status.delete(doc_id)
    pipe.lance.delete_document(doc_id)
    pipe.graph.remove_document(doc_id)
    return {"deleted": doc_id}