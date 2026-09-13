"""Pydantic domain models shared across the worker.

These mirror the JSON contracts consumed by the TypeScript/Express layer so
that a single source of truth drives both the Python internals and the API.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class TemporalValidity(BaseModel):
    """Explicit temporal validity interval attached to a fact.

    * ``start``/``end`` are ISO dates; either side may be *open* (``None``)
      to express "valid from X onwards" or "valid until X".
    * The labels keep the human-readable expressions found in the source text
      (e.g. ``"Q3 2024"``) for citation purposes.
    """

    start: Optional[date] = None
    end: Optional[date] = None
    start_label: Optional[str] = None
    end_label: Optional[str] = None

    @property
    def is_open_ended(self) -> bool:
        return self.end is None

    @property
    def is_open_started(self) -> bool:
        return self.start is None

    def overlaps(self, other: "TemporalValidity") -> bool:
        a0, a1 = self.start or date.min, self.end or date.max
        b0, b1 = other.start or date.min, other.end or date.max
        return a0 <= b1 and b0 <= a1

    def contains(self, d: date) -> bool:
        start_ok = self.start is None or d >= self.start
        end_ok = self.end is None or d <= self.end
        return start_ok and end_ok

    def __str__(self) -> str:  # pragma: no coverage - cosmetic only
        lhs = self.start_label or (self.start.isoformat() if self.start else "…")
        rhs = self.end_label or (self.end.isoformat() if self.end else "…")
        return f"{lhs} → {rhs}" if self.start or self.end else "untemporal"


class EntityOccurrence(BaseModel):
    """One extracted entity mention with provenance."""

    id: str
    name: str
    type: str = "UNKNOWN"  # e.g. Regulation, Organization, Concept, Party, Jurisdiction…
    aliases: list[str] = Field(default_factory=list)
    sentence: str = ""
    confidence: float = 0.5


class RelationOccurrence(BaseModel):
    """One extracted (subject → predicate → object) triple with validity."""

    subject: str
    predicate: str
    object: str
    validity: TemporalValidity = Field(default_factory=TemporalValidity)
    sentence: str = ""
    confidence: float = 0.5

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject.lower().strip(), self.predicate.lower().strip(), self.object.lower().strip())


class ExtractionResult(BaseModel):
    """Structured output of the extraction SLM for a single chunk."""

    chunk_id: str
    entities: list[EntityOccurrence] = Field(default_factory=list)
    relations: list[RelationOccurrence] = Field(default_factory=list)
    chunk_temporal: TemporalValidity = Field(default_factory=TemporalValidity)
    raw_json: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    """Metadata for an ingested document."""

    id: str
    filename: str
    path: str
    pages: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    size_bytes: int = 0
    status: DocumentStatus = DocumentStatus.UPLOADED
    progress: float = 0.0
    stage_detail: str = ""
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkHit(BaseModel):
    """A single vector-search hit with temporal + provenance metadata."""

    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    page: int = 0
    start: Optional[date] = None
    end: Optional[date] = None
    start_label: Optional[str] = None
    end_label: Optional[str] = None
    score: float = 0.0
    entities: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    subject: str
    predicate: str
    object: str
    start: Optional[date] = None
    end: Optional[date] = None
    start_label: Optional[str] = None
    end_label: Optional[str] = None
    doc_id: str = ""
    chunk_id: str = ""
    evidence: str = ""
    weight: float = 1.0


class GraphSnapshot(BaseModel):
    valid_at: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    edge_count: int = 0
    node_count: int = 0


class RetrievalContext(BaseModel):
    """Everything the retriever found for a question, ready for synthesis."""

    question: str
    window_start: Optional[date] = None
    window_end: Optional[date] = None
    window_label: str = ""
    hits: list[ChunkHit] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    seed_entities: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    context: Optional[RetrievalContext] = None
    engine: str = "llm"
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class Comparison(BaseModel):
    window_start: Optional[date] = None
    window_end: Optional[date] = None
    window_label: str = ""
    added: list[GraphEdge] = Field(default_factory=list)
    removed: list[GraphEdge] = Field(default_factory=list)
    changed: list[dict[str, Any]] = Field(default_factory=list)
    unchanged: list[GraphEdge] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)


class CompareResponse(BaseModel):
    question: str
    window_a_label: str = ""
    window_b_label: str = ""
    comparison: Optional[Comparison] = None
    context_a: Optional[RetrievalContext] = None
    context_b: Optional[RetrievalContext] = None
    narrative: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class GraphStats(BaseModel):
    node_count: int = 0
    edge_count: int = 0
    document_count: int = 0
    entity_types: dict[str, int] = Field(default_factory=dict)
    predicates: dict[str, int] = Field(default_factory=dict)
    temporally_validated_edges: int = 0
    open_ended_edges: int = 0
    temporal_range_start: Optional[date] = None
    temporal_range_end: Optional[date] = None
    chunk_vector_count: int = 0
    vector_dim: int = 0