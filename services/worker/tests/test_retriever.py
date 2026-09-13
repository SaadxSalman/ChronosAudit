"""Integration-style test of the hybrid temporal retriever against a real
LanceDB (tmp) + NetworkX graph built from deterministic extraction."""

import math
from datetime import date

from chronos.config import Settings
from chronos.embeddings import EmbeddingProvider
from chronos.extraction.chunker import Page, chunk_document
from chronos.extraction.fallback import extract_fallback
from chronos.retrieval.retriever import TemporalRetriever
from chronos.storage.graph_store import TemporalGraph
from chronos.storage.lancedb_store import LanceStore

_TEXT_A = (
    "The Data Transfer Policy requires data localization for Personal Data "
    "effective 2022-01-01 until 2024-06-30. "
    "The Data Transfer Policy permits cross-border data transfers when the recipient "
    "implements Standard Contractual Clauses starting 2024-07-01. "
    "Cross-Border Data Regulation governs the transfer of Personal Data within the EU."
)
_TEXT_B = (
    "The Tax Reform Act imposes a 15 percent corporate minimum tax on Large "
    "Multinational Groups from 2023-01-01. "
    "The Tax Reform Act exempts small domestic companies until 2025-12-31. "
    "Corporate Finance Act amends the Tax Reform Act effective 2024-10-01."
)


def build_env(tmp_path):
    settings = Settings(
        model_provider="none",
        embedding_provider="hashing",
        embedding_dim=128,
        lance_db_path=str(tmp_path / "lancedb"),
        graph_store_path=str(tmp_path / "graph.gpickle"),
    )
    embedder = EmbeddingProvider(settings)
    lance = LanceStore(settings, embedder)
    graph = TemporalGraph(settings)

    for doc_id, text in (("doc-a", _TEXT_A), ("doc-b", _TEXT_B)):
        chunks = chunk_document([Page(1, text)], doc_id, chunk_size_tokens=200, overlap_tokens=30)
        extractions = []
        for c in chunks:
            er = extract_fallback(c.text, chunk_id=c.chunk_id, doc_name=doc_id)
            extractions.append(er)
        lance.add_chunks(
            doc_id, doc_id,
            [{"chunk_id": c.chunk_id} for c in chunks],
            [c.text for c in chunks],
            [er.chunk_temporal for er in extractions],
            [1] * len(chunks),
            [list({e.name for e in er.entities}) for er in extractions],
        )
        graph.upsert_document(doc_id, doc_id, extractions)
    return TemporalRetriever(lance, graph)


def test_retrieve_window_filters_vector_and_graph(tmp_path):
    retriever = build_env(tmp_path)

    # In Q3 2024 the localization requirement (ended 2024-06-30) must NOT appear.
    ctx = retriever.retrieve("What does the Data Transfer Policy require?", "Q3 2024", top_k=5)
    assert any("Standard Contractual Clauses" in h.text for h in ctx.hits)
    assert any(e.predicate == "permits" for e in ctx.graph_edges)
    # localization fact ended before Q3 2024 → absent from the windowed graph facts
    assert not any(e.predicate == "requires" and "localization" in (e.evidence or "").lower() for e in ctx.graph_edges)

    # Open window should include BOTH the old and new regime at vector level
    ctx_open = retriever.retrieve("Data transfer compliance", top_k=10)
    assert len(ctx_open.hits) >= 2


def test_retrieve_compares_windows(tmp_path):
    retriever = build_env(tmp_path)
    ctx_a, ctx_b = retriever.compare("What is required for cross-border transfers?", "Q1 2022", "Q4 2024")
    sig_a = {(e.subject, e.predicate, e.object) for e in ctx_a.graph_edges}
    sig_b = {(e.subject, e.predicate, e.object) for e in ctx_b.graph_edges}
    # "permits" only appears in B; "requires localization" only in A
    assert any(pred == "permits" for _, pred, _ in sig_a) or any(pred == "permits" for _, pred, _ in sig_b)
    assert ctx_a.window_label and ctx_b.window_label