"""End-to-end pipeline test with a real (in-memory generated) PDF.

Uses the deterministic fallback extractor + hashing embeddings so the test
needs no API keys, no Redis and no Celery.
"""

import math  # noqa: F401  (kept for parity with sibling tests)

from datetime import date

from chronos.config import Settings
from chronos.pipeline import Pipeline
from chronos.retrieval.qa import AnswerEngine
from chronos.retrieval.retriever import TemporalRetriever

PDF_TEXT = (
    "ChronosAudit Regulatory Compliance Manual\n\n"
    "Section 1. Cross-Border Data Transfer Policy\n"
    "The Cross-Border Data Transfer Policy prohibits the transfer of Personal "
    "Data outside the EU effective 2022-03-15. From 2024-07-01 the Data "
    "Transfer Policy permits cross-border data transfers when the recipient "
    "implements Standard Contractual Clauses approved by the Data Protection "
    "Authority.\n\n"
    "Section 2. Tax Code Amendments\n"
    "The Corporate Tax Code imposes a 25 percent standard corporate income "
    "tax rate from 2023-01-01. The Tax Reform Act 2024 reduces the rate to "
    "21 percent effective 2025-01-01 for Large Multinational Groups."
)


def _make_pdf(path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), PDF_TEXT, fontsize=8)  # long text may overflow; fine for test
    for para in PDF_TEXT.split("\n\n"):
        page.insert_textbox(  # pragma: no cover - best-effort layout
            fitz.Rect(72, 72, 540, 720), para, fontsize=10
        )
    doc.save(path)
    doc.close()


def test_pipeline_end_to_end(tmp_path):
    settings = Settings(
        model_provider="none",
        embedding_provider="hashing",
        embedding_dim=256,
        lance_db_path=str(tmp_path / "lancedb"),
        graph_store_path=str(tmp_path / "graph.gpickle"),
        doc_state_dir=str(tmp_path / "state"),
        chunk_size_tokens=120,
        chunk_overlap_tokens=20,
    )
    pdf = tmp_path / "manual.pdf"
    _make_pdf(str(pdf))

    pipe = Pipeline(settings)
    record = pipe.run_ingestion("doc-manual", "manual.pdf", str(pdf))

    assert record.status.value == "indexed", record.error
    assert record.chunk_count > 0
    assert record.entity_count > 0

    # Query with the retriever + extractive QA engine
    retriever = TemporalRetriever(pipe.lance, pipe.graph)
    ctx = retriever.retrieve("What is the cross-border data transfer policy?", "Q4 2024", top_k=5)
    assert ctx.hits, "must have vector hits"
    assert ctx.window_start == date(2024, 10, 1)
    assert ctx.window_end == date(2024, 12, 31)

    engine = AnswerEngine(settings, llm=None)
    result = engine.answer("What is the data transfer policy?", ctx)
    assert result.answer
    assert result.context == ctx


def test_pipeline_failed_pdf(tmp_path):
    settings = Settings(
        model_provider="none",
        embedding_provider="hashing",
        embedding_dim=64,
        lance_db_path=str(tmp_path / "lancedb"),
        graph_store_path=str(tmp_path / "graph.gpickle"),
        doc_state_dir=str(tmp_path / "state"),
    )
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 fake")  # not a real PDF

    pipe = Pipeline(settings)
    record = pipe.run_ingestion("doc-bad", "bad.pdf", str(bad))
    assert record.status.value == "failed"
    assert record.error