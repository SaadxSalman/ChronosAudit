"""Unit tests for chunking and the deterministic fallback extractor."""

from chronos.extraction.chunker import Page, chunk_document, estimate_tokens
from chronos.extraction.fallback import extract_fallback, find_dates_in_text


def _mk_pages(paragraphs: list[str]) -> list[Page]:
    return [Page(number=i + 1, text=p) for i, p in enumerate(paragraphs)]


def test_chunker_splits_and_preserves_pages():
    pages = [
        Page(1, "Section A.\n\nThe first policy statement covers cross-border flows."),
        Page(2, "Section B. The second statement sets limits for Q3 2024."),
    ]
    chunks = chunk_document(pages, "doc-1", chunk_size_tokens=20, overlap_tokens=5)
    assert len(chunks) >= 2
    assert all(c.doc_id == "doc-1" for c in chunks)
    assert all(c.page_start >= 1 and c.page_end >= c.page_start for c in chunks)
    assert "Section A" in chunks[0].text


def test_chunker_respects_budget():
    big = "Regulation 10/23 defines new thresholds for cross-border data transfers. " * 60
    pages = [Page(1, big)]
    chunks = chunk_document(pages, "d", chunk_size_tokens=40, overlap_tokens=8)
    assert len(chunks) > 1
    # overlap: chunks share some token content
    tok_a = set(chunks[0].text.split())
    tok_b = set(chunks[1].text.split())
    assert len(tok_a & tok_b) > 0


def test_fallback_extracts_entities_and_relations():
    text = (
        "The Cross-Border Data Regulation requires data localization for "
        "Personal Data on 2024-03-01. The EU Data Act amends the Data Governance Act "
        "effective 2024-01-01 until 2026-12-31."
    )
    res = extract_fallback(text, chunk_id="c0", doc_name="test")

    # A chunk-level window must be inferred from the dates in the text
    assert res.chunk_temporal.start is not None
    assert res.chunk_temporal.end is not None

    dates = find_dates_in_text(text)
    assert dates, "must find at least one date"

    # Entities and at least one relation should surface
    assert res.entities, "fallback should extract capitalized entities"
    assert any(r.subject for r in res.relations), "fallback should find verb frames"


def test_fallback_dates():
    text = "Q3 2024 saw a change; effective January 2024 through March 2024."
    dates = sorted(d for d, _ in find_dates_in_text(text))
    assert len(dates) >= 2
    assert dates[0] < dates[-1]