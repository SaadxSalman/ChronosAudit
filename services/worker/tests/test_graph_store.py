"""Unit tests for the NetworkX temporal knowledge graph."""

from datetime import date

from chronos.models import EntityOccurrence, ExtractionResult, RelationOccurrence, TemporalValidity
from chronos.storage.graph_store import TemporalGraph
from chronos.temporal.interval import Interval


def _extraction(chunk_id: str) -> ExtractionResult:
    return ExtractionResult(
        chunk_id=chunk_id,
        entities=[
            EntityOccurrence(id=f"{chunk_id}:e0", name="Data Transfer Policy", type="Regulation"),
            EntityOccurrence(id=f"{chunk_id}:e1", name="EU", type="Jurisdiction"),
            EntityOccurrence(id=f"{chunk_id}:e2", name="Data Processor", type="Party"),
        ],
        relations=[
            RelationOccurrence(
                subject="Data Transfer Policy",
                predicate="governs",
                object="Data Processor",
                validity=TemporalValidity(
                    start=date(2024, 1, 1), end=date(2025, 12, 31),
                    start_label="1 Jan 2024", end_label="31 Dec 2025",
                ),
                sentence="Data Transfer Policy governs Data Processor from 1 Jan 2024 to 31 Dec 2025.",
            ),
            RelationOccurrence(
                subject="EU",
                predicate="amends",
                object="Data Transfer Policy",
                validity=TemporalValidity(start=date(2025, 3, 1), end=None),
                sentence="The EU amends the Data Transfer Policy effective March 2025.",
            ),
        ],
    )


def test_upsert_and_snapshot(tmp_path, monkeypatch):
    graph = TemporalGraph()
    monkeypatch.setitem(graph.settings.__dict__, "graph_store_path", str(tmp_path / "g.gpickle"))

    graph.upsert_document("doc-a", "a.pdf", [_extraction("c1")])

    assert graph.graph.number_of_nodes() == 3
    assert graph.graph.number_of_edges() == 2

    # as-of the middle of the first relation's validity
    snap = graph.snapshot(valid_at=date(2024, 6, 1))
    assert any(e.subject == "Data Transfer Policy" and e.predicate == "governs" for e in snap.edges)
    assert not any(e.predicate == "amends" for e in snap.edges)  # not valid until 2025

    # during window after the amendment
    snap2 = graph.snapshot(window=Interval(date(2025, 6, 1), date(2025, 12, 31)))
    assert any(e.predicate == "amends" for e in snap2.edges)
    assert any(e.predicate == "governs" for e in snap2.edges)


def test_evolution_and_persistence(tmp_path, monkeypatch):
    import os

    graph = TemporalGraph()
    monkeypatch.setitem(graph.settings.__dict__, "graph_store_path", str(tmp_path / "g.gpickle"))
    graph.upsert_document("doc-a", "a.pdf", [_extraction("c1")])

    history = graph.evolution("Data Transfer Policy")
    assert len(history) == 2

    graph.save()
    assert (tmp_path / "g.gpickle").exists()

    graph2 = TemporalGraph()
    monkeypatch.setitem(graph2.settings.__dict__, "graph_store_path", str(tmp_path / "g.gpickle"))
    graph2.load()
    assert graph2.graph.number_of_edges() == 2


def test_remove_document(tmp_path, monkeypatch):
    graph = TemporalGraph()
    monkeypatch.setitem(graph.settings.__dict__, "graph_store_path", str(tmp_path / "g.gpickle"))
    graph.upsert_document("doc-a", "a.pdf", [_extraction("c1")])
    graph.remove_document("doc-a")
    assert graph.graph.number_of_edges() == 0
    assert graph.graph.number_of_nodes() == 0


def test_stats(tmp_path, monkeypatch):
    graph = TemporalGraph()
    monkeypatch.setitem(graph.settings.__dict__, "graph_store_path", str(tmp_path / "g.gpickle"))
    graph.upsert_document("doc-a", "a.pdf", [_extraction("c1")])
    stats = graph.stats()
    assert stats.node_count == 3
    assert stats.edge_count == 2
    assert stats.temporally_validated_edges > 0
    assert "Regulation" in stats.entity_types