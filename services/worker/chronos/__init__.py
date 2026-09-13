"""ChronosAudit worker package.

Temporal Knowledge-Graph RAG for live financial and regulatory auditing.

The worker owns the heavy lifting:

* PDF parsing + smart chunking,
* temporal-aware entity/relationship extraction via a configurable SLM
  (OpenAI, local Ollama, or a deterministic zero-config fallback),
* embedding + vector search through LanceDB,
* a NetworkX temporal knowledge graph with interval-validated edges,
* hybrid retrieval (date-windowed vector search + time-sliced graph traversal)
  and answer synthesis for audit questions.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]