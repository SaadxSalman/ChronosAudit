# ChronosAudit — Temporal Knowledge-Graph RAG

> **Temporal Knowledge-Graph RAG for live financial and regulatory auditing.**
> ChronosAudit ingests PDFs whose *facts change over time* — tax codes, compliance policies, regulator circulars — extracts entities, relationships and **explicit temporal validity intervals** (`valid_from` → `valid_until`), stores them in a vector index (LanceDB) and a temporal knowledge graph (NetworkX), and answers audit questions **as of any point in time**, or **compares two points in time**, with interval-validated citations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  "What was the compliance protocol for cross-border data transfers          │
│   in Q3 2024 versus current regulations?"                                   │
└─────────────────────────────────────────────────────────────────────────────┘
        │                                     │
        ▼                                     ▼
┌──────────────────────┐              ┌──────────────────────┐
│  Q3 2024 window      │              │  Q4 2024 window      │
│  ─ vector lens:      │              │  ─ vector lens:      │
│    chunks overlapping│              │    chunks overlapping│
│    [2024-07-01,      │              │    [2024-10-01,      │
│     2024-09-30]      │              │     2024-12-31]      │
│  ─ graph lens:       │              │  ─ graph lens:       │
│    edges whose       │              │    edges whose       │
│    validity overlaps │              │    validity overlaps │
└──────────┬───────────┘              └──────────┬───────────┘
           │  Data Transfer Policy               │  Data Transfer Policy
           │  --[requires]--> localization       │  --[permits]--> cross-border
           │  (2022-01-01 → 2024-06-30)          │  transfers w/ SCCs
           │                                     │  (2024-07-01 → ∞)
           ▼                                     ▼
     "Localize personal data."            "Transfer permitted with SCCs."
```

**Two different answers. Same question. Different points in time.** That is the whole point.

---

## Table of contents

1. [Why naive RAG fails for regulation](#1-why-naive-rag-fails-for-regulation)
2. [Core temporal model](#2-core-temporal-model)
3. [Architecture](#3-architecture)
4. [Repository layout](#4-repository-layout)
5. [The ingestion pipeline](#5-the-ingestion-pipeline)
6. [The retrieval pipeline](#6-the-retrieval-pipeline)
7. [Answer synthesis & window comparison](#7-answer-synthesis--window-comparison)
8. [API reference](#8-api-reference)
9. [Configuration reference](#9-configuration-reference)
10. [Quickstart (native)](#10-quickstart-native)
11. [Quickstart (Docker)](#11-quickstart-docker)
12. [Demo walkthrough (`seed_demo.py`)](#12-demo-walkthrough-seed_demopy)
13. [Web dashboard](#13-web-dashboard)
14. [On-disk storage layout](#14-on-disk-storage-layout)
15. [Testing](#15-testing)
16. [Design decisions & trade-offs](#16-design-decisions--trade-offs)
17. [Performance notes](#17-performance-notes)
18. [Limitations (read this before trusting it)](#18-limitations-read-this-before-trusting-it)
19. [Roadmap](#19-roadmap)
20. [Troubleshooting](#20-troubleshooting)
21. [FAQ](#21-faq)
22. [License](#22-license)

Appendices: [A — Command reference](#appendix-a--command-reference) · [B — Environment variable index](#appendix-b--environment-variable-index) · [C — The one-paragraph pitch](#appendix-c--the-one-paragraph-pitch)

---

## 1. Why naive RAG fails for regulation

Conventional RAG treats a corpus as a bag of timeless text. That breaks the moment the corpus describes rules that were **amended, replaced, or repealed**:

- A 2024 chunk saying *"the standard rate is 25%"* and a 2025 chunk saying *"the standard rate is 15%"* are both "relevant" to *"what is the standard rate?"* — the vector store returns both, and the LLM either picks one at random or hedges into uselessness.
- A chunk that contains **both** the old rule and the sentence *"repealed effective 2024-07-01"* poisons retrieval for "current" queries even though its *second half* is exactly the current rule.
- Facts that never appear verbatim in any chunk — *"Regulation X supersedes Regulation Y"*, *"Entity A is exempt until 2025-12-31"* — are exactly the facts an auditor needs, and pure vector search has no way to traverse to them.

ChronosAudit fixes this by making **time a first-class dimension of the index**, not a post-filter:

| Naive RAG | ChronosAudit |
|---|---|
| Chunks have no validity | Every chunk carries `[start, end]` inferred from the dates it mentions |
| Similarity is the only signal | Similarity **and** interval overlap, pushed *down into LanceDB* |
| Facts are locked inside chunk text | Facts are lifted into a **MultiDiGraph** whose edges each carry their own validity interval |
| The repealer sentence pollutes retrieval | A *start-frame-aware* chunk-window rule keeps "… starting 2024-07-01" chunks retrievable in later windows |
| One answer, no provenance of *when* | Answers cite `[v0]` chunks and `[g1]` graph edges, each stamped with its validity |
| Cannot compare regimes | `POST /api/query/compare` diffs the two windows' fact sets: added / removed / validity-changed / unchanged |

---

## 2. Core temporal model

ChronosAudit implements a pragmatic subset of bitemporal versioning.

### 2.1 Valid time

Every chunk, relation and graph edge carries a `TemporalValidity`:

```python
class TemporalValidity(BaseModel):
    start: Optional[date]       # valid_from (inclusive)
    end: Optional[date]         # valid_until (inclusive); None ⇒ open-ended
    start_label: Optional[str]  # verbatim source text, e.g. "1 January 2023"
    end_label: Optional[str]    # verbatim source text, e.g. "31 December 2025"
```

- **`None` endpoints mean "unknown", not "excluded".** An edge with `end=None` is treated as *open-ended* — still valid at any later date. That matches how regulations are written: they remain in force until something repeals them.
- **Labels are preserved** so every answer can quote the *exact phrase* the document used ("Supersedes the version of 31 December 2023") rather than a lossy ISO date.

### 2.2 Query windows

A *window* is the interval the auditor asks about, parsed from natural language by `chronos/temporal/parser.py`:

| Input | Resolved interval |
|---|---|
| `2024` | `[2024-01-01, 2024-12-31]` |
| `Q3 2024` / `q3 2024` | `[2024-07-01, 2024-09-30]` |
| `2024-07` | `[2024-07-01, 2024-07-31]` |
| `2024-07-15` | as-of that single day |
| `2024-01-01..2024-06-30` (also `to`, `→`, `through`) | explicit range |
| `as of 2024-07-15` | as-of that single day |
| *(absent — e.g. "current")* | `[today, today]` — "now" |

A chunk or edge is **valid in window `[W1, W2]`** iff:

```
(start IS NULL OR start <= W2)  AND  (end IS NULL OR end >= W1)
```

Open-ended things are valid in every window that starts after they begin. Fully undated things (`start=None, end=None`) are valid everywhere — a pragmatic default for facts that genuinely have no stated duration.

### 2.3 Chunk-level vs fact-level intervals

ChronosAudit maintains **two layers** of temporal metadata, deliberately:

1. **Chunk window** (`chunk_temporal`) — the *hull* of all dates mentioned in the chunk. It decides **retrieval eligibility**: does this chunk belong in this window's candidate set?
2. **Fact intervals** (per relation, `RelationOccurrence.validity`) — attached to each extracted subject–predicate–object triple and persisted as edge attributes. It decides **graph visibility**: may this edge participate in this window's answer?

Splitting the two is what makes "the chunk that announces the repeal" behave correctly: the chunk is a *container* whose hull spans both regimes, while each *fact inside it* has its own tight interval.

### 2.4 The start-frame rule (a subtle, load-bearing detail)

Consider a chunk that reads:

> *The Data Transfer Policy requires data localization … until 2024-06-30. The Data Transfer Policy permits cross-border transfers with SCCs starting 2024-07-01.*

The naive hull is `[2022-01-01, 2024-07-01]`. But the **latest** date sits in a *start frame* ("starting …") — it *opens* a regime, it does not *close* the chunk. Under the naive hull this chunk would be excluded from every "current" query (window start `2024-10-01` > `2024-07-01`) — hiding precisely the **successor facts** the system exists to surface.

`infer_chunk_window()` in `chronos/extraction/fallback.py` therefore inspects the text *immediately before* the latest date: if the nearest temporal marker is a start frame (`starting | starts | from | effective | commencing | as of | begins…`) and no end marker follows (`until | through | expires | repealed | sunset…`), the chunk's `end` is left **open**. A companion rule uses the *nearest* preceding marker, so `effective X until Y` still closes at `Y` correctly.

### 2.5 Canonical entity identity

The same discipline is applied to entity identity: `canonical_name()` strips leading articles so that *"The Data Transfer Policy"* in a sentence frame and *"Data Transfer Policy"* in an entity list resolve to the **same graph node**. Without this, the same entity appears as two nodes and interval-validated edges become unreachable during traversal — a bug that manifests as nondeterminism (Python set-iteration order interacting with seed dedup) rather than a clean failure.

### 2.6 What this model does *not* do

- **No transaction time.** ChronosAudit records *valid time* only. It does not track "what did we *believe* on date D" (that would require versioning the index itself). Re-ingesting a corrected document overwrites the previous facts.
- **No probabilistic intervals.** Validity is binary per window; there is no fuzzy "mostly valid in 2023".
- **No timezone modeling.** All dates are `datetime.date` — appropriate for regulation, which is specified per calendar day.

---
## 3. Architecture

### 3.1 Services

```
                         ┌────────────────────────────────────────────┐
                         │                 Browser                    │
                         │        web/index.html (zero-build UI)      │
                         └───────────────┬────────────────────────────┘
                                         │  /ui (static) + /api/* (JSON)
                         ┌───────────────▼────────────────────────────┐
                         │        Express gateway      (Node 20/TS)   │
                         │  services/api                              │
                         │  · multipart upload → disk                 │
                         │  · document state machine + JSON store     │
                         │  · static dashboard hosting                │
                         │  · JSONProxy → worker (retry, timeouts)    │
                         └───────────────┬────────────────────────────┘
                                         │  HTTP (localhost:8100)
                         ┌───────────────▼────────────────────────────┐
                         │        Worker bridge      (Python/FastAPI) │
                         │  services/worker/chronos/api.py            │
                         │  · /documents/ingest → Celery task or sync │
                         │  · /query, /query/compare                  │
                         │  · /graph/* (stats/snapshot/neighbors/diff)│
                         └───────┬───────────────────────┬────────────┘
                                 │ task queue            │ direct imports
                   ┌─────────────▼───────────┐   ┌───────▼────────────────────┐
                   │  Celery worker (Redis)  │   │  In-process pipeline       │
                   │  chronos/tasks/ingest.py│   │  (WORKER_MODE=sync)        │
                   └─────────────┬───────────┘   └───────┬────────────────────┘
                                 │      same code path   │
                         ┌───────▼───────────────────────▼────────────┐
                         │              Ingestion pipeline            │
                         │  PDF parse → chunk → SLM/fallback extract  │
                         │  → embed → LanceDB + NetworkX graph        │
                         └───────┬───────────────────────┬────────────┘
                                 │                       │
                     ┌───────────▼─────────┐   ┌─────────▼──────────────┐
                     │  LanceDB (vectors + │   │  NetworkX MultiDiGraph │
                     │  chunk validity)    │   │  (edges carry validity)│
                     └─────────────────────┘   └────────────────────────┘
```

Two deployment modes share one code path:

- **`WORKER_MODE=celery`** — the bridge enqueues `tasks.ingest_document`; a Celery worker processes PDFs in a separate process (horizontal scaling, retries). Requires Redis.
- **`WORKER_MODE=sync`** (default) — the bridge calls the pipeline in-process. Zero external infrastructure, ideal for demo/dev/small corpora.
### 3.2 Modules (worker)

| Module | Responsibility |
|---|---|
| `chronos/config.py` | Pydantic `Settings` (env-driven), singleton `get_settings()` |
| `chronos/models.py` | Pydantic domain models: `TemporalValidity`, `EntityOccurrence`, `RelationOccurrence`, `ExtractionResult`, `ChunkHit`, `GraphEdge`, `RetrievalContext`, `DocumentRecord`… |
| `chronos/temporal/parser.py` | Natural-language window → `[start, end]` |
| `chronos/temporal/interval.py` | Interval algebra: `intervals_overlap`, `edge_valid_within`, snapshot diffing |
| `chronos/extraction/chunker.py` | PDF pages → token-budgeted overlapping chunks (tiktoken if available, whitespace estimator otherwise) |
| `chronos/extraction/slm.py` | Extraction SLM facade: OpenAI-compatible chat call with a strict JSON schema |
| `chronos/extraction/prompts.py` | The extraction prompt (entities / relations / **validity** / chunk window) |
| `chronos/extraction/fallback.py` | Deterministic regex extractor (no-LLM mode) + `infer_chunk_window` + `canonical_name` |
| `chronos/embeddings.py` | Embedding provider facade: OpenAI embeddings or deterministic hashing fallback |
| `chronos/storage/lancedb_store.py` | LanceDB table: chunk text + vector + validity + entities; **server-side** temporal filter |
| `chronos/storage/graph_store.py` | NetworkX MultiDiGraph; edges carry `valid_from`/`valid_until`; snapshots, neighbors, evolution, GraphML export |
| `chronos/retrieval/retriever.py` | The temporal retriever: filter → vector → seeds → graph lens → context |
| `chronos/retrieval/qa.py` | `AnswerEngine`: LLM synthesis, extractive fallback, window comparison |
| `chronos/pipeline.py` | Orchestrates ingestion end-to-end; per-document progress + status |
| `chronos/tasks/ingest.py` | Celery task wrapper |
| `chronos/api.py` | FastAPI bridge |

### 3.3 Modules (API gateway)

| Module | Responsibility |
|---|---|
| `src/config.ts` | Env-driven config (port, worker URL, upload dir, auth token) |
| `src/routes/health.ts` | `/api/health`, `/api/system/info` |
| `src/routes/documents.ts` | Upload (multipart), list, status, delete, reindex |
| `src/routes/queries.ts` | `/api/query`, `/api/query/compare` |
| `src/routes/graph.ts` | `/api/graph/stats\|snapshot\|neighbors\|evolution\|diff` |
| `src/services/workerClient.ts` | Bounded-timeout JSON proxy with one retry on 502/503/504 |
| `src/services/docStore.ts` | JSON-file document registry (status, progress, counts) |
| `src/index.ts` | Express bootstrap + static `/ui` hosting |

### 3.4 Data flow of one question

```
question + window label
   │
   ▼
parse window            "Q3 2024"  →  [2024-07-01, 2024-09-30]
   │
   ▼
vector lens             LanceDB .search() with .where(temporal predicate)
                        → top_k chunks whose validity overlaps the window
   │
   ▼
seed entities           union of entities attached to the hits
                        + question-token matches against graph nodes
   │
   ▼
graph lens              BFS (≤2 hops) over MultiDiGraph from seeds;
                        every candidate edge must pass edge_valid_within(window)
   │
   ▼
RetrievalContext        hits + graph_edges + seed_entities + trace[]
   │
   ▼
AnswerEngine            LLM synthesis (grounded, citation-forced)
                        or extractive fallback when no LLM configured
```

Every stage appends a human-readable line to `context.trace`, so the UI can show *why* an answer looks the way it does.
## 4. Repository layout

```
ChronosAudit/
├── README.md                        ← you are here
├── LICENSE                          MIT
├── Makefile                         dev shortcuts (api / worker / celery / test / demo)
├── package.json                     root scripts delegating into services
├── docker-compose.yml               full stack: redis + worker + api (+ dashboard on :4000)
├── .github/workflows/ci.yml         CI: pytest (py3.11–3.13) + TS build/test
│
├── scripts/
│   ├── make_sample_pdfs.py          generates two demo PDFs (reportlab), no assets needed
│   └── seed_demo.py                 end-to-end demo driver against a running API
│
├── web/
│   └── index.html                   zero-build audit dashboard (served by the API at /ui)
│
├── services/
│   ├── api/                         Express gateway (TypeScript, Node ≥20)
│   │   ├── src/
│   │   │   ├── index.ts             bootstrap
│   │   │   ├── app.ts               Express app + routes + static /ui
│   │   │   ├── config.ts            env config
│   │   │   ├── types.ts             shared API types + ApiError + asyncHandler
│   │   │   ├── routes/              health / documents / queries / graph
│   │   │   ├── services/            workerClient (HTTP proxy), docStore (JSON registry)
│   │   │   ├── utils/format.ts      small formatting helpers
│   │   │   └── test/format.test.ts  node:test unit tests
│   │   ├── Dockerfile  tsconfig.json  package.json  .env.example
│   │   └── data/                    runtime: uploads/ + meta/documents.json (gitignored)
│   │
│   └── worker/                      Python worker (Celery + FastAPI + LanceDB + NetworkX)
│       ├── chronos/
│       │   ├── config.py  models.py  celery_app.py  pipeline.py  api.py  embeddings.py
│       │   ├── temporal/            parser.py (window NL parsing), interval.py (algebra)
│       │   ├── extraction/          chunker.py, slm.py, prompts.py, fallback.py
│       │   ├── storage/             lancedb_store.py, graph_store.py
│       │   ├── retrieval/           retriever.py, qa.py
│       │   └── tasks/               ingest.py (Celery task)
│       ├── tests/                   pytest suite (temporal math → pipeline e2e)
│       ├── requirements.txt  pyproject.toml  Dockerfile  .env.example
│       └── data/                    runtime: lancedb/, graph/, state/ (gitignored)
```

Runtime directories are created on demand; nothing needs to be pre-created.

---
## 5. The ingestion pipeline

`chronos/pipeline.py::Pipeline.ingest_document()` drives five stages, updating a `DocumentRecord` (status + progress) after each so the dashboard can show live progress:

### Stage 1 — Parse (`PARSING`, progress 0.15)

`pypdf` extracts text page by page. A document with zero extractable text (e.g. a pure scan) fails into `DocumentStatus.FAILED` with the reason stored on the record.

### Stage 2 — Chunk (`CHUNKING`, progress 0.3)

`extraction/chunker.py` packs page text into chunks of ~`CHUNK_SIZE_TOKENS` tokens (default 400; tiktoken `cl100k_base` when installed, whitespace-token estimator otherwise) with `CHUNK_OVERLAP_TOKENS` (default 80) of token overlap. Page boundaries are tracked (`page_start`/`page_end`) so citations can point at a page. The splitter prefers paragraph and sentence breaks over mid-word cuts.

### Stage 3 — Extract (`EXTRACTING`, progress 0.5)

For each chunk, the **extraction SLM** (`extraction/slm.py`) is asked — via a strict JSON prompt (`extraction/prompts.py`) — to return:

```json
{
  "entities":  [{"name": "...", "type": "Regulation|Organization|Instrument|Jurisdiction|Party|Event|Other",
                 "aliases": ["..."]}],
  "relations": [{"subject": "...", "predicate": "requires|permits|prohibits|amends|supersedes|...",
                 "object": "...",
                 "validity": {"start": "YYYY-MM-DD|null", "end": "YYYY-MM-DD|null",
                              "start_label": "verbatim text", "end_label": "verbatim text"},
                 "sentence": "supporting sentence", "confidence": 0.85}],
  "chunk_temporal": {"start": "...|null", "end": "...|null"}
}
```

The point of using a *small* model is that extraction is a constrained, schema-shaped task — not open-ended generation. Any OpenAI-compatible endpoint works (`EXTRACTION_MODEL`, `OPENAI_BASE_URL`); temperature 0, JSON mode when supported, with a **repair pass** that strips markdown fences and retries once on parse failure.

**When no LLM is configured** (`MODEL_PROVIDER=none`, the default), `extraction/fallback.py` runs a deterministic extractor instead — same `ExtractionResult` schema, zero network:

- **Entities**: frequent capitalized noun phrases (1–4 words), stopword-filtered, top-24 per chunk; classified as Regulation / Organization / Instrument / Jurisdiction / Party by keyword.
- **Relations**: sentences containing a known regulatory verb (`requires, prohibits, permits, governs, amends, replaces, supersedes, imposes, applies, mandates, allows, defines, establishes, sets, limits, exempts…`) are framed as `subject --verb--> object`, with subject = nearest entity before the verb and object = the ≤4-word noun phrase after it.
- **Validity per relation**: dates in the sentence are assigned by marker proximity — `effective / from / on / starting` → `start`; `until / through / expires / repealed` → `end`; a lone date implies a point-in-time (`start = end`).
- **Chunk window**: `infer_chunk_window()` — hull of all chunk dates, with the start-frame rule of §2.4.

This makes the entire system runnable offline, and it is what the test suite exercises.

### Stage 4 — Embed (`EMBEDDING`, progress 0.65)

`embeddings.py` computes one vector per chunk:

- **OpenAI-compatible** (`EMBEDDING_PROVIDER=openai`): any `/v1/embeddings` endpoint, batched.
- **Hashing** (`hashing`, default): deterministic feature-hashing bag-of-words projection with sublinear tf weighting and L2 normalization. Not semantically deep, but stable, fast, dependency-free, and honest: the dimension is configurable (`EMBEDDING_DIM`, default 768).

The same provider must be used at query time — mixing providers across ingest/query would silently corrupt similarity, so provider+dim are recorded and validated against the table.

### Stage 5 — Index (`INDEXING`, progress 0.85 → `INDEXED`, 1.0)

- **LanceDB** (`storage/lancedb_store.py`): one `chunks` table. Each row: `doc_id, doc_name, chunk_id, page, text, start, end, start_label, end_label, entities[], vector[dim]`. Dates are stored as ISO strings because LanceDB's SQL `where` filter needs comparable columns. `add_chunks` first `delete`s existing rows for the `doc_id` (idempotent re-ingest), then appends.
- **Graph** (`storage/graph_store.py`): relations become edges of a `MultiDiGraph`. Node = canonical entity name (with `type`, `mentions`); edge data = `subject, predicate, object, valid_from, valid_until, start_label, end_label, doc_id, evidence (sentence), confidence`. `upsert_document` removes the previous document's edges first (re-ingest = replace), then merges the new ones. The graph persists to `GRAPH_STORE_PATH` via pickle plus a side-car `doc_meta.json` mapping documents to their edge keys for clean removal.

---
## 6. The retrieval pipeline

`chronos/retrieval/retriever.py::TemporalRetriever.retrieve()` — the heart of the system.

### Step 0 — Resolve the window

The window can arrive three ways, in priority order: explicit `window_start/window_end` params (the API parses the label first), a `window` label ("Q3 2024"), or — absent both — **embedded in the question itself** ("…in Q3 2024…" is found and parsed). When nothing resolves, the window stays empty and the trace records `no temporal filter (open window)`.

### Step 1 — Vector lens (temporal filter *inside* LanceDB)

```python
qvec = self.embedder.embed(question)
hits = self.lance.search(qvec, top_k=top_k,
                         window_start=window_start, window_end=window_end)
```

`LanceStore.search` builds the predicate

```sql
(end IS NULL OR end >= :W1) AND (start IS NULL OR start <= :W2)
```

and passes it to LanceDB's `.where(...)` — so expired chunks are **never even candidates**. This is the difference between a temporal index and a temporal *post-filter*: filtered vector search stays fast as the corpus grows.

**Broad-recall fallback.** Chunk windows are coarse hulls; a document whose coverage *ends just before* the window may still describe facts that were in force *inside* it. When the windowed search returns fewer than `max(4, top_k)` hits, the retriever merges in an *unwindowed* search (`top(8, top_k+4)`) to backfill candidates, and the trace records `+N broad-recall`. The graph lens (step 3) still interval-validates every fact those extra chunks contribute, so this widens recall **without** letting stale facts into the answer.

### Step 2 — Seed entities

Seeds come from two sources:

1. **Entities attached to the hits** — each LanceDB row carries the entity names extracted from that chunk. Near-duplicates are collapsed with a Jaccard token-overlap > 0.4 dedupe (`token_overlap`), so "The Data Transfer Policy" and "Data Transfer Policy" don't crowd out other seeds.
2. **Entities named in the question** — any graph node whose canonical token set almost fully appears in the question (`_question_matches_node`) is added even with zero vector hits. This anchors "as-of" audits where the top-K chunks are thin.

Capped at 12 seeds.

### Step 3 — Graph lens (time-sliced traversal)

Breadth-first walk over the MultiDiGraph, up to `hops=2` rounds:

```python
frontier = {_canon(s) for s in seed_entities}          # article/case-insensitive
for each edge (u, v, data):
    if _canon(u) not in frontier and _canon(v) not in frontier: skip
    if not edge_valid_within(data, window): skip        # ← the time slice
    emit edge; expand frontier with _canon(u), _canon(v)
```

Every emitted edge must independently pass `edge_valid_within` — the *chunk* got us here, but the **edge's own validity** decides whether the fact is visible in this window. Results are deduped on `(subject, predicate, object)` and capped at `max_graph_edges` (40 default).

### Step 4 — Context assembly

Everything lands in a `RetrievalContext`:

```python
RetrievalContext(
    question, window_start, window_end, window_label,
    hits: list[ChunkHit],          # text + page + validity + score
    graph_edges: list[GraphEdge],  # subject/predicate/object + validity + evidence
    seed_entities: list[str],
    trace: list[str],              # human-readable, shown in the UI
)
```

`compare(question, window_a, window_b)` simply runs `retrieve` twice and returns both contexts; `diff_between_windows` delegates to the graph store's snapshot diff.

---

## 7. Answer synthesis & window comparison

`chronos/retrieval/qa.py::AnswerEngine` turns a `RetrievalContext` into an auditor-grade answer.

### 7.1 With an LLM configured

The prompt (`_build_prompt`) interleaves the two evidence lenses with citation keys:

```
Question: What was the compliance protocol for cross-border data transfers in Q3 2024?
Requested time window: Q3 2024 (2024-07-01 → 2024-09-30)

## VECTOR EVIDENCE
[v0] (doc=cross_border_data_policy.pdf, page=1, validity=1 September 2024 → …)
     <chunk text, truncated to 1200 chars>
[v1] …

## FACT EDGES
[g0] Data Transfer Policy --[requires]--> data localization  (validity: 2022-01-01 → 2024-06-30, src=doc-…)
[g1] Data Transfer Policy --[permits]--> cross-border transfers  (validity: 2024-07-01 → …, src=doc-…)

## Instruction
Answer the question for the requested window using only the evidence above.
… Return JSON: {"answer": "...with [v0]/[g1] citations",
                "facts": ["..."], "confidence": 0-1, "unknowns": ["..."]}
```

The LLM must return **structured JSON** with inline `[vN]`/`[gN]` citations and an explicit `unknowns` list — an auditor cares as much about what the corpus *doesn't* say as what it does. The parse is defensive (JSON extraction, fence stripping) and falls back to the extractive engine on failure, so an LLM outage degrades quality, never availability.

### 7.2 Without an LLM (default, deterministic)

The **extractive engine** composes an answer directly from the evidence: window label, the interval-validated fact lines (`subject --[predicate]--> object`, each with its validity), the top chunk snippets, and any open-ended facts flagged as such. Nothing is paraphrased; every line is traceable to a chunk or an edge. It reads like an audit working paper — which is exactly the point.

### 7.3 Comparison (`POST /api/query/compare`)

Two contexts (window A, window B) are diffed **on fact identity** `(subject, predicate, object)`:

| Bucket | Meaning |
|---|---|
| `removed` | facts valid in A that are absent (or no longer valid) in B — the repealed rules |
| `added` | facts present only in B — the new regime |
| `changed` | same `(s,p,o)` but a *different validity interval* — e.g. an obligation retained but re-dated |
| `unchanged` | same fact, same interval — the durable obligations |

Plus a `narrative` text block:

```
Comparing "2023" vs "Q4 2024":
- Added: 2 fact(s)
- Removed: 1 fact(s)
  ✗ Document --[supersedes]--> version of 31 (was valid 2023-01-01 → 2024-09-01)
  ✓ Global --[imposes]--> 15 percent corporate (valid 2024-01-01 → None)
  ✓ Anti-avoidance --[limits]--> interest deduction to 30 (valid 2023-01-01 → 2025-01-01)
```

Because both windows were already interval-validated during retrieval, the diff is a diff of *what was actually in force*, not a diff of document text.

---

## 8. API reference

Base URL: `http://localhost:4000` (Express BFF — the API's default port). It proxies to the FastAPI worker bridge on `:8100`.

### 8.1 Health & system

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Aggregate health: `{ok, service, version, worker_upstream, worker:{...}}` |
| `GET` | `/api/system/info` | Runtime info: ports, upstream worker settings |

The worker's health payload includes `mode` (`celery`/`sync`), `model_provider`, `embedding_provider`, `embedding_dim`, `lancedb_chunks`, `graph_nodes`, `graph_edges`, `documents`, `broker`.

### 8.2 Documents

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/documents` | List ingested documents with status/progress/counters |
| `POST` | `/api/documents/upload` | `multipart/form-data` (`file`, optional `wait`=`true\|false`, `timeout_seconds`) |
| `GET` | `/api/documents/:id/status` | Live status for one document |
| `POST` | `/api/documents/:id/reindex` | Re-run the pipeline on an already-stored file |
| `DELETE` | `/api/documents/:id` | Purge chunks, graph subgraph and the stored file |

### 8.3 Temporal queries

| Method | Path | Body / query | Returns |
|---|---|---|---|
| `POST` | `/api/query` | `{question, window?, top_k?}` | `AnswerResponse` |
| `POST` | `/api/query/compare` | `{question, window_a, window_b, top_k?}` | `CompareResponse` |

`AnswerResponse` (abridged):

```jsonc
{
  "question": "…",
  "answer": "… with [v0] and [g1] citations …",
  "citations": [
    {"key": "v0", "kind": "chunk", "doc_name": "policy.pdf", "page": 3,
     "validity": "1 January 2023 → …", "score": 0.42, "snippet": "…"},
    {"key": "g0", "kind": "graph_fact", "fact": "X → requires → Y",
     "validity": "2022-01-01 → 2024-06-30", "evidence": "verbatim sentence"}
  ],
  "facts": [ { "subject": "…", "predicate": "…", "object": "…",
               "validity": "2022-01-01 → 2024-06-30", "start": "2022-01-01",
               "end": "2024-06-30", "doc_id": "…", "chunk_id": "…",
               "evidence": "verbatim sentence" } ],
  "context": { "window_label": "Q3 2024", "window_start": "2024-07-01",
               "window_end": "2024-09-30", "hits": [...], "graph_edges": [...],
               "seed_entities": [...], "trace": ["temporal filter applied: …", ...] },
  "engine": "llm:openai | extractive",
  "confidence": 0.0
}
```

### 8.4 Graph exploration

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/graph/stats` | Node/edge/entity counts, predicate histogram, temporal range |
| `GET` | `/api/graph/snapshot?start=&end=&max_edges=` | All edges valid in a window |
| `GET` | `/api/graph/neighbors?entity=&hop=&valid_at=` | Edges touching an entity |
| `GET` | `/api/graph/evolution?entity=&predicate=` | How one (entity,predicate) changed over time |
| `POST` | `/api/graph/diff` | `{window_a, window_b}` → added/removed/changed facts |

---

## 9. Configuration reference

Environment variables only (`.env` per service; see `.env.example` files). The worker resolves `Settings` via pydantic `BaseSettings`; the API via a tiny typed loader.

### Worker (`services/worker/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `WORKER_MODE` | `sync` | `sync` (execute inline — dev/demo) or `async` (enqueue via Celery+Redis) |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker (only used in `async` mode) |
| `LANCE_DB_PATH` | `./data/lancedb` | Vector store directory |
| `GRAPH_STORE_PATH` | `./data/graph/chronos.gpickle` | Pickled NetworkX MultiDiGraph |
| `GRAPH_EXPORT_PATH` | `./data/graph/exports` | GraphML/JSON export directory |
| `DOC_STATE_DIR` | `./data/state` | Per-document JSON state files |
| `UPLOAD_DIR` | `./data/uploads` | Stored PDFs |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` (offline, deterministic) or `openai` |
| `EMBEDDING_DIM` | `768` | Vector width |
| `MODEL_PROVIDER` | `none` | `none` (deterministic extraction) / `openai` / `ollama` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | – | for `openai` / OpenAI-compatible endpoints |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | – | for `ollama` |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `800` / `120` | Chunking budget (tiktoken if available, whitespace fallback) |
| `WORKER_API_TOKEN` | – | Optional bearer token the bridge accepts |

### API (`services/api/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `4000` | HTTP port (TypeScript/Express BFF) |
| `WORKER_BASE_URL` | `http://localhost:8100` | Upstream FastAPI bridge |
| `WORKER_API_TOKEN` | – | Bearer token forwarded upstream |
| `DATA_DIR` | `./data` | Where uploaded files and doc state live |
| `WEB_DIR` | `../../web` | Static dashboard directory |

---

## 10. Quickstart (native)

### 10.1 Prerequisites

- **Python 3.11+** (3.13 tested) — worker
- **Node.js 20+** — API
- **Redis** (optional) — only needed for `WORKER_MODE=celery`; the default demo path uses `sync` mode and needs nothing external

### 10.2 Clone, create environments, configure

```bash
git clone https://github.com/SaadxSalman/ChronosAudit.git
cd ChronosAudit

# ---- Python worker -------------------------------------------------------
py -3 -m venv .venv                        # or: python3 -m venv .venv
.venv\Scripts\python -m pip install -r services/worker/requirements.txt   # *nix: .venv/bin/python
copy services\worker\.env.example services\worker\.env                    # cp on *nix

# ---- TypeScript API ------------------------------------------------------
cd services/api
npm install
copy .env.example .env                                                     # cp on *nix
cd ../..
```

> The defaults are deliberately **zero-dependency**: `MODEL_PROVIDER=none` (deterministic regex/heuristic extraction) and `EMBEDDING_PROVIDER=hashing` (stable feature-hashing embeddings). The full stack runs offline with no API keys.

### 10.3 Run the three processes

Terminal 1 — worker bridge (FastAPI on :8100):

```bash
cd services/worker
$env:WORKER_MODE="sync"                 # *nix: export WORKER_MODE=sync
..\..\.venv\Scripts\python -m uvicorn chronos.api:app --host 127.0.0.1 --port 8100
```

Terminal 2 — API (Express on :4000):

```bash
cd services/api
npm run build
node dist/index.js
```

Terminal 3 (optional) — dashboard is already served by the API at **http://localhost:4000/ui/**.

Terminal 4 (only for `WORKER_MODE=async`): start Redis and the Celery worker:

```bash
docker run -p 6379:6379 redis:7           # or: make dev-redis
cd services/worker
..\..\.venv\Scripts\python -m celery -A chronos.celery_app worker --loglevel=INFO --pool=solo
```

### 10.4 Smoke test

```bash
curl http://localhost:4000/api/health
# → {"ok":true,"service":"chronos-audit-api","worker_upstream":true, ...}
```

Then open `http://localhost:4000/ui/`, upload a PDF, and ask a question with a time window.

---

## 11. Quickstart (Docker)

```bash
docker compose up --build
# api    → http://localhost:4000  (dashboard at /ui/)
# worker → http://localhost:8100  (FastAPI bridge)
# redis  → localhost:6379         (broker for async mode)
```

`docker-compose.yml` wires the three services together, mounts named volumes for `data/` (LanceDB, graph, uploads, state) so your index survives rebuilds, and passes the relevant environment variables through. To switch the worker to queue-backed ingestion set `WORKER_MODE=async` in `services/worker/.env` (compose already points the broker at the bundled Redis).

To load the demo corpus into the running stack:

```bash
.venv\Scripts\python scripts/make_sample_pdfs.py --out-dir sample_docs
.venv\Scripts\python scripts/seed_demo.py --api-url http://localhost:4000 --sample-dir sample_docs
```

---

## 12. Demo walkthrough (`seed_demo.py`)

`scripts/seed_demo.py` is the one-command guided tour. It:

1. generates two sample regulatory PDFs on the fly (via `scripts/make_sample_pdfs.py` — a cross-border data policy and a corporate tax code with dated amendments), unless `--sample-dir` already has PDFs;
2. uploads them through the live API and **waits** for full ingestion;
3. prints graph stats;
4. asks four temporal questions, each with a different window style;
5. runs two **window comparisons** and prints added/removed/changed facts.

Full expected output (abridged — from an actual verified run):

```
→ Seeding against http://127.0.0.1:4100        (transcript recorded with the API on :4100)
  system health: chronos-audit-api | model=None
→ Uploading cross_border_data_policy.pdf …
  ✓ doc-13474852cc71 status=indexed chunks=1 entities=23 relations=3
→ Uploading tax_code_amendments_2023_2025.pdf …
  ✓ doc-19f22ee4e9bf status=indexed chunks=1 entities=24 relations=6
→ Graph now has 30 nodes, 7 edges

Q: What was the compliance protocol for cross-border data transfers in Q3 2024?
A: Evidence for "Q3 2024" (extractive engine): [v0] …cross_border_data_policy.pdf, valid 2022 → Q3 2024…
Q: What is the standard corporate income tax rate for 2025?
A: Evidence for "2025" … [v0] tax_code_amendments_2023_2025.pdf, valid 2023-01-01 → 2025-12-31 …

⇋ How did the cross-border data transfer compliance obligations change between 2023 and the current regime?
Comparing "2023" vs "Q4 2024":
   - Added: 2 fact(s)   - Removed: 1 fact(s)   - Changed: 0
     ✗ Document --[supersedes]--> version of 31 (was valid 2023-01-01 → 2024-09-01)
     ✓ Global --[imposes]--> 15 percent corporate (valid 2024-01-01 → None)

⇋ What changed in the corporate tax code between 2023 and 2025?
Comparing "2023" vs "2025":
   - Added: 2   - Removed: 2
     ✗ Corporate Tax Code --[imposes]--> 25 percent standard (was valid 2023-01-01 → 2023-01-01)
     ✓ Corporate Finance Act --[amends]--> Tax Reform Act (valid 2025-01-01 → 2025-01-01)
     ✓ Global --[imposes]--> 15 percent corporate (valid 2024-01-01 → None)
```

Read the last comparison carefully: the 25% standard rate fact is **removed** in 2025 and the 15% minimum-tax fact **added** in 2024 — the sample documents encode a genuine regime change, and the diff surfaces it as structured, interval-validated facts rather than vibes.

Usage:

```
scripts/seed_demo.py [--api-url URL] [--sample-dir DIR] [--skip-queries]
scripts/make_sample_pdfs.py [--out-dir DIR]
```

---

## 13. Web dashboard

`web/index.html` is a dependency-free single-file console served by the Express API at `/ui/` (no build step — it is plain HTML/CSS/JS). It is organized in three tabs plus a document panel:

- **Documents panel** — upload (`POST /api/documents/upload` with *wait until indexed*), live status badges (`queued → parsing → chunking → extracting → embedding → indexing → indexed` / `failed`), progress bars, per-document counters (pages/chunks/entities/relations/size), and *reindex* / *delete* actions.
- **Ask tab** — question + natural-language window + `top_k`. Renders the answer, the interval-validated fact list, vector evidence with validity spans, the retrieval trace (temporal filter → vector hits → seed entities → graph lens) and seed-entity pills.
- **Compare tab** — two window labels, color-coded verdict: **removed** (red), **added** (green), **validity-changed** (amber), **unchanged**.
- **Graph tab** — live stats tiles (nodes/edges/entities/relations), a window-filtered edge snapshot, and an entity-neighborhood explorer.

A health strip in the header polls `/api/health` every 10 s and surfaces worker mode, embedding provider, and current index sizes.

---

## 14. On-disk storage layout

```
services/worker/data/
├── lancedb/                     # LanceDB dataset directory
│   └── chunks.lance/            # one table: chunk rows + vectors + temporal cols
├── graph/
│   ├── chronos.gpickle          # pickled NetworkX MultiDiGraph (atomic tmp+rename writes)
│   └── exports/
│       ├── chronos.graphml      # full-graph GraphML export (Gephi-ready)
│       └── chronos.json         # JSON node/edge export
├── state/                       # one JSON file per document (DocumentRecord)
│   └── doc-<hash>.json
└── uploads/                     # the stored PDFs, named by doc_id
```

Vector table schema (one row per chunk):

| column | type | notes |
|---|---|---|
| `chunk_id` | string | `doc:c#####` |
| `doc_id`, `doc_name`, `page` | provenance fields | |
| `text` | string | the chunk body |
| `start`, `end` | string (`YYYY-MM-DD` or `""`) | chunk-level validity |
| `start_label`, `end_label` | string | verbatim date phrases |
| `entities` | list\<string\> | seed-entity material |
| `tokens` | int | chunk size |
| `vector` | list\<float32\>[dim] | embedding |

Graph node/edge attributes: nodes carry `name`, `type`, `mentions`; edges carry `predicate`, `valid_from`/`valid_until` (ISO), `start_label`/`end_label`, `doc_id`, `chunk_id`, `evidence` (verbatim sentence, truncated to 240 chars).

**Crash-safety:** both stores are append-mostly and rewritten atomically — the graph pickler writes to a temp file and `os.replace()`s it; LanceDB commits row-groups transactionally. A killed process leaves at most a partially ingested *document*, never a corrupt store.

---

## 15. Testing

```bash
# Python worker (interval algebra, parser, chunker/fallback, graph store,
# retriever, pipeline e2e — no network, no models)
cd services/worker
../../.venv/Scripts/python -m pytest tests -v        # Windows venv
python -m pytest tests -v                            # POSIX

# TypeScript API (format utils + route smoke tests)
cd services/api
npm test

# Full-stack smoke: boots worker bridge + API on ephemeral ports,
# uploads a generated PDF, asks windowed questions, asserts on facts.
make e2e                                             # → scripts/e2e_smoke.py
```

The pipeline e2e test (`tests/test_pipeline_fallback.py`) ingests a **real generated PDF** through the actual chunker → fallback extraction → embeddings → LanceDB + graph, then retrieves with a window and asserts interval-validated facts come back. Every answer ships a `context.trace` — the exact filter → search → seeds → traversal steps — which doubles as a production debugging tool (the dashboard renders it under "Retrieval trace").

---

## 16. Design decisions & trade-offs

**Why NetworkX + pickle instead of a graph database?** Zero-dependency determinism. The graph is small (thousands of edges, not millions), fully loaded per process, and the pickled file is rewritten atomically (temp file + `os.replace`). Every query is pure in-memory traversal — microseconds. A native graph DB (Neo4j/Memgraph) would add ops burden without changing query semantics: the temporal predicate is just a filter on edge attributes, implemented identically by `edge_valid_within()`. When scale demands it, `TemporalGraph` is the only seam to reimplement.

**Why LanceDB instead of pgvector/Qdrant?** Embedded, columnar, zero-ops, with `where=` SQL-style predicate pushdown — the temporal filter is applied *inside* the vector search, not as a post-filter, so `top_k` is filled with eligible rows only. It also supports starting with **no embedding model at all** (hashing fallback), so the whole system runs deterministically offline.

**Why a fallback extractor at all?** Reproducibility and a hard floor. SLM extraction is probabilistic; audits require a baseline you can diff across runs. The fallback is deterministic regex/lexicon extraction with honest `confidence=0.5`, used (a) when no model is configured, (b) as per-chunk degradation when the SLM fails or returns garbage. The deterministic path is also what CI exercises — the test suite never calls a model.

**Why two interval layers (chunk hull vs fact interval)?** A chunk is a *container*; a fact is a *claim*. The hull decides retrieval eligibility cheaply at the vector layer; fact intervals decide graph visibility precisely at the traversal layer. Collapsing them (one interval per chunk) either over-restricts retrieval or under-restricts facts.

**Why does `end=None` mean "still valid"?** Regulatory facts remain in force until repealed — open-ended is the *normal* case, not an edge case. Treating "unknown" as "exclude" would silently drop the current regime from current-window answers. The cost is symmetric and honest: an auditor reads `… → open` as "no end stated", never as a confirmed eternity.

**Why dedupe relations within a document but keep cross-document duplicates?** Duplicate `(s,p,o)` inside one chunk is extraction noise. The same fact restated across documents with a different validity is *not* noise — it's the versioning signal. `upsert_document` dedupes within a document and keeps cross-document duplicates as **parallel edges** (MultiDiGraph semantics), retaining the freshest `evidence`.

**Why extractive answers by default?** A system whose purpose is auditability should not paraphrase by default. Every extractive line maps to a citation. LLM synthesis is opt-in for teams that want prose.

---

## 17. Performance notes

- **Chunking** is O(n) over pages; token counting uses tiktoken when present (`cl100k_base`), else a whitespace-word approximation (`≈ 1.3 × words`), so budgeting never depends on a network call.
- **Vector search** — LanceDB with predicate pushdown (`where` clause) so the temporal filter runs *inside* ANN. Extra candidates are fetched and the temporal filter is re-applied client-side as a recall safety net (the `+1 broad-recall` trace line).
- **Graph lens** — bounded BFS over the `MultiDiGraph` (`hops ≤ 2`, `max_graph_edges=40` default). Traversal cost is bounded by the frontier, not graph size; 10k-edge graphs answer in milliseconds.
- **Embeddings** — batched, single-flight per provider via an `RLock`; the hashing provider has zero network cost and is fully deterministic.
- **Ingestion throughput** — in `sync` mode the bridge runs the pipeline in a thread pool (the bridge stays responsive); in `celery` mode scale horizontally: each task re-opens its stores per invocation, so workers share no mutable state.
- **What dominates latency in practice** — the extraction SLM during ingestion and the answer LLM during queries, when configured. Retrieval itself (filter → ANN → seeds → traversal) is single-digit milliseconds on the reference corpus.

---

## 18. Limitations (read this before trusting it)

This is a demonstration-grade system engineered to real architectural ideas — not a certified audit platform:

- **Extraction is heuristic by default.** Fallback frames are lexicon-driven (`requires/prohibits/governs/permits/imposes/amends/supersedes…`); unusual verb constructions are missed, and 4-word object truncation mangles long objects ("interest deduction to 30" for "…to 30 percent"). Configure an SLM for production-grade frames.
- **Chunk-window inference is statistical.** The start-frame/end-frame rules fit common regulatory phrasing; mixed-frame chunks can get a wider hull than their facts. Wide hulls are conservative (extra recall, then fact-level filtering) — not wrong, but visible in `trace`.
- **No layout/table understanding.** PyMuPDF extraction is linear; PDF tables become interleaved lines. Numerical schedules (tax tables) should be pre-converted to sentences before ingest.
- **No document-internal reference resolution.** "As amended by Section 4 of Act Y" is not resolved across documents unless both are ingested and both state the relationship explicitly.
- **Single-writer stores.** LanceDB and the graph pickle tolerate one writer + many readers; two pipelines writing the same data directory is unsupported. Isolate deployments per data directory.
- **"Now" is server time.** `window=None` resolves to the bridge host clock.
- **No auth by default.** `WORKER_API_TOKEN` protects only the API→bridge hop; put a gateway in front for internet-facing deployments.
- **English-centric** month names, frames, verbs, tokenizer heuristics.

---

## 19. Roadmap

- [ ] Bitemporal `system_time` columns (query what the system believed on date D)
- [ ] Date *modifiers* ("within 30 days of", "no later than") → interval arithmetic
- [ ] Cross-document amendment chains (transitive `supersedes` walks)
- [ ] Layout-aware PDF extraction (tables → sentences)
- [ ] Multi-tenancy with per-tenant stores
- [ ] AuthN/AuthZ + document-level ACLs
- [ ] Streaming ingestion for gazette feeds
- [ ] Graph visualization with temporal animation (time scrubber)

---

## 20. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/api/health` shows `worker_upstream: false` | bridge not running / wrong `WORKER_BASE_URL` | start the bridge (`uvicorn chronos.api:app --port 8100`), check `WORKER_BASE_URL` |
| Upload stuck in `queued` | `WORKER_MODE=celery` but no broker | set `WORKER_MODE=sync` (dev) or start Redis + `celery -A chronos.celery_app worker` |
| `extractable=False` on upload | scanned PDF / no text layer | run OCR first — PyMuPDF needs a text layer |
| Answers always `"engine": "extractive"` | no LLM configured | set `MODEL_PROVIDER=openai` + `OPENAI_API_KEY` (or `ollama`) |
| Vector lens returns nothing for a historical window | chunk hulls don't overlap the window | confirm the document actually mentions those dates — hulls come from *mentioned* dates only |
| Graph lens empty while vector lens works | seed entities don't match node names | check `context.trace` seeds; "The X"/"X"/case mismatches are canonicalized automatically |
| `401 Invalid worker API token` | `WORKER_API_TOKEN` set on bridge but not sent by API | set the same value in `services/api/.env` |
| Sample generation fails | missing `fpdf2` | `.venv/Scripts/pip install fpdf2` (or `pip install -r services/worker/requirements.txt`) |
| Port already in use | previous instance alive | kill the PID from `data/*.pid`, or change `PORT` / `WORKER_PORT` |
| LanceDB errors after upgrade | dependency drift | reinstall from `requirements.txt` (pinned versions) |

---

## 21. FAQ

**Q: Can it answer "what changed between Q1 2023 and Q4 2024"?**
Yes — `POST /api/query/compare`. It diffs the interval-validated fact sets of both windows: removed (repealed), added (new regime), validity-changed (re-dated), unchanged.

**Q: Where does the LLM fit — is it required?**
No. The default `MODEL_PROVIDER=none` path is fully deterministic: regex/heuristic extraction + hashing embeddings + extractive answers. The LLM improves extraction quality and answer prose; the temporal machinery never depends on it.

**Q: How is this different from GraphRAG?**
GraphRAG builds a community hierarchy over timeless text. ChronosAudit's edges carry **validity intervals**, and both lenses (vector + graph) are **time-sliced before** ranking — retrieval answers *as of a date*, which GraphRAG does not model.

**Q: What happens when two documents disagree about the same fact in the same window?**
Both edges survive as parallel edges, both surface in context, and both appear in comparisons. Conflict *resolution* (provenance ranking, regulator hierarchy) is a roadmap item.

**Q: How do I ingest 10,000 PDFs?**
Script `POST /api/documents/upload` with `wait=false`, or use the Celery path with a real broker. Ingestion is idempotent per `doc_id`; `POST /api/documents/{id}/reindex` = delete + re-ingest.

**Q: Is `end=None` "forever" or "unknown"?**
A deliberate design decision: **in force** (open-ended). Regulation stays valid until repealed. When a source states an end date, the extractor sets `end`; otherwise the fact remains open in every later window.

**Q: Why does my answer cite `[gN]` facts I can't find in the snippet?**
Graph facts are extracted relations, not verbatim text. Each carries its `evidence` (verbatim source sentence, 240-char preview) plus `doc_id`/`chunk_id` for full context.

**Q: Can I use a real embedding model without OpenAI?**
Yes — `EMBEDDING_PROVIDER=sentence-transformers` with any local model, or `openai-compatible` pointed at any OAI-shaped `/v1/embeddings` endpoint (LM Studio, vLLM, TEI).

**Q: How do I wipe everything?**
`make clean`, or delete `services/worker/data/` — the stores are plain directories.

**Q: Why not store intervals natively in LanceDB?**
LanceDB's `where` pushdown works on scalar columns; `start`/`end` ISO strings compare lexicographically correctly for `YYYY-MM-DD`, which gives interval algebra inside SQL pushdown with zero UDFs.

---

## 22. License

MIT — see [LICENSE](LICENSE).

---

## Appendix A — Command reference

```bash
# ---- native, dev ----
make install            # venv + pip install + npm install + build API
make api                # Express API   → http://localhost:4100  (/ui/ dashboard)
make worker             # worker bridge → http://localhost:8100  (/docs OpenAPI UI)
make api-celery         # both, Celery mode (needs docker compose up redis)

# ---- docker ----
make up                 # redis + worker + api (compose)
make logs               # tail compose logs
make down               # stop compose
make clean              # stop + remove venv/node_modules

# ---- data ----
make seed               # scripts/seed_demo.py against localhost:4100
make samples            # regenerate the two demo PDFs only
make e2e                # full-stack smoke test

# ---- tests ----
make test               # pytest (worker) + API test runner
```

## Appendix B — Environment variable index

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `4100` | Express API port |
| `WORKER_BASE_URL` | `http://127.0.0.1:8100` | bridge location for the API |
| `WORKER_API_TOKEN` | *(empty)* | Bearer token for API→bridge calls |
| `DATA_DIR` | `./data/api` | uploaded PDFs + document state (API side) |
| `WEB_DIR` | `../../web` | static dashboard directory |
| `MAX_UPLOAD_MB` | `50` | upload size limit |
| `WORKER_MODE` | `sync` | `sync` (in-process) or `celery` (task queue) |
| `WORKER_PORT` | `8100` | bridge port (uvicorn invocation) |
| `MODEL_PROVIDER` | `none` | `none` / `openai` / `openai-compatible` / `ollama` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME` | — | SLM extraction + answer synthesis |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` / `openai-compatible` / `sentence-transformers` |
| `EMBEDDING_DIM` | `768` | hashing dims; ignored by model providers |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers model |
| `LANCE_DB_PATH` | `./data/lancedb` | vector store directory |
| `GRAPH_STORE_PATH` | `./data/graph/chronos.gpickle` | NetworkX pickle path |
| `GRAPH_EXPORT_PATH` | `./data/graph/exports` | GraphML/JSON export directory |
| `DOC_STATE_DIR` | `./data/state` | per-document JSON records |
| `UPLOAD_DIR` | `./data/uploads` | stored PDFs (worker side) |
| `CHUNK_SIZE_TOKENS` | `700` | target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `120` | chunk overlap |
| `MAX_GRAPH_CONTEXT_EDGES` | `60` | graph-lens cap for answers |
| `BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `RESULT_BACKEND` | `redis://localhost:6379/1` | Celery results |

## Appendix C — The one-paragraph pitch

Auditors don't ask *"what does the corpus say?"* — they ask **what did the rules require at that time, and what replaced them?** ChronosAudit is a full-stack answer to that question: PDFs in; a time-sliced vector index and a validity-interval knowledge graph out; an Express API and audit dashboard on top. Ask it about Q3 2024 and you get Q3 2024's rules, cited. Ask it to compare two dates and you get the repealed, the new, and the re-dated — each fact stamped with the exact phrase the source used. It runs entirely offline with deterministic heuristics, and accepts an extraction SLM and real embeddings when you have them.














