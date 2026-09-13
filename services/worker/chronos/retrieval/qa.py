"""Answer synthesis on top of temporal retrieval context.

When an LLM is configured (OpenAI / Ollama) the engine asks it to produce a
JSON answer with inline citations and explicit "unknowns". When no LLM is
available (offline/deterministic mode) it builds a transparent extractive
summary that auditors can audit as easily as the RAG output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from chronos.config import Settings, get_settings
from chronos.extraction.slm import _LLM, build_llm
from chronos.models import (
    AnswerResponse,
    CompareResponse,
    Comparison,
    RetrievalContext,
    TemporalValidity,
)

log = logging.getLogger("chronos.retrieval.qa")


def _edge_to_fact(e) -> dict:
    return {
        "subject": e.subject,
        "predicate": e.predicate,
        "object": e.object,
        "validity": str(TemporalValidity(start=e.start, end=e.end, start_label=e.start_label, end_label=e.end_label)),
        "start": e.start.isoformat() if e.start else None,
        "end": e.end.isoformat() if e.end else None,
        "doc_id": e.doc_id,
        "chunk_id": e.chunk_id,
        "evidence": e.evidence,
    }


def _build_prompt(question: str, ctx: RetrievalContext) -> str:
    blocks = [f"Question: {question}"]
    blocks.append(
        f"Requested time window: {ctx.window_label or 'untemporal'} ({ctx.window_start or '…'} → {ctx.window_end or '…'})"
    )
    blocks.append("")
    blocks.append("## VECTOR EVIDENCE")
    for i, h in enumerate(ctx.hits):
        blocks.append(
            f"[v{i}] (doc={h.doc_name}, page={h.page}, validity={h.start_label or h.start} → {h.end_label or h.end})\n{h.text[:1200]}"
        )
    blocks.append("")
    blocks.append("## FACT EDGES")
    for i, e in enumerate(ctx.graph_edges):
        v = TemporalValidity(start=e.start, end=e.end, start_label=e.start_label, end_label=e.end_label)
        blocks.append(f"[g{i}] {e.subject} --[{e.predicate}]--> {e.object}  (validity: {v}, src={e.doc_id})")
    blocks.append("")
    blocks.append(
        '## Instruction\nAnswer the question for the requested window using only the evidence above. '
        'If asked to compare with "current" regulations and the current window is not in the context, '
        'use the most recent evidence and say so. Return JSON: {"answer": "...with [v0]/[g1] citations", '
        '"facts": ["..."], "confidence": 0-1, "unknowns": ["..."]}'
    )
    return "\n".join(blocks)


class AnswerEngine:
    def __init__(self, settings: Optional[Settings] = None, llm: Optional[_LLM] = None):
        self.settings = settings or get_settings()
        self.llm = llm if llm is not None else build_llm(self.settings)

    @property
    def engine_name(self) -> str:
        return self.settings.model_provider if self.llm else "extractive"
# ---------------------------------------------------------------- Q&A
    def answer(self, question: str, ctx: RetrievalContext) -> AnswerResponse:
        if self.llm is not None:
            try:
                payload = self._llm_answer(question, ctx)
                return AnswerResponse(
                    question=question,
                    answer=str(payload.get("answer", "")).strip(),
                    citations=self._citations(ctx),
                    facts=payload.get("facts") or self._default_facts(ctx),
                    context=ctx,
                    engine=f"llm:{self.settings.model_provider}",
                )
            except Exception as exc:  # noqa: BLE001 - degrade to extractive
                log.warning("LLM answer failed (%s); using extractive engine.", exc)
        return self._extractive_answer(question, ctx)

    def _llm_answer(self, question: str, ctx: RetrievalContext) -> dict:
        reply = self.llm.invoke(_build_prompt(question, ctx))
        cleaned = reply.strip().strip("```")
        cleaned = re.sub(r"^json\s*", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if 0 <= start < end:
                return json.loads(cleaned[start:end + 1])
            raise ValueError("LLM did not return JSON")

    def _extractive_answer(self, question: str, ctx: RetrievalContext) -> AnswerResponse:
        """Deterministic, fully-transparent answer built from the context."""
        lines = [f"Evidence for “{ctx.window_label or 'open window'}” (extractive engine):"]
        for i, h in enumerate(ctx.hits[:5]):
            v = f"{h.start_label or h.start or '…'} → {h.end_label or h.end or '…'}"
            snippet = re.sub(r"\s+", " ", h.text)[:220]
            lines.append(f"[v{i}] ({h.doc_name}, p{h.page}, valid {v}) {snippet}…")
        for i, e in enumerate(ctx.graph_edges[:20]):
            v = TemporalValidity(start=e.start, end=e.end, start_label=e.start_label, end_label=e.end_label)
            lines.append(f"[g{i}] {e.subject} --[{e.predicate}]--> {e.object} (valid {v})")
        if not ctx.hits and not ctx.graph_edges:
            lines.append("Insufficient evidence within the requested window.")
        return AnswerResponse(
            question=question,
            answer="\n".join(lines),
            citations=self._citations(ctx),
            facts=self._default_facts(ctx),
            context=ctx,
            engine="extractive",
        )

    @staticmethod
    def _citations(ctx: RetrievalContext) -> list[dict]:
        out = []
        for i, h in enumerate(ctx.hits):
            out.append(
                {
                    "key": f"v{i}",
                    "doc_id": h.doc_id,
                    "doc_name": h.doc_name,
                    "page": h.page,
                    "chunk_id": h.chunk_id,
                    "validity": f"{h.start_label or h.start or '…'} → {h.end_label or h.end or '…'}",
                    "score": round(h.score, 4),
                    "snippet": re.sub(r"\s+", " ", h.text)[:280],
                }
            )
        for i, e in enumerate(ctx.graph_edges):
            out.append(
                {
                    "key": f"g{i}",
                    "kind": "graph_fact",
                    "doc_id": e.doc_id,
                    "chunk_id": e.chunk_id,
                    "fact": f"{e.subject} → {e.predicate} → {e.object}",
                    "validity": str(TemporalValidity(start=e.start, end=e.end, start_label=e.start_label, end_label=e.end_label)),
                    "evidence": e.evidence,
                }
            )
        return out

    @staticmethod
    def _default_facts(ctx: RetrievalContext) -> list[dict]:
        return [_edge_to_fact(e) for e in ctx.graph_edges[:30]]
# ------------------------------------------------------------ compare
    def compare(self, question: str, ctx_a: RetrievalContext, ctx_b: RetrievalContext) -> CompareResponse:
        sig = lambda e: (e.subject, e.predicate, e.object)  # noqa: E731
        keys_a = {sig(e) for e in ctx_a.graph_edges}
        keys_b = {sig(e) for e in ctx_b.graph_edges}

        added = [e for e in ctx_b.graph_edges if sig(e) not in keys_a]
        removed = [e for e in ctx_a.graph_edges if sig(e) not in keys_b]
        common = [e for e in ctx_b.graph_edges if sig(e) in keys_a]
        by_sig_a = {sig(e): e for e in ctx_a.graph_edges}
        changed = [
            {"before": by_sig_a[sig(e)], "after": e}
            for e in common
            if e.start != by_sig_a[sig(e)].start or e.end != by_sig_a[sig(e)].end
        ]
        unchanged = [e for e in common if e.start == by_sig_a[sig(e)].start and e.end == by_sig_a[sig(e)].end]

        comp = Comparison(
            window_start=ctx_b.window_start,
            window_end=ctx_b.window_end,
            window_label=ctx_b.window_label,
            added=added,
            removed=removed,
            changed=changed,
            unchanged=unchanged,
            facts=[f"{e.subject} {e.predicate} {e.object}" for e in removed],
        )

        lines = [
            f"Comparing “{ctx_a.window_label}” vs “{ctx_b.window_label}”:",
            f"- Added: {len(added)} fact(s)",
            f"- Removed: {len(removed)} fact(s)",
            f"- Changed validity/evidence: {len(changed)} fact(s)",
        ]
        for e in removed:
            lines.append(f"  ✗ {e.subject} --[{e.predicate}]--> {e.object} (was valid {e.start} → {e.end})")
        for c in changed:
            lines.append(f"  ↻ {c['after'].subject} --[{c['after'].predicate}]--> {c['after'].object}")
        for e in added:
            lines.append(f"  ✓ {e.subject} --[{e.predicate}]--> {e.object} (valid {e.start} → {e.end})")

        return CompareResponse(
            question=question,
            window_a_label=ctx_a.window_label,
            window_b_label=ctx_b.window_label,
            comparison=comp,
            context_a=ctx_a,
            context_b=ctx_b,
            narrative="\n".join(lines),
            engine=f"compare:{self.engine_name}",
        )