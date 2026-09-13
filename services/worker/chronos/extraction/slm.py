"""Extraction SLM (small language model) wrapper.

The extraction SLM is the component that turns a text chunk into structured,
*time-annotated* knowledge: entities, relation triples and explicit validity
intervals. Providers:

* ``openai`` — ChatOpenAI (``gpt-4o-mini`` by default) via LangChain,
* ``ollama``  — local ChatOllama (e.g. ``llama3.2``) via LangChain,
* ``none``    — deterministic heuristic extractor (``extract_fallback``) that
  needs zero credentials and keeps the offline demo fully functional.

If an LLM provider is configured but unreachable, we degrade gracefully to the
fallback extractor and record the reason on the trace.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Protocol

from chronos.config import Settings, get_settings
from chronos.extraction.fallback import extract_fallback
from chronos.extraction.prompts import EXTRACTION_SYSTEM
from chronos.models import EntityOccurrence, ExtractionResult, RelationOccurrence, TemporalValidity

log = logging.getLogger("chronos.extraction.slm")


class _LLM(Protocol):
    temperature: float

    def invoke(self, prompt: str) -> str: ...  # noqa: E704


def _coerce_date(value: Any) -> Any:
    if value in (None, "", "null", "none", "N/A"):
        return None
    if isinstance(value, str):
        m = re.match(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+(\d{2}:\d{2}:\d{2}))?\s*$", value)
        if m:
            from datetime import date, datetime

            try:
                if m.group(4):
                    parsed = datetime.strptime(
                        f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}", "%Y-%m-%d %H:%M:%S"
                    )
                    return parsed.date()
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    return value


def _coerce_validity(raw: Optional[dict]) -> TemporalValidity:
    if not raw:
        return TemporalValidity()
    start = _coerce_date(raw.get("start"))
    end = _coerce_date(raw.get("end"))
    return TemporalValidity(
        start=start,
        end=end,
        start_label=raw.get("start_label") or (raw.get("start") if isinstance(raw.get("start"), str) and start else None),
        end_label=raw.get("end_label") or (raw.get("end") if isinstance(raw.get("end"), str) and end else None),
    )


def _extract_json(text: str) -> tuple[dict, bool]:
    """Pull the first JSON object out of a model reply, tolerating markdown."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        pass
    # Salvage attempt via bracket match
    start, end = None, None
    for i, ch in enumerate(cleaned):
        if ch == "{" and start is None:
            start = i
        if ch == "}":
            end = i
    if start is not None and end is not None and end > start:
        try:
            return json.loads(cleaned[start:end + 1]), True
        except json.JSONDecodeError:
            pass
    return {}, False


def _normalise_result(raw: dict, chunk_id: str) -> ExtractionResult:
    entities = []
    for e in raw.get("entities", []) or []:
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        entities.append(
            EntityOccurrence(
                id=f"{chunk_id}:e{len(entities)}",
                name=name,
                type=str(e.get("type", "Other")) or "Other",
                aliases=[str(a) for a in (e.get("aliases") or []) if a],
                sentence="",
                confidence=float(e.get("confidence", 0.5)),
            )
        )
    relations = []
    for r in raw.get("relations", []) or []:
        subj = str(r.get("subject", "")).strip()
        pred = str(r.get("predicate", "")).strip()
        obj = str(r.get("object", "")).strip()
        if not (subj and pred and obj):
            continue
        relations.append(
            RelationOccurrence(
                subject=subj,
                predicate=pred,
                object=obj,
                validity=_coerce_validity(r.get("validity")),
                sentence=str(r.get("sentence", ""))[:300],
                confidence=float(r.get("confidence", 0.5)),
            )
        )
    return ExtractionResult(
        chunk_id=chunk_id,
        entities=entities,
        relations=relations,
        chunk_temporal=_coerce_validity(raw.get("chunk_temporal")),
        raw_json=raw,
    )


class ExtractionSLM:
    """Facade over the configured extraction provider."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.llm = build_llm(self.settings)
        self.provider = self.settings.model_provider if self.llm else "none"

    def extract(self, text: str, chunk_id: str, doc_name: str = "", page: int = 0) -> tuple[ExtractionResult, str]:
        """Extract structured knowledge. Returns (result, engine_name)."""
        if self.llm is None:
            return extract_fallback(text, chunk_id=chunk_id, doc_name=doc_name), "fallback"
        try:
            user = (
                f"Document: {doc_name}\nPage: {page}\nChunk: {chunk_id}\n"
                f"--- Snapshot start ---\n{text}\n--- Snapshot end ---\n"
                "Extract the entities, relations and temporal intervals now."
            )
            prompt = self.llm.invoke(f"{EXTRACTION_SYSTEM}\n\n---\n\n{user}")
            raw, ok = _extract_json(prompt)
            if not ok:
                raise ValueError("SLM did not return parseable JSON")
            result = _normalise_result(raw, chunk_id)
            if not result.entities and not result.relations:
                raise ValueError("SLM returned an empty extraction")
            return result, self.settings.model_provider
        except Exception as exc:  # noqa: BLE001 - graceful degradation is the point
            log.warning("SLM extraction failed (%s); using fallback extractor.", exc)
            return extract_fallback(text, chunk_id=chunk_id, doc_name=doc_name), "fallback"


def build_llm(settings: Settings) -> Optional[_LLM]:
    """Instantiate the configured LangChain chat model."""
    if settings.model_provider == "openai":
        if not settings.openai_api_key:
            log.warning("MODEL_PROVIDER=openai but OPENAI_API_KEY is empty; falling back.")
            return None
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.openai_extraction_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
                temperature=settings.openai_temperature,
                max_retries=1,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to initialise OpenAI SLM: %s", exc)
            return None
    if settings.model_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=settings.model_name,
                base_url=settings.ollama_base_url,
                temperature=settings.ollama_temperature,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to initialise Ollama SLM: %s", exc)
            return None
    return None