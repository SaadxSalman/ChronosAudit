"""Deterministic, zero-dependency extraction fallback.

Used when ``MODEL_PROVIDER=none`` (or when the configured LLM is unreachable).
It is intentionally simple but *functional*: it extracts

* date-anchored intervals (ISO dates, quarters, month+year, year),
* entities as frequent noun phrases / proper nouns,
* relations via lightweight subject-verb-object heuristics and date frames
  (``<Entity> <verb> <Object> on <date>`` patterns).

The fallback guarantees the demo works offline and gives the retrieval layer
real (if coarse) temporal signal.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from chronos.models import EntityOccurrence, ExtractionResult, RelationOccurrence, TemporalValidity

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "at", "with", "by",
    "from", "this", "that", "these", "those", "as", "is", "are", "was", "were", "be",
    "been", "shall", "will", "must", "may", "should", "not", "no", "per", "its", "it",
    "all", "any", "each", "such", "which", "who", "whom", "whose", "within", "under",
}
_VERBS = {
    "requires", "prohibits", "governs", "amends", "replaces", "supersedes", "imposes",
    "applies", "mandates", "allows", "permits", "defines", "establishes", "sets",
    "prescribes", "obliges", "limits", "restricts", "bars", "forbids", "authorizes",
}
# Time-marker verbs create only weak frames ("X effective <date>"); they are used
# only when a sentence has no substantive relation verb.
_TIMING_VERBS = {"effective", "take", "took", "becomes", "became", "ends", "expires", "commences"}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_Q_MONTHS = {"q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12)}

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_SLASH_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def _parse_date_literal(token: str) -> Optional[date]:
    m = _ISO_DATE.search(token)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = _SLASH_DATE.search(token)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None
    return None


def find_dates_in_text(text: str) -> list[tuple[date, str]]:
    """All explicit dates present in the text: (date, raw_label).

    Bare years that are *inside* an ISO/US date token are skipped so that
    "starting 2024-07-01" yields one date, not {2024-01-01, 2024-07-01}.
    """
    found: list[tuple[date, str]] = []
    occupied: set[tuple[int, int]] = set()

    for m in _ISO_DATE.finditer(text):
        d = _parse_date_literal(m.group(0))
        if d:
            found.append((d, m.group(0)))
            occupied.add(m.span())
    for m in _SLASH_DATE.finditer(text):
        d = _parse_date_literal(m.group(0))
        if d:
            found.append((d, m.group(0)))
            occupied.add(m.span())

    def inside_occupied(span: tuple[int, int]) -> bool:
        return any(s <= span[0] and span[1] <= e for s, e in occupied)

    low = text.lower()
    # Month + year / quarter + year
    for mon, mnum in _MONTHS.items():
        for m in re.finditer(rf"\b{mon}\s+(\d{{4}})\b", low):
            found.append((date(int(m.group(1)), mnum, 1), m.group(0)))
    for q, (m0, m1) in _Q_MONTHS.items():
        for m in re.finditer(rf"\b{q}\s*,?\s*(\d{{4}})\b", low):
            from calendar import monthrange

            _, last = monthrange(int(m.group(1)), m1)
            found.append((date(int(m.group(1)), m0, 1), m.group(0)))
            found.append((date(int(m.group(1)), m1, last), m.group(0)))
    # Bare years (skip spans already captured by an ISO/SLASH date)
    for m in _YEAR.finditer(text):
        if inside_occupied(m.span()):
            continue
        d = date(int(m.group(0)), 1, 1)
        if all(d != fd for fd, _ in found):
            found.append((d, m.group(0)))
    found.sort(key=lambda x: x[0])
    return found


def infer_chunk_window(text: str) -> TemporalValidity:
    """Heuristic chunk-level validity: min/max dates mentioned on the chunk."""
    dates = find_dates_in_text(text)
    if not dates:
        return TemporalValidity()
    return TemporalValidity(
        start=dates[0][0], end=dates[-1][0],
        start_label=dates[0][1], end_label=dates[-1][1],
    )


def _candidate_entities(sentences: list[str]) -> list[str]:
    """Frequent Capitalized noun phrases of length 2-4, deduped, stopword-filtered.

    Strips leading articles ("The Cross-Border Data Regulation" → also records
    "Cross-Border Data Regulation") so the real subject survives the greedy
    regex match.
    """
    freqs: dict[str, int] = {}
    leads = {"the", "a", "an"}

    def record(phrase: str) -> None:
        phrase = phrase.strip()
        words = phrase.split()
        if not (1 <= len(words) <= 4):
            return
        if words[0].lower() in _STOPWORDS:
            # strip trailing stopwords from the head of the phrase
            while words and words[0].lower() in leads:
                words = words[1:]
            if not words:
                return
        freqs[phrase] = freqs.get(phrase, 0) + 1
        if len(words) >= 2:
            freqs[" ".join(words)] = freqs.get(" ".join(words), 0) + 1

    for sent in sentences:
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3})\b", sent):
            record(m.group(1).strip())
    # Prefer terms that look regulatory; keep top 24 by freq, then deterministic sort
    candidates = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))
    return [phrase for phrase, _ in candidates[:24]]


def _classify_entity(name: str, text: str) -> str:
    low = f" {name.lower()} "
    low_text = text.lower()
    if any(k in low for k in ("regulation", "act", "directive", "standard", "policy", "protocol", "guideline", "circular")):
        return "Regulation"
    if any(k in low for k in ("firm", "bank", "authority", "company", "corp", "ltd", "inc", "commission", "agency", "board")):
        return "Organization"
    if any(k in low for k in ("tax", "charge", "rate", "fee", "levy", "credit", "deduction", "threshold")):
        return "Instrument"
    if any(k in low for k in ("party", "citizen", "resident", "taxpayer", "entity", "customer", "person")):
        return "Party"
    for country in ("eu", "us", "uk", "usa", "germany", "france", "japan", "china", "canada", "india"):
        if country in low or country in low_text:
            return "Jurisdiction"
    return "Concept"


def extract_fallback(text: str, chunk_id: str = "", doc_name: str = "") -> ExtractionResult:
    """Deterministic extraction used when no SLM is configured."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunk_window = infer_chunk_window(text)

    # ---- Entities ----------------------------------------------------------
    entity_names = _candidate_entities(sentences)
    entities: list[EntityOccurrence] = []
    entity_ids: dict[str, str] = {}
    for i, name in enumerate(entity_names):
        eid = f"{chunk_id or 'chunk'}:e{i}"
        entity_ids[name.lower()] = eid
        entities.append(
            EntityOccurrence(
                id=eid,
                name=name,
                type=_classify_entity(name, text),
                aliases=[],
                confidence=0.5,
            )
        )

    # ---- Relations via VERB + (on/as of/effective/until date) frames -------
    relations: list[RelationOccurrence] = []
    used: set[str] = set()
    verb_order = sorted(_VERBS, key=len, reverse=True) + sorted(_TIMING_VERBS, key=len, reverse=True)
    for sent in sentences:
        low = sent.lower()
        for verb in verb_order:
            if f" {verb} " not in f" {low} ":
                continue
            vpos = low.find(verb)
            subj = None
            for cand in entity_names:
                pos = low.find(cand.lower())
                if pos != -1 and pos < vpos:
                    subj = cand
            # object: words after verb up to a date or sentence end
            tail = sent[vpos + len(verb):]
            tail = re.split(r"\b(as of|until|effective|on|commencing)\b", tail, maxsplit=1, flags=re.I)[0]
            tail = re.split(r"[.!?;]|\(see", tail)[0].strip(" ,;:-")
            words = tail.split()
            if not words:
                continue
            obj = " ".join(words[:4]).strip()
            if not obj or len(obj) < 2:
                continue
            if obj.lower() in _STOPWORDS:
                continue
            key = (subj or "?", verb, obj)
            if key in used:
                continue
            used.add(key)

            dates = find_dates_in_text(sent)
            validity = TemporalValidity()
            if dates:
                validity.start = dates[0][0]
                validity.start_label = dates[0][1]
                if len(dates) > 1:
                    validity.end = dates[-1][0]
                    validity.end_label = dates[-1][1]
                elif re.search(r"\b(before|until|through|to)\b.*", low):
                    validity.end = validity.start
                    validity.end_label = validity.start_label

            relations.append(
                RelationOccurrence(
                    subject=subj or "",
                    predicate=verb,
                    object=obj,
                    validity=validity,
                    sentence=sent[:300],
                    confidence=0.55,
                )
            )
            break

    return ExtractionResult(
        chunk_id=chunk_id,
        entities=entities,
        relations=relations,
        chunk_temporal=chunk_window,
    )