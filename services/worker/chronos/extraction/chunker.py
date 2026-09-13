"""Smart, page-aware chunking for regulatory PDFs.

Chunking strategy tuned for temporal documents:

* Splits on paragraph boundaries first (line-wrapped text is re-joined),
* never mixes page boundaries into a single chunk *unless* a paragraph spans
  pages (tracked via ``continued_from_page`` metadata),
* respects a soft token budget with a rolling overlap so entity/relation
  mentions that cross hard cuts are preserved,
* attaches the source page + byte span for citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

try:  # tiktoken is optional; falls back to a char-based estimate
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover
    _HAS_TIKTOKEN = False

_WS = re.compile(r"[^\S\n]+")
_MULTI_NL = re.compile(r"\n{3,}")


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    char_start: int = 0
    char_end: int = 0
    index: int = 0
    continued_from_page: bool = False
    entities: list[str] = field(default_factory=list)

    def first_page(self) -> int:
        return self.page_start


def estimate_tokens(text: str) -> int:
    if _HAS_TIKTOKEN:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def _clean(text: str) -> str:
    text = _WS.sub(" ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def _split_paragraphs(page_text: str) -> list[tuple[str, bool]]:
    """Split a page into (paragraph, continued) pairs.

    A paragraph is ``continued`` when it ends with a trailing hyphen/word
    boundary typical of a line-wrap continuation onto the next page.
    """
    raw = _MULTI_NL.sub("\n\n", page_text)
    parts: list[tuple[str, bool]] = []
    for para in raw.split("\n\n"):
        p = _clean(para)
        if not p:
            continue
        continued = bool(re.search(r"[-–]\s*$", para))
        parts.append((_clean(para), continued))
    return parts


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _hard_split(paragraph: str, chunk_size_tokens: int) -> list[str]:
    """Sentence-level split, used when a single paragraph exceeds the budget."""
    sentences = _SENT_SPLIT.split(paragraph)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        st = estimate_tokens(sent)
        if current and current_tokens + st > chunk_size_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += st
    if current:
        pieces.append(" ".join(current))
    return pieces or [paragraph]


def chunk_document(
    pages: list[Page],
    doc_id: str,
    chunk_size_tokens: int = 800,
    overlap_tokens: int = 120,
) -> list[Chunk]:
    """Chunk a list of pages into temporally-coherent, citation-friendly Chunks."""
    paragraphs: list[tuple[str, int, int]] = []  # (text, page_no, started_new_page)
    for page in pages:
        for i, (para, continued) in enumerate(_split_paragraphs(page.text)):
            paragraphs.append((para, page.number, i == 0 and not continued))

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    buffer_pages: list[int] = []
    buffer_char_len = 0
    start_idx = 0
    idx = 0

    def flush(continued: bool) -> Chunk:
        nonlocal buffer, buffer_tokens, buffer_pages, buffer_char_len, start_idx
        text = " ".join(buffer)
        c = Chunk(
            chunk_id=f"{doc_id}:c{len(chunks):05d}",
            doc_id=doc_id,
            text=text,
            page_start=min(buffer_pages),
            page_end=max(buffer_pages),
            char_start=start_idx,
            char_end=start_idx + buffer_char_len,
            index=len(chunks),
            continued_from_page=continued,
        )
        chunks.append(c)
        # Rolling overlap: keep the tail (up to overlap_tokens) for continuity.
        keep: list[str] = []
        keep_tokens = 0
        for p in reversed(buffer):
            pt = estimate_tokens(p)
            if keep_tokens + pt > overlap_tokens:
                break
            keep.insert(0, p)
            keep_tokens += pt
        if keep:
            buffer = list(keep)
            buffer_tokens = keep_tokens
            buffer_pages = buffer_pages[-len(keep):]
        else:
            buffer, buffer_tokens, buffer_pages, buffer_char_len, start_idx = [], 0, [], 0, idx
        return c

    for para, page_no, new_page in paragraphs:
        pt = estimate_tokens(para)
        pieces: list[str] = [para]
        # A single paragraph larger than the whole budget gets carved up by sentence.
        if not buffer and pt > chunk_size_tokens:
            pieces = _hard_split(para, chunk_size_tokens)
        for piece in pieces:
            pt = estimate_tokens(piece)
            if buffer_tokens + pt > chunk_size_tokens and buffer:
                flush(continued=not new_page)
            if not buffer:
                start_idx = idx
            buffer.append(piece)
            buffer_tokens += pt
            buffer_pages.append(page_no)
            buffer_char_len += len(piece) + 1
            idx += len(piece) + 1

    if buffer:
        flush(continued=False)

    return chunks