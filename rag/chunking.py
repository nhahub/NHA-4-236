"""Shared text-cleaning and chunking utilities used by all source loaders.

Kept separate from ``ingest`` and the loaders so every dataset produces the
same ``Passage`` shape with the same ~300-500 token chunking, and so loaders
can import these helpers without a circular dependency on ``ingest``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import settings

_WS_RE = re.compile(r"\s+")
_HTML_RE = re.compile(r"<[^>]+>")


@dataclass
class Passage:
    """A retrievable chunk plus the metadata needed to cite it.

    The schema is shared across every dataset (MedQuAD, PubMedQA, MedlinePlus,
    Symptom2Disease); each loader maps its records onto these fields so the
    retriever stays source-agnostic.
    """

    id: str
    text: str
    question: str
    title: str
    url: str
    source: str
    qtype: str


def clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return _WS_RE.sub(" ", (text or "").strip())


def strip_html(text: str) -> str:
    """Remove HTML tags (MedlinePlus summaries are HTML) and normalize space."""
    return clean(_HTML_RE.sub(" ", text or ""))


def approx_tokens(text: str) -> int:
    # Cheap word-count proxy for tokens; good enough for chunk sizing.
    return len(text.split())


def chunk_text(
    text: str,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """Split text into ~target_tokens word-windows with a small overlap.

    Splits on sentence boundaries first, then packs sentences into windows so
    chunks stay semantically coherent rather than cutting mid-sentence. A single
    sentence longer than the target is broken into word windows so run-on text
    (or text with no punctuation) still chunks.
    """
    target = target_tokens or settings.chunk_target_tokens
    overlap = overlap_tokens or settings.chunk_overlap_tokens

    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences: list[str] = []
    for sentence in raw_sentences:
        words = sentence.split()
        if len(words) > target:
            for i in range(0, len(words), target):
                sentences.append(" ".join(words[i : i + target]))
        elif sentence.strip():
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = approx_tokens(sentence)
        if current and current_len + s_len > target:
            chunks.append(" ".join(current))
            if overlap > 0:
                tail = " ".join(current).split()[-overlap:]
                current = [" ".join(tail)]
                current_len = len(tail)
            else:
                current = []
                current_len = 0
        current.append(sentence)
        current_len += s_len

    if current:
        chunks.append(" ".join(current))
    return [c for c in (c.strip() for c in chunks) if c]
