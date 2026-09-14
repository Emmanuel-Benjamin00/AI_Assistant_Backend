from __future__ import annotations

import re
from typing import List


_PARA_SPLIT_RE = re.compile(r"\n\s*\n+", flags=re.MULTILINE)

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 200


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[str]:
    """
    Split text into chunks of at most max_chars, keeping paragraphs together.

    - Paragraphs (blank-line separated) are packed into one chunk until the next would overflow
    - A paragraph longer than max_chars is split on sentence or word boundaries, with
      overlap_chars carried into the next piece so a fact cut at a boundary survives whole
    """
    if overlap_chars >= max_chars // 2:
        raise ValueError("overlap_chars must be less than half of max_chars")

    cleaned = (text or "").strip()
    if not cleaned:
        return []

    pieces: List[str] = []
    for paragraph in _PARA_SPLIT_RE.split(cleaned):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_long(paragraph, max_chars, overlap_chars))

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 2 + len(piece) <= max_chars:
            current = f"{current}\n\n{piece}"
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    parts: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            # Prefer ending on a sentence, then a word, in the back half of the window.
            floor = start + max_chars // 2
            sentence_end = text.rfind(". ", floor, end)
            if sentence_end != -1:
                end = sentence_end + 1
            else:
                space = text.rfind(" ", floor, end)
                if space != -1:
                    end = space
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= length:
            break
        # end is past the window midpoint and overlap is under half, so start always advances.
        start = end - overlap_chars
        if text[start - 1] != " ":
            next_space = text.find(" ", start, end)
            if next_space != -1:
                start = next_space + 1
    return parts
