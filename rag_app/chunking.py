from __future__ import annotations

import re
from typing import List


_PARA_SPLIT_RE = re.compile(r"\n\s*\n+", flags=re.MULTILINE)


def chunk_text_paragraphs(text: str, *, min_chars: int = 1) -> List[str]:
    """
    V1 chunking: split on blank lines (paragraph boundaries).

    - Removes empty chunks
    - Trims whitespace
    - Drops tiny chunks (often noise) via min_chars (set higher later if needed)
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    parts = [p.strip() for p in _PARA_SPLIT_RE.split(cleaned)]
    parts = [p for p in parts if len(p) >= min_chars]

    return parts

