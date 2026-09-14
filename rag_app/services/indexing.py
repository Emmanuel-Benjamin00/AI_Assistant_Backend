"""Chunk, embed and store a document. Shared by the text and file ingest endpoints and seed_demo."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from django.db import transaction

from ..chunking import chunk_text
from ..models import Chunk, Document
from . import llm
from .extraction import Section

logger = logging.getLogger(__name__)


class NothingToIndex(ValueError):
    """The document has no text to index."""


@dataclass(frozen=True)
class PlannedChunk:
    text: str
    metadata: dict


def plan_chunks(sections: list[Section]) -> list[PlannedChunk]:
    """
    Chunk each section separately so a chunk never spans two PDF pages.

    metadata["index"] is the chunk's position in the whole document; "page" is set when known.
    """
    planned: list[PlannedChunk] = []
    for section in sections:
        for text in chunk_text(section.text):
            metadata = {"index": len(planned)}
            if section.page is not None:
                metadata["page"] = section.page
            planned.append(PlannedChunk(text, metadata))
    return planned


def index_document(
    title: str,
    sections: list[Section],
    *,
    source: str = Document.Source.TEXT,
    filename: str = "",
) -> tuple[Document, int]:
    """
    Embed and save a document; returns (document, chunk count).

    Raises NothingToIndex, llm.LLMConfigError or llm.LLMRequestError. Nothing is saved on failure.
    """
    planned = plan_chunks(sections)
    if not planned:
        raise NothingToIndex("The document has no text to index.")

    # Embed before opening the transaction so no DB connection is held during network calls.
    vectors = llm.embed_texts([p.text for p in planned])
    if len(vectors) != len(planned):
        logger.error("Embedding count mismatch: %d chunks, %d vectors", len(planned), len(vectors))
        raise llm.LLMRequestError("embedding count mismatch")

    with transaction.atomic():
        doc = Document.objects.create(title=title, source=source, filename=filename)
        Chunk.objects.bulk_create(
            [
                Chunk(document=doc, text=p.text, embedding=v, metadata=p.metadata)
                for p, v in zip(planned, vectors, strict=True)
            ]
        )

    logger.info("Indexed document %s (%s, %d chunks)", doc.id, source, len(planned))
    return doc, len(planned)
