"""The RAG answer step: build a grounded prompt from retrieved chunks and ask the chat model."""
from __future__ import annotations

from typing import Iterator

from . import llm
from .retrieval import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. "
    "When possible, cite chunk numbers like [1], [2]."
)

NO_CONTENT_ANSWER = "I don't have any indexed content to search yet. Ingest documents first."


def build_context(results: list[RetrievedChunk]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        page = r.chunk.metadata.get("page")
        where = f"{r.chunk.document.title}, page {page}" if page else r.chunk.document.title
        blocks.append(f"[{i}] (from: {where})\n{r.chunk.text}")
    return "\n\n".join(blocks)


def user_message(question: str, results: list[RetrievedChunk]) -> str:
    return f"Context:\n{build_context(results)}\n\nQuestion:\n{question}"


def answer(question: str, results: list[RetrievedChunk]) -> str:
    if not results:
        return NO_CONTENT_ANSWER
    return llm.chat(SYSTEM_PROMPT, user_message(question, results))


def answer_stream(question: str, results: list[RetrievedChunk]) -> Iterator[str]:
    return llm.chat_stream(SYSTEM_PROMPT, user_message(question, results))


def serialize_sources(results: list[RetrievedChunk]) -> list[dict]:
    """API shape of the retrieved chunks; ref matches the [n] citations in the answer."""
    return [
        {
            "ref": i,
            "chunk_id": r.chunk.id,
            "document_id": r.chunk.document_id,
            "document_title": r.chunk.document.title,
            "page": r.chunk.metadata.get("page"),
            "similarity": r.similarity,
            "vector_rank": r.vector_rank,
            "keyword_rank": r.keyword_rank,
            "rrf_score": round(r.rrf_score, 5) if r.rrf_score is not None else None,
            "rerank_score": r.rerank_score,
            "text": r.chunk.text,
        }
        for i, r in enumerate(results, start=1)
    ]
