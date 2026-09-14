"""
Find the chunks most relevant to a question.

  vector   cosine distance on embeddings (pgvector, HNSW index): good at meaning and paraphrase
  keyword  Postgres full-text search (tsvector, GIN index): good at exact terms like "835" or "CO"
  hybrid   both, merged with reciprocal rank fusion (RRF)
  rerank   optional second pass: the chat model scores each candidate for relevance
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from django.contrib.postgres.search import SearchQuery, SearchRank
from django.db.models import F
from pgvector.django import CosineDistance

from ..models import SEARCH_CONFIG, Chunk
from . import llm

logger = logging.getLogger(__name__)

VECTOR = "vector"
HYBRID = "hybrid"
MODES = (VECTOR, HYBRID)
DEFAULT_MODE = os.getenv("RETRIEVAL_MODE", HYBRID)

# Standard RRF constant from the original paper; larger values flatten rank differences.
RRF_K = 60
# How many results each search contributes to the fusion, per requested result.
CANDIDATE_MULTIPLIER = 4
# How many fused candidates the re-ranker reads. More = better recall, more tokens.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "12"))
RERANK_SNIPPET_CHARS = 700

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


def retrieve(question: str, top_k: int, mode: str = DEFAULT_MODE, rerank: bool = False) -> list[RetrievedChunk]:
    """Embed the question and return up to top_k chunks, best first."""
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}")
    q_vec = llm.embed_texts([question])[0]
    return retrieve_with_vector(question, q_vec, top_k, mode=mode, rerank=rerank)


def retrieve_with_vector(
    question: str, q_vec: list[float], top_k: int, *, mode: str = DEFAULT_MODE, rerank: bool = False
) -> list[RetrievedChunk]:
    wanted = max(top_k, RERANK_CANDIDATES) if rerank else top_k
    if mode == VECTOR:
        results = vector_search(q_vec, wanted)
    else:
        results = hybrid_search(question, q_vec, wanted)
    if rerank and results:
        results = rerank_chunks(question, results, top_k)
    return results[:top_k]


def vector_search(q_vec: list[float], limit: int) -> list[RetrievedChunk]:
    rows = (
        Chunk.objects.filter(embedding__isnull=False)
        .select_related("document")
        .annotate(distance=CosineDistance("embedding", q_vec))
        .order_by("distance")[:limit]
    )
    return [
        RetrievedChunk(chunk=c, similarity=_similarity(c.distance), vector_rank=rank)
        for rank, c in enumerate(rows, start=1)
    ]


def keyword_query(question: str) -> SearchQuery | None:
    """
    OR the question's words together: "what does CO mean" -> 'what | does | co | mean'.

    Postgres' plain/websearch queries AND every word, so a natural-language question would
    almost never match. The english config stems each word and drops stop words. Only
    [a-z0-9] survives, so user input can never break the tsquery syntax.
    """
    words = list(dict.fromkeys(_WORD_RE.findall(question.lower())))
    if not words:
        return None
    return SearchQuery(" | ".join(words), search_type="raw", config=SEARCH_CONFIG)


def keyword_search(question: str, limit: int, q_vec: list[float] | None = None) -> list[RetrievedChunk]:
    query = keyword_query(question)
    if query is None:
        return []
    rows = (
        Chunk.objects.filter(search_vector=query)
        .select_related("document")
        # cover_density ranks chunks where matched words appear close together higher.
        .annotate(rank=SearchRank(F("search_vector"), query, cover_density=True))
    )
    if q_vec is not None:
        rows = rows.annotate(distance=CosineDistance("embedding", q_vec))
    rows = rows.order_by("-rank", "id")[:limit]
    return [
        RetrievedChunk(
            chunk=c,
            similarity=_similarity(getattr(c, "distance", None)),
            keyword_rank=rank,
        )
        for rank, c in enumerate(rows, start=1)
    ]


def hybrid_search(question: str, q_vec: list[float], limit: int) -> list[RetrievedChunk]:
    candidates = limit * CANDIDATE_MULTIPLIER
    return fuse(vector_search(q_vec, candidates), keyword_search(question, candidates, q_vec))[:limit]


def fuse(vector_results: list[RetrievedChunk], keyword_results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Reciprocal rank fusion: score = sum over lists of 1 / (RRF_K + rank).

    Uses ranks only, so cosine similarity (0..1) and ts_rank (unbounded) never need to be
    put on the same scale. A chunk found by both searches beats one found by only one.
    """
    merged: dict[int, RetrievedChunk] = {}
    for results, rank_attr in ((vector_results, "vector_rank"), (keyword_results, "keyword_rank")):
        for item in results:
            rank = getattr(item, rank_attr)
            entry = merged.setdefault(item.chunk.id, item)
            setattr(entry, rank_attr, rank)
            entry.rrf_score = (entry.rrf_score or 0.0) + 1.0 / (RRF_K + rank)
    # Ties (same fused score) fall back to the better vector rank, then chunk id for stability.
    return sorted(
        merged.values(),
        key=lambda r: (-r.rrf_score, r.vector_rank or float("inf"), r.chunk.id),
    )


RERANK_PROMPT = (
    "You score how useful each passage is for answering the question. "
    "Score 0 (irrelevant) to 10 (directly answers it). Judge only the passage text. "
    'Reply with JSON only: {"scores": [{"id": <passage id>, "score": <0-10>}, ...]} '
    "with one entry for every passage."
)


def rerank_chunks(question: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """
    Re-order candidates by an LLM relevance score (a "listwise LLM re-ranker").

    A cross-encoder model is the other common choice; it needs PyTorch and a GPU-friendly
    host, which is heavy for App Service. If scoring fails, the first-pass order is kept,
    so a re-ranking failure never breaks a request.
    """
    candidates = results[:RERANK_CANDIDATES]
    passages = "\n\n".join(
        f"[id={i}]\n{r.chunk.text[:RERANK_SNIPPET_CHARS]}" for i, r in enumerate(candidates)
    )
    try:
        reply = llm.chat_json(RERANK_PROMPT, f"Question:\n{question}\n\nPassages:\n{passages}")
        scores = {int(s["id"]): float(s["score"]) for s in reply.get("scores", [])}
    except (llm.LLMRequestError, llm.LLMConfigError, KeyError, TypeError, ValueError) as e:
        logger.warning("Re-ranking failed, keeping first-pass order: %s", e)
        return results[:top_k]

    for i, r in enumerate(candidates):
        r.rerank_score = scores.get(i, 0.0)
    # sorted() is stable, so equal scores keep their first-pass order.
    return sorted(candidates, key=lambda r: -r.rerank_score)[:top_k]


def _similarity(distance: float | None) -> float:
    return round(1 - distance, 4) if distance is not None else 0.0
