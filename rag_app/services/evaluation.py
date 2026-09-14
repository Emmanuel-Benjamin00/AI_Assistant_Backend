"""
RAG evaluation metrics. The scoring functions are pure so they can be unit-tested; only
judge_faithfulness calls the LLM.

Retrieval (no chat model needed, costs only one embedding per question):
  hit@k   share of answerable questions with a relevant chunk in the top k
  MRR     mean of 1/rank of the first relevant chunk (0 when none); rewards ranking it first

Answers:
  fact recall    share of expected facts that appear in the answer (string match)
  faithfulness   LLM-as-judge: is every claim in the answer supported by the retrieved context?
  abstention     for unanswerable questions, did the assistant say it does not know?
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from . import llm

DATASET_PATH = Path(__file__).resolve().parents[1] / "eval" / "dataset.json"

# Phrases that mean "the context does not have the answer" (matched after _normalize).
ABSTAIN_PHRASES = (
    "don't know",
    "do not know",
    "not in the context",
    "not in the provided",
    "not mentioned",
    "no information",
    "does not contain",
    "doesn't contain",
    "does not mention",
    "doesn't mention",
    "does not provide",
    "doesn't provide",
    "cannot find",
    "can't find",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "unable to find",
)


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    evidence: list[str]
    facts: list[list[str]]
    answerable: bool
    expected_document: str | None = None


@dataclass
class EvalRow:
    question: EvalQuestion
    retrieved_texts: list[str]
    hit_rank: int | None
    answer: str | None = None
    fact_recall: float | None = None
    abstained: bool | None = None
    faithful: bool | None = None
    judge_notes: str = ""
    latency_ms: float = 0.0


def load_dataset(path: Path = DATASET_PATH) -> list[EvalQuestion]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = []
    for item in raw["questions"]:
        q = EvalQuestion(
            id=item["id"],
            question=item["question"],
            evidence=list(item.get("evidence", [])),
            facts=[list(group) for group in item.get("facts", [])],
            answerable=bool(item.get("answerable", True)),
            expected_document=item.get("expected_document"),
        )
        if q.answerable and not (q.evidence and q.facts):
            raise ValueError(f"answerable question {q.id} needs evidence and facts")
        questions.append(q)
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique")
    return questions


def first_hit_rank(retrieved_texts: list[str], evidence: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk containing any evidence phrase."""
    needles = [_normalize(e) for e in evidence]
    for rank, text in enumerate(retrieved_texts, start=1):
        haystack = _normalize(text)
        if any(n in haystack for n in needles):
            return rank
    return None


def fact_recall(answer: str, facts: list[list[str]]) -> float:
    if not facts:
        return 1.0
    haystack = _normalize(answer)
    found = sum(any(_normalize(alt) in haystack for alt in group) for group in facts)
    return found / len(facts)


def is_abstention(answer: str) -> bool:
    text = _normalize(answer or "")
    return any(phrase in text for phrase in ABSTAIN_PHRASES)


JUDGE_PROMPT = (
    "You check a RAG system's answer for faithfulness. An answer is faithful when every factual "
    "claim in it is supported by the context. Saying it does not know is faithful. "
    "Citations like [1] are not claims. Ignore whether the answer is complete. "
    'Reply with JSON only: {"faithful": true|false, "unsupported_claims": ["..."]}'
)


def judge_faithfulness(question: str, context: str, answer: str) -> tuple[bool, str]:
    reply = llm.chat_json(
        JUDGE_PROMPT,
        f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}",
    )
    faithful = reply.get("faithful")
    if not isinstance(faithful, bool):
        raise llm.LLMRequestError("judge reply has no boolean 'faithful'")
    claims = reply.get("unsupported_claims") or []
    return faithful, "; ".join(str(c) for c in claims)


def summarize(rows: list[EvalRow], top_k: int) -> dict:
    answerable = [r for r in rows if r.question.answerable]
    unanswerable = [r for r in rows if not r.question.answerable]

    def avg(values):
        values = [v for v in values if v is not None]
        return round(mean(values), 4) if values else None

    return {
        "questions": len(rows),
        f"hit@{top_k}": avg([1.0 if r.hit_rank else 0.0 for r in answerable]),
        "mrr": avg([1.0 / r.hit_rank if r.hit_rank else 0.0 for r in answerable]),
        "fact_recall": avg([r.fact_recall for r in answerable]),
        "faithfulness": avg([None if r.faithful is None else float(r.faithful) for r in rows]),
        "abstention": avg([None if r.abstained is None else float(r.abstained) for r in unanswerable]),
        "avg_latency_ms": avg([r.latency_ms for r in rows]),
    }


def _normalize(text: str) -> str:
    # Case-, whitespace- and dash-insensitive matching ("UB-04" vs "UB‑04").
    text = text.lower().replace("\u2011", "-").replace("\u2013", "-").replace("\u2019", "'")
    return re.sub(r"\s+", " ", text)
