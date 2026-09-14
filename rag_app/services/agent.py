"""
Agent mode: the chat model decides which tools to call, the server runs them, and the loop
repeats until the model answers or the step limit is reached.

    user question
      -> model (with tool definitions)
      -> tool_calls?  yes: run each tool, append results, call the model again
                      no:  that reply is the final answer
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from django.db.models import Count

from ..models import Chunk, Document
from . import llm, retrieval

logger = logging.getLogger(__name__)

# Model rounds that may call tools. A limit stops a confused model from looping (and spending) forever.
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))
# Caps what one read_document call puts into the prompt (~3k tokens).
READ_DOCUMENT_MAX_CHARS = 12_000
SEARCH_MAX_RESULTS = 8

SYSTEM_PROMPT = (
    "You are a document assistant with tools. Use the tools to gather facts before answering, "
    "and answer ONLY from tool results, never from general knowledge. "
    "Use search_documents for specific questions. Use list_documents to find document ids, then "
    "read_document to summarize or compare whole documents. "
    "Cite search results with their ref numbers like [1]. "
    "If the tools do not give you the answer, say you don't know."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Hybrid (keyword + semantic) search over all indexed document chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of chunks to return (1-{SEARCH_MAX_RESULTS}).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the indexed documents with their ids, titles and chunk counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read the full text of one document by id, for summaries and comparisons.",
            "parameters": {
                "type": "object",
                "properties": {"document_id": {"type": "integer"}},
                "required": ["document_id"],
            },
        },
    },
]


class ToolError(ValueError):
    """Bad tool arguments; reported back to the model so it can correct itself."""


@dataclass
class AgentRun:
    question: str
    messages: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    documents_read: list[dict] = field(default_factory=list)


def run_agent(question: str) -> dict:
    """Returns {"answer", "steps", "sources", "documents_read"}. Raises llm errors."""
    run = AgentRun(
        question=question,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    for _ in range(MAX_STEPS):
        message = llm.complete(run.messages, tools=TOOLS, temperature=0)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return _result(run, message.get("content") or "")

        # The assistant turn with its tool_calls must precede the tool results.
        run.messages.append(
            {"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls}
        )
        for call in tool_calls:
            output = _run_tool_call(run, call)
            run.messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(output)}
            )

    # Out of steps: force a final text answer from what was gathered so far.
    logger.info("Agent hit the %d-step limit; forcing an answer", MAX_STEPS)
    message = llm.complete(run.messages, tools=TOOLS, tool_choice="none", temperature=0)
    return _result(run, message.get("content") or "")


def _run_tool_call(run: AgentRun, call: dict) -> dict:
    name = call.get("function", {}).get("name", "")
    raw_args = call.get("function", {}).get("arguments") or "{}"
    step = {"tool": name, "arguments": raw_args}
    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        handler = _HANDLERS.get(name)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        output = handler(run, **args)
        step["arguments"] = args
        step["summary"] = output.pop("_summary")
    except (ToolError, json.JSONDecodeError, TypeError) as e:
        # TypeError covers unexpected or missing keyword arguments.
        output = {"error": str(e)}
        step["summary"] = f"error: {e}"
    run.steps.append(step)
    return output


def _search_documents(run: AgentRun, query: str, top_k: int = 5) -> dict:
    query = str(query).strip()
    if not query:
        raise ToolError("query must not be empty")
    try:
        top_k = max(1, min(int(top_k), SEARCH_MAX_RESULTS))
    except (TypeError, ValueError) as e:
        raise ToolError("top_k must be an integer") from e

    results = retrieval.retrieve(query, top_k, mode=retrieval.HYBRID)
    items = []
    for r in results:
        # Refs are numbered across the whole run so citations stay unique between searches.
        existing = next((s for s in run.sources if s["chunk_id"] == r.chunk.id), None)
        if existing is None:
            existing = {
                "ref": len(run.sources) + 1,
                "chunk_id": r.chunk.id,
                "document_id": r.chunk.document_id,
                "document_title": r.chunk.document.title,
                "page": r.chunk.metadata.get("page"),
                "similarity": r.similarity,
                "text": r.chunk.text,
            }
            run.sources.append(existing)
        items.append(
            {
                "ref": existing["ref"],
                "document": existing["document_title"],
                "page": existing["page"],
                "text": r.chunk.text,
            }
        )
    return {"results": items, "_summary": f"{len(items)} chunks for {query!r}"}


def _list_documents(run: AgentRun) -> dict:
    docs = Document.objects.annotate(chunk_count=Count("chunks")).order_by("id")[:100]
    items = [{"id": d.id, "title": d.title, "chunks": d.chunk_count, "source": d.source} for d in docs]
    return {"documents": items, "_summary": f"{len(items)} documents"}


def _read_document(run: AgentRun, document_id: int) -> dict:
    try:
        doc = Document.objects.get(pk=int(document_id))
    except (Document.DoesNotExist, TypeError, ValueError) as e:
        raise ToolError(f"no document with id {document_id!r}; call list_documents first") from e

    texts, used, truncated = [], 0, False
    for text in Chunk.objects.filter(document=doc).order_by("id").values_list("text", flat=True):
        if used + len(text) > READ_DOCUMENT_MAX_CHARS:
            texts.append(text[: READ_DOCUMENT_MAX_CHARS - used])
            truncated = True
            break
        texts.append(text)
        used += len(text)

    if not any(d["document_id"] == doc.id for d in run.documents_read):
        run.documents_read.append({"document_id": doc.id, "document_title": doc.title})
    return {
        "document_id": doc.id,
        "title": doc.title,
        "text": "\n\n".join(texts),
        "truncated": truncated,
        "_summary": f"read {doc.title!r}" + (" (truncated)" if truncated else ""),
    }


_HANDLERS = {
    "search_documents": _search_documents,
    "list_documents": _list_documents,
    "read_document": _read_document,
}


def _result(run: AgentRun, answer: str) -> dict:
    return {
        "answer": answer,
        "steps": run.steps,
        "sources": run.sources,
        "documents_read": run.documents_read,
    }
