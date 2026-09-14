"""
LLM client (embeddings + chat) for OpenAI or Azure OpenAI. Reads env; no keys in code.

LLM_PROVIDER=openai (default) uses OPENAI_* variables.
LLM_PROVIDER=azure uses AZURE_OPENAI_* variables and your Azure deployments.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterator, List

import httpx

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")

# Inputs per embeddings request; keeps each request well under provider limits.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
MAX_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 10.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_client = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)


class LLMConfigError(RuntimeError):
    """The provider is not configured (missing key, endpoint or deployment)."""


class LLMRequestError(RuntimeError):
    """The provider request failed, after retries where retrying makes sense."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _target(kind: str) -> tuple[str, dict, dict]:
    """Return (url, headers, extra payload) for kind 'embeddings' or 'chat'."""
    path = "embeddings" if kind == "embeddings" else "chat/completions"

    if LLM_PROVIDER == "azure":
        if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY):
            raise LLMConfigError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.")
        deployment = (
            AZURE_OPENAI_EMBEDDING_DEPLOYMENT if kind == "embeddings" else AZURE_OPENAI_CHAT_DEPLOYMENT
        )
        if not deployment:
            raise LLMConfigError(f"Azure OpenAI deployment name for {kind} is not set.")
        url = (
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment}/{path}"
            f"?api-version={AZURE_OPENAI_API_VERSION}"
        )
        return url, {"api-key": AZURE_OPENAI_API_KEY}, {}

    if not OPENAI_API_KEY:
        raise LLMConfigError("OPENAI_API_KEY is not set. Add it to .env (see .env.example).")
    model = OPENAI_EMBEDDING_MODEL if kind == "embeddings" else OPENAI_CHAT_MODEL
    return f"{OPENAI_BASE_URL}/{path}", {"Authorization": f"Bearer {OPENAI_API_KEY}"}, {"model": model}


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        try:
            return min(float(response.headers.get("retry-after", "")), MAX_RETRY_DELAY_SECONDS)
        except ValueError:
            pass
    return min(2.0 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS)


def _send(kind: str, payload: dict, *, stream: bool = False) -> httpx.Response:
    """
    POST to the provider, retrying transport errors and retryable statuses.

    Returns a successful response. With stream=True the body is not read yet: the caller
    iterates it and must close it. Retries only happen before any output is produced.
    """
    url, headers, extra = _target(kind)
    body = {**extra, **payload}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = None
        try:
            if stream:
                request = _client.build_request("POST", url, json=body, headers=headers)
                response = _client.send(request, stream=True)
            else:
                response = _client.post(url, json=body, headers=headers)
        except httpx.TransportError as e:
            error = LLMRequestError(f"LLM provider unreachable ({e.__class__.__name__})")
        else:
            if response.status_code < 400:
                return response
            if stream:
                response.read()
                response.close()
            error = LLMRequestError("LLM provider request failed", response.status_code)
            if response.status_code not in RETRYABLE_STATUS:
                logger.error(
                    "LLM %s request failed: %s %s", kind, response.status_code, response.text[:500]
                )
                raise error

        if attempt == MAX_ATTEMPTS:
            logger.error("LLM %s request failed after %d attempts: %s", kind, attempt, error)
            raise error
        delay = _retry_delay(response, attempt)
        logger.warning("LLM %s attempt %d failed (%s); retrying in %.1fs", kind, attempt, error, delay)
        time.sleep(delay)

    raise AssertionError("unreachable")


def _post(kind: str, payload: dict) -> dict:
    return _send(kind, payload).json()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return one embedding vector per input string (same order)."""
    if not texts:
        return []
    vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        data = _post("embeddings", {"input": batch})
        items = sorted(data["data"], key=lambda x: x["index"])
        vectors.extend(item["embedding"] for item in items)
    return vectors


def complete(messages: list[dict], **options) -> dict:
    """
    One chat completion; returns the assistant message dict.

    options pass straight to the API, e.g. tools=[...] for tool calling or
    response_format={"type": "json_object"} for JSON output.
    """
    data = _post("chat", {"messages": messages, **options})
    return data["choices"][0]["message"]


def chat(system_prompt: str, user_message: str) -> str:
    message = complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    return message.get("content") or ""


def chat_json(system_prompt: str, user_message: str) -> dict:
    """Chat completion in JSON mode, parsed. Raises LLMRequestError if the reply is not JSON."""
    message = complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        return json.loads(message.get("content") or "")
    except json.JSONDecodeError as e:
        raise LLMRequestError("LLM returned invalid JSON") from e


def chat_stream(system_prompt: str, user_message: str) -> Iterator[str]:
    """
    Stream the answer as text deltas (Server-Sent Events from the provider).

    Connection and HTTP errors raise before the first delta, so callers can still return a
    normal error response; a failure mid-stream raises LLMRequestError from the iterator.
    """
    response = _send(
        "chat",
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": True,
        },
        stream=True,
    )
    return _iter_deltas(response)


def _iter_deltas(response: httpx.Response) -> Iterator[str]:
    try:
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue  # blank keep-alives and SSE comments
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            event = json.loads(data)
            # Azure sends a first event with empty choices (content-filter results).
            for choice in event.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    yield text
    except (httpx.TransportError, json.JSONDecodeError) as e:
        raise LLMRequestError(f"LLM stream interrupted ({e.__class__.__name__})") from e
    finally:
        response.close()
