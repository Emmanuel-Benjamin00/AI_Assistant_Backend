# AI Assistant — Backend

Django REST API for the **AI Assistant**: a retrieval-augmented generation (RAG) service. You feed it text or files (PDF, DOCX, TXT, Markdown), it splits them into chunks, embeds each chunk into a vector, and stores it in PostgreSQL with **pgvector** plus a full-text index. When you ask a question, **hybrid search** (vector + keyword, merged with reciprocal rank fusion, optionally re-ranked by the LLM) finds the best chunks, and an LLM answers **using only those chunks** — streamed token by token, with its sources.

It also has a **tool-calling agent** (the model searches, lists and reads documents on its own), the same RAG flow rebuilt with **LangChain** for comparison, and an **evaluation command** that scores retrieval and answers on a 23-question test set.

The React UI that talks to this API lives in a separate repo: **[AI_Assistant_Frontend](https://github.com/Emmanuel-Benjamin00/AI_Assistant_Frontend)**.

---

## Table of contents

- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quickstart — clone to running API](#quickstart--clone-to-running-api)
- [Environment variables](#environment-variables)
- [Verify it works](#verify-it-works)
- [API reference](#api-reference)
- [Evaluation](#evaluation)
- [Running the frontend too](#running-the-frontend-too)
- [Everyday commands](#everyday-commands)
- [Project layout](#project-layout)
- [Configuration: dev vs prod](#configuration-dev-vs-prod)
- [Security notes](#security-notes)
- [Deploying to Azure](#deploying-to-azure)
- [Troubleshooting](#troubleshooting)

---

## How it works

```
                 INGEST                                              ASK
 ┌───────────────────────────────────┐          ┌──────────────────────────────────────────┐
 │ POST /api/documents/        text  │          │ POST /api/ask/          one JSON reply    │
 │ POST /api/documents/upload/ file  │          │ POST /api/ask/stream/   Server-Sent Events│
 └─────────────────┬─────────────────┘          └─────────────────────┬────────────────────┘
                   │                                                  │
  extract text (PDF page by page, DOCX, TXT/MD)                embed the question
                   │                                                  │
  chunk each section (≤1,200 chars, overlap)          ┌───────────────┴───────────────┐
                   │                                  │                               │
  embed chunks in batches ─── OpenAI            vector search                 keyword search
                   │                          (pgvector, cosine, HNSW)   (tsvector, GIN, ts_rank_cd)
  store text + vector; Postgres                       └───────────────┬───────────────┘
  computes the tsvector column                          reciprocal rank fusion (hybrid)
                                                                      │
                                                   optional: LLM re-ranks the candidates
                                                                      │
                                                   prompt = question + top-k chunks
                                                                      │
                                                   chat model ─── OpenAI (streamed)
                                                                      │
                                                   { answer, sources[] with page + ranks }
```

Answers are grounded: the system prompt instructs the model to use **only** the retrieved context and to say it doesn't know otherwise.

Two more ways to ask:

- **`POST /api/agent/`** — a tool-calling loop. The model is given three tools (`search_documents`, `list_documents`, `read_document`) and decides which to call, up to 5 rounds, before answering. Suited to "summarize X" or "compare X and Y", which a single top-k search handles badly.
- **`POST /api/ask/langchain/`** — the `/api/ask/` flow written with LangChain (LCEL). It reuses the same retrieval, so the difference is the orchestration code only.

Production features: per-IP rate limiting, retries with backoff on provider errors, batched embeddings, input and file size limits, HNSW and GIN indexes, an ingest access key for public deployments, a health check endpoint, structured logs to stdout, and support for either OpenAI or Azure OpenAI.

---

## Tech stack

| Layer | Choice |
|---|---|
| API | Django 6 + Django REST Framework |
| Database | PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector) (runs in Docker) |
| Embeddings | `text-embedding-3-small` (1536 dimensions) via OpenAI or Azure OpenAI |
| Chat | `gpt-4o-mini` via OpenAI or Azure OpenAI |
| Keyword search | PostgreSQL full-text search (`tsvector` generated column + GIN index) |
| File parsing | `pypdf` (PDF), `python-docx` (DOCX) |
| HTTP client | `httpx` for the core flow (no OpenAI SDK dependency) |
| Comparison pipeline | LangChain (`langchain-core`, `langchain-openai`) |
| Serving (prod) | gunicorn (threaded workers, for streaming) + WhiteNoise |

---

## Prerequisites

Install these before you start:

- **Python 3.12+** — `python3 --version`
- **Docker** + Docker Compose v2 — `docker compose version`
- **Git**
- An **OpenAI API key** — create one at <https://platform.openai.com/api-keys>

> Calls to `/api/documents/` and `/api/ask/` spend real OpenAI credits. They are cheap (fractions of a cent per request with the default models), but they are not free.

Linux/macOS/WSL is assumed. Windows works too — Windows-specific commands are noted where they differ.

---

## Quickstart — clone to running API

Run these **in order**, from the repo root (the folder containing `manage.py`).

### 1. Clone

```bash
git clone https://github.com/Emmanuel-Benjamin00/AI_Assistant_Backend.git
cd AI_Assistant_Backend
```

### 2. Start PostgreSQL + pgvector

```bash
docker compose up -d
```

This starts a container named `aiassistant-db`. Confirm it is healthy:

```bash
docker compose ps
```

> **Port note:** Postgres is published on host port **5433**, not the usual 5432, so it will not clash with a Postgres you may already run locally. Inside the container it is still 5432.

### 3. Create a virtualenv and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.venv/` **inside this clone**, next to `manage.py`. It is gitignored and must never be committed.

### 4. Add your OpenAI key

```bash
cp .env.example .env
```

Open `.env` and set your key:

```dotenv
OPENAI_API_KEY=sk-your-real-key-here
```

`.env` is gitignored — it will never be committed. Nothing else in `.env` is required; the other variables have working defaults.

### 5. Apply migrations

```bash
python manage.py migrate
```

This enables the `vector` extension in Postgres and creates the `Document` and `Chunk` tables.

### 6. Run the server

```bash
python manage.py runserver
```

The API is now live at **<http://127.0.0.1:8000/>**.

That's it — six steps, no other configuration needed.

---

## Environment variables

Copy [`.env.example`](.env.example) to `.env`. Only the first is required.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | No | `openai` | `openai` or `azure`. For `azure`, set the `AZURE_OPENAI_*` variables listed in [`.env.example`](.env.example) instead of the OpenAI ones. |
| `OPENAI_API_KEY` | **Yes** (for `openai`) | — | Authenticates embedding and chat calls. Without it, both endpoints return `503`. |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Must produce **1536-dim** vectors to match the DB column. |
| `OPENAI_CHAT_MODEL` | No | `gpt-4o-mini` | Model that writes the final answer. |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | Override only for a proxy or compatible endpoint. |
| `INGEST_API_KEY` | No | — | When set, `POST /api/documents/` and `/api/documents/upload/` require the `X-Ingest-Key` header to match. |
| `RETRIEVAL_MODE` | No | `hybrid` | Default search when a request doesn't send `mode`: `hybrid` or `vector`. |
| `RERANK_CANDIDATES` | No | `12` | How many chunks the LLM re-ranker scores when `rerank` is true. |
| `AGENT_MAX_STEPS` | No | `5` | Tool-calling rounds before the agent must answer. |
| `THROTTLE_ASK_RATE` / `THROTTLE_INGEST_RATE` / `THROTTLE_AGENT_RATE` | No | `30/hour` / `10/hour` / `15/hour` | Per-IP rate limits. |
| `LLM_TIMEOUT_SECONDS`, `EMBED_BATCH_SIZE`, `LOG_LEVEL` | No | `30`, `96`, `INFO` | Tuning. |

Production-only variables (`DJANGO_ENV`, `SECRET_KEY`, `DATABASE_URL`, `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `CORS_ALLOWED_ORIGINS`) are described under [dev vs prod](#configuration-dev-vs-prod). **You do not need any of them for local development.**

> If you change the embedding model to one with a different vector size, you must also change `dimensions=1536` in [`rag_app/models.py`](rag_app/models.py) and create a new migration. Existing embeddings would need to be regenerated.

---

## Verify it works

With the server running, open a second terminal.

**1. Ingest a document:**

```bash
curl -X POST http://127.0.0.1:8000/api/documents/ \
  -H 'Content-Type: application/json' \
  -d '{
        "title": "Company facts",
        "text": "Our support hours are 9am to 5pm on weekdays.\n\nWe offer a 30-day refund on all plans."
      }'
```

Expected:

```json
{"document_id": 1, "chunks_created": 1, "source": "text"}
```

**Or upload a file** (PDF, DOCX, TXT, Markdown):

```bash
curl -X POST http://127.0.0.1:8000/api/documents/upload/ -F file=@handbook.pdf
```

**2. Ask a question about it:**

```bash
curl -X POST http://127.0.0.1:8000/api/ask/ \
  -H 'Content-Type: application/json' \
  -d '{"question": "When can I get support?", "top_k": 3}'
```

Expected — an answer grounded in your text, with the chunks it used:

```json
{
  "answer": "Support is available from 9am to 5pm on weekdays [1].",
  "mode": "hybrid",
  "sources": [{"ref": 1, "chunk_id": 1, "document_title": "Company facts", "page": null, "similarity": 0.61, "vector_rank": 1, "keyword_rank": 1, "...": "..."}]
}
```

**3. Watch an answer stream** (`-N` turns off curl's buffering):

```bash
curl -N -X POST http://127.0.0.1:8000/api/ask/stream/ \
  -H 'Content-Type: application/json' \
  -d '{"question": "When can I get support?"}'
```

Or load the bundled sample documents instead: `python manage.py seed_demo`.

If you get both responses, your full stack — Django, Postgres, pgvector, and OpenAI — is working.

---

## API reference

Base URL in local development: `http://127.0.0.1:8000`

There are no user accounts. Endpoints are rate-limited per client IP, and ingest is locked in production unless the caller sends `X-Ingest-Key` (see `INGEST_API_KEY`).

### `GET /api/documents/` — list indexed documents

Returns up to 100 documents, newest first: `[{"id", "title", "source", "filename", "chunks", "created_at"}]`. `source` is `text`, `pdf`, `docx` or `plain`.

### `POST /api/documents/` — ingest text

Splits `text` into chunks of up to 1,200 characters (paragraphs kept together; long paragraphs split on sentences with 200 characters of overlap), embeds them in batches, and stores them.

| Field | Type | Required | Notes |
|---|---|---|---|
| `title` | string | yes | Max 255 characters |
| `text` | string | yes | Max 100,000 characters |

**Response `201`:** `{"document_id": 1, "chunks_created": 2, "source": "text"}`

Ingest is atomic — if embedding fails partway, no partial document is saved.

### `POST /api/documents/upload/` — ingest a file

`multipart/form-data` with:

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | `.pdf`, `.docx`, `.txt`, `.md`; max 10 MB; max 300,000 characters of text |
| `title` | string | no | Defaults to the file name without its extension |

PDFs are read page by page and every chunk stores its `page`, which `/api/ask/` returns with the sources. The file's first bytes must match its extension. Scanned (image-only) PDFs, password-protected PDFs and non-UTF-8 text files are rejected with a `400` that explains why. Same response as text ingest, with `source` set to the file type.

### `POST /api/ask/` — ask a question

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | Max 2,000 characters |
| `top_k` | integer | no | Chunks to retrieve. Default `5`, range `1`–`10` |
| `mode` | string | no | `hybrid` (default) or `vector` |
| `rerank` | boolean | no | Default `false`. The LLM scores up to 12 candidates and keeps the best `top_k` (one extra model call). |

**Response `200`**

```json
{
  "answer": "... [1]",
  "mode": "hybrid",
  "sources": [{
    "ref": 1, "chunk_id": 7, "document_id": 2, "document_title": "...", "page": 3,
    "similarity": 0.61, "vector_rank": 2, "keyword_rank": 1, "rrf_score": 0.03227, "rerank_score": null,
    "text": "..."
  }]
}
```

`ref` matches the `[n]` citations in the answer. `vector_rank` / `keyword_rank` show where each search placed the chunk (`null` if that search didn't return it). If nothing has been ingested yet, you get a `200` with an empty `sources` array and a message telling you to ingest first.

### `POST /api/ask/stream/` — ask, streamed

Same request as `/api/ask/`. The response is `text/event-stream`:

```
event: sources
data: {"mode": "hybrid", "sources": [...]}

event: token
data: {"text": "Support is "}

event: token
data: {"text": "9 to 5 [1]."}

event: done
data: {}
```

If the provider fails mid-answer you get `event: error` with `{"detail": "..."}` instead of `done`. Errors before the first token (validation, rate limit, provider down) are normal JSON responses with the usual status codes. Browsers can't use `EventSource` for POST, so the frontend reads the stream with `fetch`.

### `POST /api/agent/` — tool-calling agent

**Request:** `{"question": "Compare the EDI primer and the NEMT document"}`

**Response `200`**

```json
{
  "answer": "...",
  "steps": [
    {"tool": "list_documents", "arguments": {}, "summary": "5 documents"},
    {"tool": "read_document", "arguments": {"document_id": 2}, "summary": "read 'US Healthcare Claims and EDI Primer'"}
  ],
  "sources": [{"ref": 1, "chunk_id": 4, "document_title": "...", "page": null, "similarity": 0.5, "text": "..."}],
  "documents_read": [{"document_id": 2, "document_title": "US Healthcare Claims and EDI Primer"}]
}
```

Tool errors (unknown tool, bad arguments, missing document) are sent back to the model as tool results so it can correct itself. After 5 tool rounds the model is forced to answer. Rate limit: `THROTTLE_AGENT_RATE`.

### `POST /api/ask/langchain/` — the same flow with LangChain

Same request and response as `/api/ask/`, plus `"pipeline": "langchain"`. See [`rag_app/services/langchain_rag.py`](rag_app/services/langchain_rag.py).

### Error responses

| Status | Meaning |
|---|---|
| `400` | Invalid request body (missing field, too long, `top_k` out of range), or a file that can't be indexed |
| `403` | Ingest is locked and `X-Ingest-Key` is missing or wrong |
| `405` | Wrong method — the ask and agent endpoints are **POST only** |
| `413` | Uploaded file is larger than 10 MB |
| `429` | Rate limit exceeded for this IP |
| `502` | The AI provider returned an error (after retries) |
| `503` | The AI provider is not configured, or is rate-limiting us |

### `GET /api/health/` — health check

`200 {"status": "ok"}` when the app can reach the database, otherwise `503`. Used by Azure App Service health checks.

### Django admin

Available at <http://127.0.0.1:8000/admin/>. Create a login first:

```bash
python manage.py createsuperuser
```

---

## Evaluation

`python manage.py evaluate_rag` scores the system on [`rag_app/eval/dataset.json`](rag_app/eval/dataset.json): 20 questions answered by the demo documents plus 3 that aren't (the assistant should say it doesn't know). Each question lists the evidence phrase that answers it and the facts a correct answer must contain.

| Metric | What it measures | How |
|---|---|---|
| `hit@k` | Share of answerable questions with a relevant chunk in the top k | A retrieved chunk contains the evidence phrase |
| `mrr` | How high the first relevant chunk ranks (1.0 = always first) | Mean of 1 / rank |
| `fact_recall` | Share of expected facts present in the answer | String match |
| `faithfulness` | Every claim in the answer is supported by the retrieved context | LLM-as-judge, JSON verdict |
| `abstention` | Unanswerable questions answered with "I don't know" | Phrase match |

```bash
python manage.py seed_demo                          # index the documents the questions are about
python manage.py evaluate_rag --retrieval-only      # hit@k + MRR only: 23 embeddings, well under a cent
python manage.py evaluate_rag                       # all metrics for vector, hybrid, hybrid+rerank
python manage.py evaluate_rag --configs hybrid,langchain --top-k 3 --output eval-report.json
```

It prints one row per configuration so you can compare them:

```
config                    hit@3            mrr    fact_recall   faithfulness     abstention avg_latency_ms
vector                      ...            ...            ...            ...            ...            ...
hybrid                      ...
```

The report (`--output`) keeps every answer, judge note and retrieved chunk, so you can inspect each miss. A test ([`test_evaluation.py`](rag_app/tests/test_evaluation.py)) fails if a demo document edit breaks an evidence phrase.

---

## Running the frontend too

The API is usable on its own, but the React UI is the intended way to use it.

```bash
# in a separate folder, alongside this repo
git clone https://github.com/Emmanuel-Benjamin00/AI_Assistant_Frontend.git
cd AI_Assistant_Frontend
npm install
npm run dev
```

The UI opens at <http://localhost:5173/> and proxies `/api` to `http://127.0.0.1:8000` — so **keep this backend running** in its own terminal. No frontend configuration is needed for local development.

---

## Everyday commands

Activate the virtualenv first (`source .venv/bin/activate`).

| Task | Command |
|---|---|
| Start the database | `docker compose up -d` |
| Run the API | `python manage.py runserver` |
| Stop the API | `Ctrl + C` |
| Stop the database | `docker compose down` |
| Apply migrations | `python manage.py migrate` |
| Create migrations after model changes | `python manage.py makemigrations` |
| Create an admin user | `python manage.py createsuperuser` |
| Open a Django shell | `python manage.py shell` |
| Run tests | `python manage.py test` (needs the database running) |
| Index the sample documents | `python manage.py seed_demo` |
| Evaluate retrieval and answers | `python manage.py evaluate_rag` (see [Evaluation](#evaluation)) |
| Check for config problems | `python manage.py check` |

### Database safety

- `docker compose down` stops the container and **keeps** your data (it lives in the `pgdata` volume).
- `docker compose down -v` **deletes the volume and all your data**. Only use it when you want a clean slate.

### Connecting with a GUI (pgAdmin, DBeaver, TablePlus)

| Setting | Value |
|---|---|
| Host | `127.0.0.1` (not the container name) |
| Port | `5433` |
| Database | `aiassistant` |
| User | `aiassistant` |
| Password | `aiassistant` |

These credentials come from [`docker-compose.yml`](docker-compose.yml) and are **local-development only**.

---

## Project layout

```
AI_Assistant_Backend/
├── AI_Assistant/            # Django project package
│   ├── settings.py          # dev/prod environment switch, throttle rates
│   ├── urls.py              # routes /admin/ and /api/
│   └── wsgi.py, asgi.py
├── rag_app/                 # the RAG application
│   ├── models.py            # Document (source, filename), Chunk (vector + generated tsvector)
│   ├── views.py             # ingest, upload, ask, ask/stream, agent, ask/langchain, health
│   ├── serializers.py       # request validation and size limits
│   ├── chunking.py          # size-bounded chunks with overlap
│   ├── throttling.py        # per-IP rate limit, ingest key permission
│   ├── services/
│   │   ├── llm.py           # OpenAI / Azure OpenAI client: embeddings, chat, JSON mode, streaming, tools
│   │   ├── extraction.py    # PDF / DOCX / TXT / MD -> text sections (with page numbers)
│   │   ├── indexing.py      # chunk + embed + save, shared by every ingest path
│   │   ├── retrieval.py     # vector, keyword, hybrid (RRF) search and LLM re-ranking
│   │   ├── rag.py           # grounded prompt, answer, streamed answer, source format
│   │   ├── agent.py         # tool-calling loop and its three tools
│   │   ├── langchain_rag.py # the RAG flow in LangChain LCEL
│   │   └── evaluation.py    # hit@k, MRR, fact recall, abstention, LLM judge
│   ├── eval/dataset.json    # 23 evaluation questions
│   ├── management/commands/
│   │   ├── seed_demo.py     # indexes demo_docs/
│   │   └── evaluate_rag.py  # compares search configurations
│   ├── demo_docs/           # sample documents (also the evaluation corpus)
│   ├── tests/               # one file per feature
│   └── migrations/
│       ├── 0001_enable_pgvector.py   # CREATE EXTENSION vector
│       ├── 0002_initial.py
│       ├── 0003_chunk_embedding_hnsw.py
│       └── 0004_document_source_chunk_search_vector.py  # tsvector column + GIN index
├── docker-compose.yml       # Postgres 16 + pgvector on host port 5433
├── requirements.txt         # pinned dependencies
├── startup.sh               # production entrypoint (migrate, cache table, collectstatic, seed, gunicorn)
├── DEPLOY_AZURE.md          # step-by-step Azure deployment
├── .env.example             # template — copy to .env
└── manage.py
```

Everything needed to run the stack is in this repo. `git clone` plus the [quickstart](#quickstart--clone-to-running-api) is sufficient.

---

## Configuration: dev vs prod

[`settings.py`](AI_Assistant/settings.py) switches on a single environment variable:

```python
DJANGO_ENV = os.getenv("DJANGO_ENV", "dev").lower()
```

**Local development** is the default — you set nothing. You get `DEBUG=True`, permissive CORS and hosts, and the local Docker database.

**Production** requires `DJANGO_ENV=prod` plus these variables supplied by the host (Azure App Service settings, or equivalent):

| Variable | Example |
|---|---|
| `SECRET_KEY` | a long random string — **required**, no fallback |
| `DATABASE_URL` | `postgres://user:pass@host:5432/db` (SSL enforced) |
| `ALLOWED_HOSTS` | `myapp.azurewebsites.net` |
| `CSRF_TRUSTED_ORIGINS` | `https://myapp.azurewebsites.net` |
| `CORS_ALLOWED_ORIGINS` | `https://myapp.azurestaticapps.net` |
| `OPENAI_API_KEY` | your key (or the `AZURE_OPENAI_*` settings) |
| `INGEST_API_KEY` | random string; without it, ingest is disabled in prod |
| `SEED_DEMO` | `true` to index the sample documents on startup |

In production, `DEBUG` is off, the API renders JSON only, CORS is restricted to the listed origins, secure cookies and HSTS are on, rate-limit counters are shared across gunicorn workers via a database cache table, static files are served by WhiteNoise, and [`startup.sh`](startup.sh) runs migrations, creates the cache table, collects static files, optionally seeds demo documents, and launches gunicorn with threaded workers (`GUNICORN_THREADS`, default 4), so a streaming answer doesn't block a whole worker.

---

## Deploying to Azure

See **[DEPLOY_AZURE.md](DEPLOY_AZURE.md)** for step-by-step portal instructions: PostgreSQL Flexible Server with pgvector, App Service, Static Web Apps, and OpenAI or Azure OpenAI.

---

## Security notes

- **Never commit `.env`.** It is gitignored, along with `.env.*` and `*.backup`. Only `.env.example` — which contains no values — is committed.
- **No API keys appear in source.** [`rag_app/services/llm.py`](rag_app/services/llm.py) reads every credential from the environment.
- **The dev `SECRET_KEY` in `settings.py` is intentionally public.** It is prefixed `django-insecure-` and is only ever used when `DJANGO_ENV` is not `prod`. Production reads `SECRET_KEY` from the environment and **fails loudly** if it is missing, so the dev key can never leak into a deployment.
- **The Postgres password in `docker-compose.yml` is local-only** and never reachable from outside your machine.
- **The API has no user accounts.** Anyone who can reach `/api/ask/` can spend AI credits, so it is rate-limited per IP and ingest is locked in production. Also set a spending limit with your AI provider.
- **If you ever commit a real key by accident:** rotate it immediately at the provider. Removing the commit is not enough — assume any key that reached GitHub is compromised.

---

## Troubleshooting

**`OPENAI_API_KEY is not set` (503)**
`.env` is missing or the key line is empty. Confirm `.env` sits next to `manage.py` and contains `OPENAI_API_KEY=sk-...`, then restart `runserver` — env vars load at startup.

**`connection refused` on port 5433**
The database container is not running. `docker compose up -d`, then check with `docker compose ps`.

**`permission denied` on `/var/run/docker.sock` (Linux)**
Your user is not in the `docker` group yet in this shell. Run `newgrp docker` and retry in the same terminal, or restart your terminal/IDE.

**`port is already allocated` on 5433**
Something else holds the port. Change the host side in `docker-compose.yml` (e.g. `"5434:5432"`) and update the port in the dev `DATABASES` block in `settings.py` to match.

**`type "vector" does not exist`**
Migrations have not run, or the container is not the pgvector image. Run `python manage.py migrate` — `0001_enable_pgvector` creates the extension. Confirm `docker-compose.yml` uses `pgvector/pgvector:pg16`.

**`ModuleNotFoundError` on any package**
The virtualenv is not active. Run `source .venv/bin/activate`, then `pip install -r requirements.txt`.

**`405 Method Not Allowed`**
You made a GET request to `/api/ask/`, which accepts **POST only**.

**Streamed answers arrive all at once**
Something between the browser and Django is buffering the response. The API sends `X-Accel-Buffering: no` for nginx. With `runserver` or gunicorn directly it streams; check any proxy you added. With curl, pass `-N`.

**"No text could be extracted" for a PDF**
The PDF is a scan (images of text). Run it through OCR first, or copy the text into the paste box.

**Seeded demo documents are out of date**
`seed_demo` skips titles that already exist. Delete the old document (Django admin, or `Document.objects.filter(title=...).delete()` in `python manage.py shell`) and run it again.

**Answers ignore what I ingested**
Retrieval only searches chunks that have embeddings. If a document was ingested while the API key was invalid, its chunks may have none. Re-ingest it, and raise `top_k` if the relevant text is not surfacing.

---

## Roadmap

Planning documents live in **[AI-Assistant_Plan-and-docs](https://github.com/Emmanuel-Benjamin00/AI-Assistant_Plan-and-docs)** (`HIGH_LEVEL_PLAN.md`, `BUILD_PLAN.md`).

Done: RAG, file uploads, hybrid search with re-ranking, streaming answers, a tool-calling agent, a LangChain comparison and an evaluation suite. Next: user accounts with per-user documents, conversation memory, and OCR for scanned PDFs.
