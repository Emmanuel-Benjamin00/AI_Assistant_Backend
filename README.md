# AI Assistant — Backend

Django REST API for the **AI Assistant**: a retrieval-augmented generation (RAG) service. You feed it text, it splits that text into chunks, embeds each chunk into a vector, and stores it in PostgreSQL with **pgvector**. When you ask a question, it embeds the question, finds the nearest chunks by cosine distance, and asks an LLM to answer **using only those chunks** — returning the answer plus its sources.

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
- [Running the frontend too](#running-the-frontend-too)
- [Everyday commands](#everyday-commands)
- [Project layout](#project-layout)
- [Configuration: dev vs prod](#configuration-dev-vs-prod)
- [Security notes](#security-notes)
- [Troubleshooting](#troubleshooting)

---

## How it works

```
                    INGEST                                    ASK
    ┌──────────────────────────────┐        ┌──────────────────────────────────┐
    │ POST /api/documents/         │        │ POST /api/ask/                   │
    │   { title, text }            │        │   { question, top_k }            │
    └──────────────┬───────────────┘        └────────────────┬─────────────────┘
                   │                                         │
      split on blank lines                          embed the question
                   │                                         │
        embed each chunk  ──── OpenAI ────  embeddings API    │
                   │                                         │
     store text + vector                     nearest-neighbour search
       in Postgres/pgvector  ◄──────────────  (cosine distance, top_k)
                                                             │
                                            build prompt from top chunks
                                                             │
                                              chat model ──── OpenAI
                                                             │
                                              { answer, sources[] }
```

Answers are grounded: the system prompt instructs the model to use **only** the retrieved context and to say it doesn't know otherwise.

---

## Tech stack

| Layer | Choice |
|---|---|
| API | Django 6 + Django REST Framework |
| Database | PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector) (runs in Docker) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dimensions) |
| Chat | OpenAI `gpt-4o-mini` |
| HTTP client | `httpx` (no OpenAI SDK dependency) |
| Serving (prod) | gunicorn + WhiteNoise |

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
| `OPENAI_API_KEY` | **Yes** | — | Authenticates embedding and chat calls. Without it, both endpoints return `503`. |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Must produce **1536-dim** vectors to match the DB column. |
| `OPENAI_CHAT_MODEL` | No | `gpt-4o-mini` | Model that writes the final answer. |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | Override only for a proxy or compatible endpoint. |

Production-only variables (`DJANGO_ENV`, `SECRET_KEY`, `DATABASE_URL`, `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `CORS_ALLOWED_ORIGINS`) are described under [dev vs prod](#configuration-dev-vs-prod). **You do not need any of them for local development.**

> If you change the embedding model to one with a different vector size, you must also change `dimensions=1536` in [`rag_app/models.py`](rag_app/models.py) and create a new migration. Existing embeddings would need to be regenerated.

---

## Verify it works

With the server running, open a second terminal.

**1. Ingest a document.** Blank lines separate chunks:

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
{"document_id": 1, "chunks_created": 2}
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
  "sources": [{"chunk_id": 1, "text": "Our support hours are 9am to 5pm on weekdays."}]
}
```

If you get both responses, your full stack — Django, Postgres, pgvector, and OpenAI — is working.

---

## API reference

Base URL in local development: `http://127.0.0.1:8000`

There is **no authentication** on these endpoints. That is fine for local development; add auth before exposing this API publicly.

### `POST /api/documents/` — ingest text

Splits `text` on blank lines, embeds each chunk, and stores them.

**Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `title` | string | yes | Max 255 characters |
| `text` | string | yes | Blank lines (`\n\n`) mark chunk boundaries |

**Response `201`**

```json
{"document_id": 1, "chunks_created": 2}
```

Ingest is atomic — if embedding fails partway, no partial document is saved.

### `POST /api/ask/` — ask a question

**Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | Natural-language question |
| `top_k` | integer | no | Chunks to retrieve. Default `5`, range `1`–`20` |

**Response `200`**

```json
{
  "answer": "...",
  "sources": [{"chunk_id": 1, "text": "..."}]
}
```

If nothing has been ingested yet, you get a `200` with an empty `sources` array and a message telling you to ingest first.

### Error responses

| Status | Meaning |
|---|---|
| `400` | Invalid request body (missing field, `top_k` out of range) |
| `405` | Wrong method — both endpoints are **POST only** |
| `502` | OpenAI returned an error |
| `503` | `OPENAI_API_KEY` is not set |

### Django admin

Available at <http://127.0.0.1:8000/admin/>. Create a login first:

```bash
python manage.py createsuperuser
```

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
| Run tests | `python manage.py test` |
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
│   ├── settings.py          # dev/prod environment switch lives here
│   ├── urls.py              # routes /admin/ and /api/
│   └── wsgi.py, asgi.py
├── rag_app/                 # the RAG application
│   ├── models.py            # Document, Chunk (VectorField, 1536 dims)
│   ├── views.py             # DocumentIngestView, AskView
│   ├── serializers.py       # request validation
│   ├── chunking.py          # splits text on blank lines
│   ├── services/llm.py      # OpenAI embeddings + chat (reads env, no keys in code)
│   └── migrations/
│       ├── 0001_enable_pgvector.py   # CREATE EXTENSION vector
│       └── 0002_initial.py
├── docker-compose.yml       # Postgres 16 + pgvector on host port 5433
├── requirements.txt         # pinned dependencies
├── startup.sh               # production entrypoint (migrate, collectstatic, gunicorn)
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
| `OPENAI_API_KEY` | your key |

In production, `DEBUG` is off, CORS is restricted to the listed origins, static files are served by WhiteNoise, and [`startup.sh`](startup.sh) runs migrations, collects static files, and launches gunicorn.

---

## Security notes

- **Never commit `.env`.** It is gitignored, along with `.env.*` and `*.backup`. Only `.env.example` — which contains no values — is committed.
- **No API keys appear in source.** [`rag_app/services/llm.py`](rag_app/services/llm.py) reads every credential from the environment.
- **The dev `SECRET_KEY` in `settings.py` is intentionally public.** It is prefixed `django-insecure-` and is only ever used when `DJANGO_ENV` is not `prod`. Production reads `SECRET_KEY` from the environment and **fails loudly** if it is missing, so the dev key can never leak into a deployment.
- **The Postgres password in `docker-compose.yml` is local-only** and never reachable from outside your machine.
- **The API has no authentication.** Add it before deploying anywhere public — anyone who can reach `/api/ask/` can spend your OpenAI credits.
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
You made a GET request. Both endpoints accept **POST only**.

**Answers ignore what I ingested**
Retrieval only searches chunks that have embeddings. If a document was ingested while the API key was invalid, its chunks may have none. Re-ingest it, and raise `top_k` if the relevant text is not surfacing.

---

## Roadmap

Planning documents live in **[AI-Assistant_Plan-and-docs](https://github.com/Emmanuel-Benjamin00/AI-Assistant_Plan-and-docs)** (`HIGH_LEVEL_PLAN.md`, `BUILD_PLAN.md`).

Local RAG works end to end. Next: file uploads beyond raw text, smarter chunking with overlap, authentication, and cloud deployment with CI/CD and monitoring.
