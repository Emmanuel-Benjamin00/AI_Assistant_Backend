# AI Assistant — Backend

Django REST backend for the AI Assistant project. PostgreSQL runs in Docker with **pgvector** for later RAG work.

## New Developer Quickstart

If you are cloning this project for the first time, run exactly this:

```bash
git clone <backend-repo-url>
cd AI_Assistant_Backend
docker compose up -d
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

API runs at `http://127.0.0.1:8000/`.

If Docker gives `permission denied` on `/var/run/docker.sock`, run `newgrp docker` and retry in the same terminal.

### OpenAI (RAG — Step 5)

Copy [`.env.example`](.env.example) to `.env` and set at least:

- `OPENAI_API_KEY`

Optional overrides (omit these to use code defaults; only add later if you want to pin specific models):

- `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `OPENAI_CHAT_MODEL` (default `gpt-4o-mini`)

Endpoints:

- `POST /api/documents/` — ingest text (chunks + embeddings)
- `POST /api/ask/` — ask a question (`{"question":"...","top_k":5}`)

### Stopping services (and keeping your DB safe)

- Stop Django dev server: press `Ctrl + C` in the terminal running `python manage.py runserver`.
- Stop Docker services for this project: `docker compose down`.
- Start again later: `docker compose up -d`.

Database safety:

- `docker compose down` **does not delete** your Postgres data volume.
- `docker compose down -v` **deletes volumes** (this will wipe DB data for this project).

## Prerequisites

- **Docker** + Docker Compose v2 (for Postgres)
- **Python 3.12+** (matches current setup; adjust if your team standard differs)
- Linux/macOS/WSL: terminal access

## What developers run (clone → running API)

From **`AI_Assistant_Backend/`** (this repo root after `git clone`):

```bash
docker compose up -d
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Use the **same working directory** for both Compose and Django commands (`AI_Assistant_Backend/`). That keeps onboarding predictable.

### Database connection (local dev)

Docker maps Postgres to your machine as:

| Setting | Value |
|--------|--------|
| Host | `127.0.0.1` |
| Port | `5433` |
| Database | `aiassistant` |
| User / password | Match [`docker-compose.yml`](docker-compose.yml) (`aiassistant` / `aiassistant` in the default file) |

**GUI tools (e.g. pgAdmin):** use host **`127.0.0.1`**, not a container name. Maintenance DB can be **`aiassistant`** or **`postgres`**.

> Dev credentials in `docker-compose.yml` are for **local development only**. Replace with secrets management before production.

### Virtualenv location

Create **`.venv/` inside this cloned repo** (next to `manage.py`). It is listed in [`.gitignore`](.gitignore) and must **not** be committed.

- Putting `.venv/` **outside** the repo on your disk works for one machine but is **not** what you document for the team.
- The usual convention for a **backend-only** repo is: clone → `python3 -m venv .venv` → activate → install → run.

### Docker permission tip (Linux)

If you see `permission denied` on the Docker socket after joining the `docker` group, run `newgrp docker` in that terminal, or fully restart your IDE/terminal session.

---

## Repo layout expectations

### This repo (`AI_Assistant_Backend/`) should contain everything another developer needs

Keep these at **this** git root:

- [`docker-compose.yml`](docker-compose.yml) — Postgres + pgvector
- [`requirements.txt`](requirements.txt)
- `manage.py`, Django project package, apps

That way `git clone` + the commands above are enough to run the stack.

### Parent folder `AIAssistant/` (optional local workspace)

If **`AI_Assistant_Backend`** is the **only** thing you push for the API, you may still keep a parent folder like `AIAssistant/` on your machine for docs, a future frontend, or experiments.

**Do not** store the **only** copy of `docker-compose.yml` and `requirements.txt` (or the canonical setup docs) **only** under that parent folder — teammates who clone the **backend repo** will not see them.

If you keep an extra Compose file at `AIAssistant/` for your own full-stack workflow, either remove it or treat it as **clearly optional** and duplicate or link to the backend README so the **backend repo** stays the source of truth.

---

## Short summary

| Artifact | Where it belongs |
|----------|-------------------|
| `requirements.txt` | This repo root (`AI_Assistant_Backend/`) |
| `docker-compose.yml` | This repo root |
| `.venv/` | This repo root, **gitignored**, created locally |
| Django code | This repo root (`manage.py`, packages, apps) |

---

## Short answer (developer-friendly)

Correct, traditional, clone-friendly: `requirements.txt` + `docker-compose.yml` + `manage.py` all under `AI_Assistant_Backend/` (backend git root).

`venv`: create inside that clone (e.g. `.venv/`), never commit — same pattern every Django/open-source backend uses.

If you want, we can next align your repo by moving `docker-compose.yml` into `AI_Assistant_Backend/` and adjusting paths in your docs — say if you want that done in the repo.

---

## Next steps (project roadmap)

See the planning docs in the parent workspace if present (`BUILD_PLAN.md`, `HIGH_LEVEL_PLAN.md`), or continue with models, RAG APIs, and tests per your backlog.
