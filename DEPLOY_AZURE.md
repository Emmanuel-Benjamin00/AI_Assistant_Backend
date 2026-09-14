# Deploying to Azure

This guide deploys the full AI Assistant to Azure using the Azure portal:

| Part | Azure service | Repo |
|---|---|---|
| Database | Azure Database for PostgreSQL – Flexible Server (with pgvector) | — |
| API | Azure App Service (Linux, Python 3.12) | `AI_Assistant_Backend` |
| UI | Azure Static Web Apps (Free plan) | `AI_Assistant_Frontend` |
| AI | OpenAI API **or** Azure OpenAI | — |

Allow about an hour the first time. Do the steps in order: each one produces a value the next one needs.

---

## 0. Before you start

- **Rotate any key that was ever committed.** Azure OpenAI settings were once committed to this repo's history. If that Azure OpenAI resource still exists, open it → **Keys and Endpoint** → **Regenerate**. If the resource has been deleted, the old key no longer works. Deleting the file does not remove it from git history.
- **Check what already exists.** In the portal open **All resources**. If an older `ai-assistant-api` App Service or database is there, either reuse it (skip its create step and just apply the settings below) or delete it so you are not paying twice.
- **Pick one region and use it for everything**, e.g. *Central India* or *Southeast Asia*. Azure for Students subscriptions only allow some regions; if a create fails with `RequestDisallowedByAzure` or a policy error, try another region.
- **Set a budget alert:** search **Cost Management** → **Budgets** → **Add**, e.g. $20/month with an email alert at 80%.

Keep a text file open and paste values into it as you go: server name, DB password, app URL, keys.

---

## 1. Create the resource group

Search **Resource groups** → **Create**.

- Name: `ai-assistant-rg`
- Region: your chosen region

Put every resource below in this group, so you can delete everything at once later.

---

## 2. Create the PostgreSQL database

Search **Azure Database for PostgreSQL flexible servers** → **Create**.

**Basics**

| Field | Value |
|---|---|
| Resource group | `ai-assistant-rg` |
| Server name | e.g. `ai-assistant-db-<yourname>` (must be globally unique) |
| Region | your region |
| PostgreSQL version | 16 |
| Workload type | **Development** |
| Compute + storage | **Configure server** → Burstable, **B1ms**, 32 GiB storage |
| Authentication | PostgreSQL authentication only |
| Admin username | e.g. `aiadmin` |
| Password | a strong password — save it |

**Networking**

- Connectivity method: **Public access (allowed IP addresses)**
- Tick **Allow public access from any Azure service within Azure to this server**
- Optional: **+ Add current client IP address** so you can connect from your laptop with pgAdmin

Click **Review + create** → **Create** and wait for it to finish (5–10 minutes).

### 2a. Allow the pgvector extension

Open the server → **Settings** → **Server parameters** → search `azure.extensions` → tick **VECTOR** → **Save**.

Without this, the first migration fails with `extension "vector" is not allow-listed`.

### 2b. Create the database

Open the server → **Settings** → **Databases** → **+ Add** → name `aiassistant` → **Save**.

### 2c. Build the connection string

```
postgres://aiadmin:<PASSWORD>@<server-name>.postgres.database.azure.com:5432/aiassistant
```

If the password contains special characters (`@ : / ? # % &`), URL-encode them first, e.g. `@` becomes `%40`. Easiest is to use only letters and numbers in the password.

---

## 3. Choose the AI provider

### Option A — OpenAI API (simplest)

1. Create a key at <https://platform.openai.com/api-keys>.
2. **Set a monthly budget** at <https://platform.openai.com/settings/organization/limits> (e.g. $5). Your demo is public; the rate limit helps, but the budget is your hard stop.

App settings you will use: `LLM_PROVIDER=openai`, `OPENAI_API_KEY=<key>`.

### Option B — Azure OpenAI (billed to your Azure credits)

Some student subscriptions cannot create Azure OpenAI resources. If creation is blocked, use Option A.

1. Search **Azure OpenAI** → **Create** → resource group `ai-assistant-rg`, a region that offers the models, Standard tier.
2. Open the resource → **Go to Azure AI Foundry portal** → **Deployments** → **Deploy model**:
   - `text-embedding-3-small` — name the deployment `text-embedding-3-small`
   - `gpt-4o-mini` — name the deployment `gpt-4o-mini`
3. Back in the Azure portal: resource → **Keys and Endpoint** → copy **Key 1** and the **Endpoint**.

App settings you will use:

```
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<key 1>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
```

> The embedding model must produce 1536-dimension vectors (`text-embedding-3-small` and `text-embedding-ada-002` do).

---

## 4. Create the API (App Service)

Search **App Services** → **Create** → **Web App**.

| Field | Value |
|---|---|
| Resource group | `ai-assistant-rg` |
| Name | e.g. `ai-assistant-api` → gives `https://ai-assistant-api.azurewebsites.net` |
| Publish | **Code** |
| Runtime stack | **Python 3.12** |
| Operating system | Linux |
| Region | your region |
| Pricing plan | **Basic B1** (recommended) or **Free F1** |

> **F1 vs B1:** F1 is free but limited to 60 CPU minutes a day and sleeps when idle, so a recruiter's first request may take a minute. B1 costs credits but supports **Always On**, so the demo responds instantly.

**Review + create** → **Create**.

### 4a. Environment variables

Open the app → **Settings** → **Environment variables** → **App settings** → add each one → **Apply**.

Generate the two random values on your laptop:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(50))"   # SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(24))"   # INGEST_API_KEY
```

| Name | Value |
|---|---|
| `DJANGO_ENV` | `prod` |
| `SECRET_KEY` | the first random value |
| `DATABASE_URL` | the connection string from step 2c |
| `ALLOWED_HOSTS` | `ai-assistant-api.azurewebsites.net` (your app's host, no `https://`) |
| `CSRF_TRUSTED_ORIGINS` | `https://ai-assistant-api.azurewebsites.net` |
| `CORS_ALLOWED_ORIGINS` | leave empty for now — filled in step 6 |
| `INGEST_API_KEY` | the second random value — you type this in the UI to add documents |
| `SEED_DEMO` | `true` — indexes the sample documents on startup |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` — Azure installs `requirements.txt` on deploy |
| + the AI provider settings | from step 3 |

### 4b. General settings

App → **Settings** → **Configuration** → **General settings**:

- **Startup Command:** `bash startup.sh`
- **HTTPS Only:** On
- **Always On:** On (B1 only)

**Save**.

### 4c. Health check

App → **Monitoring** → **Health check** → Enable → path `/api/health/` → **Save**.

### 4d. Deploy the code

**Recommended — from GitHub (redeploys on every push):**

1. Push this repo's latest code to GitHub.
2. App → **Deployment** → **Deployment Center** → Source **GitHub** → authorize → organization, repository `AI_Assistant_Backend`, branch `dev` (or `main`) → **Save**.
3. Azure commits a workflow file under `.github/workflows/` to your repo and starts a run. Watch it in the repo's **Actions** tab. Run `git pull` afterwards so your local copy has that file.

**Alternative — Azure CLI from your laptop:**

```bash
az login
git archive -o app.zip HEAD      # zips only committed files, so no venv or .env
az webapp deploy --resource-group ai-assistant-rg --name ai-assistant-api --src-path app.zip --type zip
```

### 4e. Verify

Startup runs migrations and seeds the demo documents, so the first boot takes a few minutes. Watch it in App → **Monitoring** → **Log stream**.

```bash
curl https://ai-assistant-api.azurewebsites.net/api/health/
# {"status":"ok"}

curl https://ai-assistant-api.azurewebsites.net/api/documents/
# [{"id":2,"title":"US Healthcare Claims and EDI Primer","chunks":...}, ...]

curl -X POST https://ai-assistant-api.azurewebsites.net/api/ask/ \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is an 835?"}'
```

Once the documents list shows the samples, set `SEED_DEMO` to `false` (optional — seeding skips documents that already exist, but turning it off makes restarts faster).

---

## 5. Create the UI (Static Web Apps)

1. In the **frontend** repo, set the API URL in `.env.production`, then commit and push:

   ```dotenv
   VITE_API_BASE=https://ai-assistant-api.azurewebsites.net
   ```

2. Portal: search **Static Web Apps** → **Create**:

   | Field | Value |
   |---|---|
   | Resource group | `ai-assistant-rg` |
   | Name | `ai-assistant-web` |
   | Plan type | **Free** |
   | Source | GitHub → repo `AI_Assistant_Frontend`, branch `dev` (or `main`) |
   | Build presets | **Custom** |
   | App location | `/` |
   | Api location | *(empty)* |
   | Output location | `dist` |

3. **Review + create**. Azure adds a GitHub Actions workflow to the frontend repo and builds it. When the run finishes, open the resource's **URL**, e.g. `https://<random-name>.azurestaticapps.net`.

---

## 6. Connect the UI to the API (CORS)

App Service → **Environment variables** → set:

```
CORS_ALLOWED_ORIGINS=https://<random-name>.azurestaticapps.net
```

No trailing slash. **Apply**; the app restarts.

Open the Static Web App URL, click a sample question, and you should get an answer with sources.

---

## 7. Keeping costs down

- **Stop the database when you are not job hunting:** PostgreSQL server → **Overview** → **Stop**. Azure restarts a stopped server automatically after 7 days.
- **Scale the App Service plan to F1** when you do not need the always-on demo.
- **Check Cost Management** weekly. The B1ms database and B1 App Service are the main costs; Static Web Apps Free costs nothing.
- **To remove everything:** delete the `ai-assistant-rg` resource group.

---

## Troubleshooting

Always start with App Service → **Monitoring** → **Log stream**.

| Symptom | Cause and fix |
|---|---|
| `400 Bad Request` for every URL | `ALLOWED_HOSTS` does not match the host you called. Use the bare host name, no `https://`. |
| Browser console shows a **CORS** error | `CORS_ALLOWED_ORIGINS` is missing or has a trailing slash or `http://`. It must exactly match the Static Web App URL. |
| `extension "vector" is not allow-listed` in logs | Step 2a was skipped: allow `VECTOR` in `azure.extensions`. |
| `connection ... timeout` / `no pg_hba.conf entry` | Database networking: tick *Allow public access from any Azure service*. Check the server name and password in `DATABASE_URL`. |
| `KeyError: 'SECRET_KEY'` | `SECRET_KEY` app setting is missing. |
| `ModuleNotFoundError: No module named 'django'` | The build did not install requirements: set `SCM_DO_BUILD_DURING_DEPLOYMENT=true` and redeploy. |
| Answers say *The AI service is not configured* | `LLM_PROVIDER` and its key/endpoint/deployment settings are missing or misspelled. |
| *The AI service request failed* | Wrong key, wrong Azure deployment name, or the OpenAI budget is used up. Log stream shows the provider's status code. |
| UI says *Adding documents is locked* | Expected on the public demo. Type your `INGEST_API_KEY` into the *Access key* field. |
| `429 Too Many Requests` | The per-IP rate limit. Adjust `THROTTLE_ASK_RATE` / `THROTTLE_INGEST_RATE` (e.g. `60/hour`). |
| Site shows *Application Error* | The startup script failed. Log stream shows which command (migrate, collectstatic, gunicorn). |
