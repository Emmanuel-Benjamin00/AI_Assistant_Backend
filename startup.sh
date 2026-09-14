#!/usr/bin/env bash
# Production entrypoint. Azure App Service startup command: bash startup.sh
set -euo pipefail

python manage.py migrate --noinput
python manage.py createcachetable
python manage.py collectstatic --noinput

# Index the sample documents on first boot so the demo has something to answer from.
if [ "${SEED_DEMO:-false}" = "true" ]; then
    python manage.py seed_demo || echo "seed_demo failed; starting the API anyway"
fi

# App Service provides $PORT; default to 8000 for local runs.
# gthread: a streamed answer holds its thread for several seconds, so each worker serves
# several requests at once instead of one.
exec python -m gunicorn AI_Assistant.wsgi:application \
    --bind=0.0.0.0:${PORT:-8000} \
    --workers=${GUNICORN_WORKERS:-2} \
    --worker-class=gthread \
    --threads=${GUNICORN_THREADS:-4} \
    --timeout=120 \
    --access-logfile - \
    --error-logfile -
