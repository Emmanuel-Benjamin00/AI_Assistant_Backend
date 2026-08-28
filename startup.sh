#!/usr/bin/env bash
set -e

python manage.py migrate --noinput
python manage.py collectstatic --noinput

# App Service provides $PORT; default to 8000 for local runs.
gunicorn AI_Assistant.wsgi:application \
    --bind=0.0.0.0:${PORT:-8000} \
    --workers=2 \
    --timeout=120
