#!/usr/bin/env bash
# Loads .env if present, then starts the dashboard.
set -a; [ -f .env ] && . ./.env; set +a
exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
