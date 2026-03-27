#!/bin/sh
exec uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --timeout-keep-alive 120 \
  --limit-max-requests 1000
