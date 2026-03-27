#!/bin/sh
exec uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --timeout-keep-alive 300 \
  --timeout-graceful-shutdown 30 \
  --limit-max-requests 1000
