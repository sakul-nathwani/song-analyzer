# ── Stage 1: Build React frontend ─────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Copy manifests first so this layer is cached unless deps change
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build
RUN test -f dist/index.html   # catch silent build failures

# ── Stage 2: Python runtime ────────────────────────────────────────────────
# Node.js never reaches this stage — saves ~200 MB from the final image.
FROM python:3.11-slim AS runtime

# libsndfile1 — soundfile backend for WAV / FLAC / OGG
# ffmpeg      — audioread backend for MP3 / M4A (absent before; would silently fail)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/

# Pull only the compiled assets — no source, no node_modules
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

COPY start.sh ./start.sh
RUN chmod +x start.sh

EXPOSE 8000
CMD ["sh", "start.sh"]
