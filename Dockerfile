FROM python:3.11-slim

# Install Node.js (for building the React frontend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Build the React frontend
# Copy manifests first so npm ci is cached unless dependencies change
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN npm --prefix frontend ci

# Now copy source (node_modules and dist are excluded via .dockerignore)
COPY frontend/ ./frontend/
RUN npm --prefix frontend run build
# Verify the build produced index.html (catches silent build failures)
RUN test -f frontend/dist/index.html

# Copy backend source and startup script
COPY backend/ ./backend/
COPY start.sh ./start.sh
RUN chmod +x start.sh

EXPOSE 8000

# Use shell script so $PORT is always expanded by sh, whether Railway
# injects it via the Dockerfile CMD or via its startCommand override.
CMD ["sh", "start.sh"]
