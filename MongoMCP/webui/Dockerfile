FROM node:20-slim AS frontend-build

WORKDIR /build/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM mcp-base:latest
WORKDIR /app

# ── Dependency layer (cached until requirements files change) ──────────────────
COPY --from=rootfolder ./mongomcp/ mongomcp/
ARG CACHEBUST=1
RUN pip install "./mongomcp[agent]"

# ── Application code ──────────────────────────────────────────────────────────
COPY app.py app.py
COPY mcp_processor.py mcp_processor.py
COPY aws_settings.py aws_settings.py

COPY --from=frontend-build /build/frontend/dist frontend/dist

EXPOSE 8001

CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8001", "app:app", "--timeout", "300"]
