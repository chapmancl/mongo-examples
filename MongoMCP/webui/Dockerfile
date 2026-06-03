FROM node:20-slim AS frontend-build

WORKDIR /build/frontend

COPY frontend/package*.json ./
RUN npm ci

ARG CACHEBUST=1
ARG VITE_LLM_MODEL_ID=claude
COPY frontend/ ./
RUN VITE_LLM_MODEL_ID=$VITE_LLM_MODEL_ID npm run build


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
COPY cognito_auth.py cognito_auth.py
COPY --from=rootfolder ./agent_instructions.md agent_instructions.md

COPY --from=frontend-build /build/frontend/dist frontend/dist

EXPOSE 8001

CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8001", "app:app", "--timeout", "300", "--log-level", "warning"]
