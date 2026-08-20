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
COPY requirements.txt requirements.txt
ARG CACHEBUST=1
RUN pip install "./mongomcp[agent]" && pip install -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY app.py app.py
COPY mcp_processor.py mcp_processor.py
COPY conversation_checkpoint.py conversation_checkpoint.py
COPY aws_settings.py aws_settings.py
COPY settings_loader.py settings_loader.py
COPY cognito_auth.py cognito_auth.py
COPY auth/ auth/
COPY --from=rootfolder ./agent_instructions.md agent_instructions.md

COPY --from=frontend-build /build/frontend/dist frontend/dist

# MCP session-reuse strategy (A/B-selectable): off = fresh client per call;
# turn = pool per turn on the request's own loop (parallel across requests);
# container = one shared pool across turns on a single background loop.
ENV MCP_SESSION_POOL=turn
ENV MCP_SESSION_POOL_SIZE=6

EXPOSE 8001

CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8001", "app:app", "--timeout", "300", "--log-level", "warning"]
