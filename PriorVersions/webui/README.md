Web UI for MCP Client

This project runs as a single server:
- Flask backend serves API endpoints: `/query`, `/query/stream`
- Flask also serves the built frontend from `frontend/dist`

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm

## Build

Run from the repository root (`mcpclient`):

```bash
python -m pip install -r webui/backend/requirements.txt
cd webui/frontend
npm install
npm run build
```

## Run

From `webui`:

```bash
python backend/app.py
```

## One-command build and run

From `webui`:

```bash
bash build_and_run.sh
```

## Access

- UI: `http://localhost:8000`
- API: `http://localhost:8000/query`
- Streaming API: `http://localhost:8000/query/stream`

## Notes

- Frontend uses same-origin API calls by default.
- Set `VITE_API_URL` only if you intentionally want a different API host.
- Re-run `npm run build` after frontend code changes.

## Docker

Build and run from repository root:

```bash
docker build -t mcp-webui ./webui
docker run --rm -p 8000:8000 mcp-webui
```
