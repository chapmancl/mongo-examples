Web UI for MCP Client

This folder contains a minimal Flask backend (proxy + static serve) and a Vite React frontend.

Getting started (development):

1. Start the Flask proxy (optional, provides `/query` that forwards to FastAPI):

```bash
python -m pip install -r webui/backend/requirements.txt
python webui/backend/app.py
```

2. Start the React dev server:

```bash
cd webui/frontend
npm install
npm run dev
```

In dev you can either call the FastAPI directly at `http://localhost:8000/query` (the React app calls `/api/query` so run the Flask proxy on port 5000), or configure the frontend to call the FastAPI URL directly.

Building for production (calls FastAPI directly):

Set the API URL at build time (example in `.env.example`):

```bash
cp .env.example .env
```

Then build:

```bash
cd webui/frontend
npm run build
# then serve the built files with the Flask backend
python webui/backend/app.py
```
