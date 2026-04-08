Web UI for MCP Client

This project runs as a single server:
- Flask backend serves API endpoints: `/query`, `/query/stream`
- Flask also serves the built frontend from `frontend/dist`

For local development, this UI expects the MCP server to be running separately and the Mongo config data to be initialized by `../tools/mongosetup.py`.

## Required Local Settings

Before running the Web UI locally, update `webui/local_settings.py` with your MongoDB connection details:

```python
self._credentials = {
  "username": "your_mongodb_username",
  "password": "your_mongodb_password",
  "mongoUrl": "your_cluster.mongodb.net"
}
```

The same `_credentials` values must also be set in `../local_settings.py` so both the MCP server and Web UI point at the same MongoDB cluster.

You must also set the Web UI token from the output of `python ../tools/mongosetup.py`:

```python
self.AUTH_TOKEN = "paste_the_AUTH_TOKEN_value_here"
```

The setup script prints a line in this format:

```bash
AUTH_TOKEN = "..."
```

Copy that exact value into `webui/local_settings.py` before starting `python app.py`.

## Prerequisites

- Python 3.11+
- Node.js 18+ and npm

## Build

Run from the repository root (`MongoMCP`):

```bash
python tools/mongosetup.py
pip install -e ../mongomcp[agent]
pip install -r webui/requirements.txt
cd webui/frontend
npm install
npm run build
```

## Run

Start the MCP server from the repository root:

```bash
fastmcp run mongo_mcp.py --transport http --port 8000
```

Then, from `webui`:

```bash
python app.py
```

If you are using local settings, `python tools/mongosetup.py` is the step that creates the `mcp_config` database, seeds the required collections, and generates the default `webui_chatuser` agent identity.

## One-command build and run

From `webui`:

```bash
bash build_and_run.sh
```

## Access

- UI: `http://localhost:8001`
- API: `http://localhost:8001/query`
- Streaming API: `http://localhost:8001/query/stream`

The UI expects the MCP server at `http://localhost:8000` unless `MONGO_MCP_ROOT` is overridden.

## Notes

- Frontend uses same-origin API calls by default.
- Set `VITE_API_URL` only if you intentionally want a different API host.
- Re-run `npm run build` after frontend code changes.

## Docker

Build and run from repository root:

```bash
docker build -t mcp-webui ./webui
docker run --rm -p 8001:8001 mcp-webui
```

## Pattern Cache (AI Tool Routing)

When `AI_TOOL_ROUTING` is enabled, the tool router caches successful query patterns in the `mcp_patterns` collection. To enable semantic matching of similar questions, create a vector search index on your MongoDB Atlas cluster:

```javascript
db.mcp_patterns.createSearchIndex({
  name: "pattern_embedding_index",
  type: "vectorSearch",
  definition: {
    fields: [{
      path: "embedding",
      type: "vector",
      numDimensions: 1024,
      similarity: "cosine"
    },
    {
      "type": "filter",
      "path": "tool_scope"
    }
    ]
  }
})
```

Without this index, pattern matching falls back to exact hash matching only.
