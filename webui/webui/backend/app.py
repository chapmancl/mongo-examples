import os
from flask import Flask, send_from_directory, request, jsonify, abort, Response
from flask_cors import CORS
import requests
import settings
from mcp_processor import APIQueryProcessor
import mimetypes
import traceback
from typing import Optional, List, Any
from pydantic import BaseModel
import json

mimetypes.add_type('application/javascript', '.js')

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'))
CORS(app)

if json.loads(os.getenv('USE_LOCAL_MODE', 'false').lower()):
    print("Running with local MCP server")
    MCP_CLUSTER_ROOT = settings.mongo_mcp_root_local
else:
    MCP_CLUSTER_ROOT = settings.mongo_mcp_root
processor = APIQueryProcessor()


def _sanitize_obj(o):
    """Recursively remove newline characters from string fields in an object."""
    if isinstance(o, dict):
        return {k: _sanitize_obj(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize_obj(v) for v in o]
    if isinstance(o, str):
        return o.replace('\\n', '').replace('\\r', '')
    return o


class QueryRequest(BaseModel):
    input: str
    history: Optional[List[Any]] = None


class QueryResponse(BaseModel):
    answer: Optional[str] = None
    history: Optional[List[Any]] = None
    cache_stats: Optional[dict] = None
    message: Optional[str] = None

@app.route('/', methods=['GET'])
def index():
    return "MCP Query and Viewer API is running", 200

@app.route('/query', methods=['POST'])
def api_query():
    resp = '{"status": "error", "message": "Unknown error"}'  # Default error response
    try:
        payload = request.get_json(force=True)
        
        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")

        q = (payload.get("input", "") or "").strip()
        if not q:
            raise ValueError("Empty input")
    
        try:
            # Mirror CLI commands
            if q.startswith("clear"):
                processor.clear_history()
                processor.clear_all_caches()
                return QueryResponse(message="History and caches cleared", history=processor.get_history())

            if q.startswith("cache stats"):
                stats = processor.get_cache_stats()
                return QueryResponse(cache_stats=stats, history=processor.get_history())

            if q.startswith("cache clear"):
                processor.clear_all_caches()
                return QueryResponse(message="All caches cleared", history=processor.get_history())

            # Normal question => forward to Claude/Bedrock with MCP tools
            answer, history = processor.query_claude_with_mcp_tools(q, payload.get("history", []))
            resp = QueryResponse(answer=answer, history=history)

        except Exception as e:
            traceback.print_exc()
            raise        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    return jsonify(resp.dict()), 200

@app.route('/query/stream', methods=['POST'])
def stream_query():
    """Stream the response for long-running queries."""
    # Read request payload here while request context is active
    try:
        payload = request.get_json(force=True)
        # Pass a copy of payload into generator to avoid accessing `request` inside it
        return Response(generate(payload), mimetype='application/x-ndjson')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def generate(payload):
    try:
        if processor.init_error:
            yield json.dumps(_sanitize_obj({'error': f"Processor initialization failed: {processor.init_error}"})) + '\n'
            return

        q = (payload.get("input", "") or "").strip()
        if not q:
            yield json.dumps(_sanitize_obj({'error': "Empty input"})) + '\n'
            return

        # Yield status update
        yield json.dumps(_sanitize_obj({'status': 'processing', 'message': f'Processing: {q}'})) + '\n'

        try:
            # Mirror CLI commands
            if q.startswith("clear"):
                processor.clear_history()
                processor.clear_all_caches()
                result = QueryResponse(message="History and caches cleared", history=processor.get_history())
                yield json.dumps(_sanitize_obj(result.dict())) + '\n'
                return

            if q.startswith("cache stats"):
                stats = processor.get_cache_stats()
                result = QueryResponse(cache_stats=stats, history=processor.get_history())
                yield json.dumps(_sanitize_obj(result.dict())) + '\n'
                return

            if q.startswith("cache clear"):
                processor.clear_all_caches()
                result = QueryResponse(message="All caches cleared", history=processor.get_history())
                yield json.dumps(_sanitize_obj(result.dict())) + '\n'
                return

            # Yield progress update
            yield json.dumps(_sanitize_obj({'status': 'querying', 'message': 'Querying Claude with MCP tools...'})) + '\n'

            # Normal question => forward to Claude/Bedrock with MCP tools
            answer, history = processor.query_claude_with_mcp_tools(q, payload.get("history", []))
            result = QueryResponse(answer=answer, history=history)

            # Yield final result
            yield json.dumps(_sanitize_obj(result.dict())) + '\n'

        except Exception as e:
            traceback.print_exc()
            yield json.dumps(_sanitize_obj({'error': str(e)})) + '\n'

    except Exception as e:
        traceback.print_exc()
        yield json.dumps(_sanitize_obj({'error': str(e)})) + '\n'




@app.route('/<path:path>')
def serve(path):
    static_folder = app.static_folder
    print(static_folder)
    if path != "" and os.path.exists(os.path.join(static_folder, path)):
        return send_from_directory(static_folder, path)
    index_path = os.path.join(static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(static_folder, 'index.html')
    return "Frontend not built. Run the frontend dev server or build the project.", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
