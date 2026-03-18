import os
from flask import Flask, send_from_directory, request, jsonify, abort, Response
from flask_cors import CORS
import requests
from mcp_processor import APIQueryProcessor, QueryResponse, QueryRequest
import mimetypes
import traceback
from typing import Optional, List, Any
import threading
import json

mimetypes.add_type('application/javascript', '.js')

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'frontend', 'dist'))
CORS(app)

processor = APIQueryProcessor()

@app.route('/query', methods=['POST'])
def api_query():
    resp = '{"status": "error", "message": "Unknown error"}'  # Default error response
    code = 500
    try:
        payload = request.get_json(force=True)
        
        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")

        q = (payload.get("input", "") or "").strip()
        if not q:
            raise ValueError("Empty input")

        req = QueryRequest(input=q, history=payload.get("history", []))  # Validate input with Pydantic model
        # Mirror CLI commands
        if q.startswith("clear history"):
            processor.clear_history()
            resp = processor.clear_all_caches().json()
            code = 205
        elif q.startswith("clear"):            
            resp = processor.clear_all_caches().json()
            code = 205
        elif q.startswith("cache stats"):
            resp = processor.get_cache_stats().json() 
            code = 200

        elif q.startswith("cache clear"):
            resp = processor.clear_all_caches().json()
            code = 200
        else:
            # Normal question => forward to LLM/Bedrock with MCP tools
            resp = processor.query_with_mcp_tools(req).json()    
            code = 200

    except Exception as e:
        traceback.print_exc()
        resp = QueryResponse(status="Error", error=str(e)).json()
    
    return jsonify(resp), code


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


@app.route('/history/reset', methods=['POST'])
def reset_history():
    try:
        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")

        resp = processor.clear_history().json()
        return jsonify(resp), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify(QueryResponse(status="Error", error=str(e)).json()), 500

def generate(payload):
    try:
        if processor.init_error:
            yield QueryResponse(error=f"Processor initialization failed: {processor.init_error}").json() + '\n'
            return

        q = (payload.get("input", "") or "").strip()
        if not q:
            yield QueryResponse(status="Error", error="Empty input").json() + '\n'
            return

        # Generic threaded executor - returns (result, exception)
        def execute_in_thread(func):
            """Execute a function in a background thread and stream messages."""
            result = [None]
            exception = [None]
            
            def wrapper():
                try:
                    result[0] = func()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=wrapper, daemon=True)
            thread.start()
            
            # Stream messages as they arrive from the handler
            while thread.is_alive():
                for msg in processor.read_message_stream(timeout=0.1):
                    yield msg + '\n'
            
            # Drain any remaining messages after thread completes
            for msg in processor.pop_queued_messages():
                yield msg + '\n'
            
            # Yield the final result and exception as a tuple
            yield (result[0], exception[0])
        
        result = None
        exception = None
        
        # Mirror CLI commands
        if q.startswith("clear history"):
            for item in execute_in_thread(processor.clear_history):
                if isinstance(item, tuple):
                    result, exception = item
                else:
                    yield item
        elif q.startswith("clear"):
            for item in execute_in_thread(processor.clear_all_caches):
                if isinstance(item, tuple):
                    result, exception = item
                else:
                    yield item
        elif q.startswith("cache stats"):
            for item in execute_in_thread(processor.get_cache_stats):
                if isinstance(item, tuple):
                    result, exception = item
                else:
                    yield item
        elif q.startswith("cache clear"):
            for item in execute_in_thread(processor.clear_all_caches):
                if isinstance(item, tuple):
                    result, exception = item
                else:
                    yield item
        else:
            # Yield progress update
            yield QueryResponse(status='querying', message='Querying Claude with MCP tools...').json() + '\n'

            # Run the query in a background thread to allow concurrent message streaming
            req = QueryRequest(input=q, history=payload.get("history", []))
            for item in execute_in_thread(lambda: processor.query_with_mcp_tools(req)):
                if isinstance(item, tuple):
                    result, exception = item
                else:
                    yield item
        
        # Yield final result
        if exception:
            yield QueryResponse(error=str(exception)).json() + '\n'
        elif result:
            yield result.json() + '\n'

    except Exception as e:
        traceback.print_exc()
        yield QueryResponse(error=str(e)).json() + '\n'

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder = app.static_folder
    print(static_folder)
    if path != "" and os.path.exists(os.path.join(static_folder, path)):
        return send_from_directory(static_folder, path)
    index_path = os.path.join(static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(static_folder, 'index.html')
    return "Frontend not found. you must first build the project.", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
