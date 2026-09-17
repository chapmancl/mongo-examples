import os
import sys
import logging
import time
from flask import Flask, send_from_directory, request, jsonify, abort, Response
from flask_cors import CORS
import requests
from mcp_processor import APIQueryProcessor, QueryResponse, QueryRequest
from mongomcp import __version__ as MCP_VERSION
import mimetypes
import traceback
from typing import Optional, List, Any, Dict
import threading
import json
import queue
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(base_dir))

logging.basicConfig(level=logging.INFO)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
USE_LOCAL_MODE = os.getenv('USE_LOCAL_MODE', 'false').lower() == "true"
# Import settings to check SAVE_LLM_HISTORY flag
if USE_LOCAL_MODE:
    print("# ------ Using local_settings.py ------ #")
    from local_settings import settings
else:
    # Running with kubernetes in EKS/Fargate
    try:
        from AWS_settings import settings
    except ImportError:
        print("# ------ ERROR, no settings file ----- #")
        sys.exit(1)

mimetypes.add_type('application/javascript', '.js')

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'frontend', 'dist'))
CORS(app)

processor = APIQueryProcessor()
_site_warmup_lock = threading.Lock()
_site_warmup_done = False


def _warmup_tool_discovery_once() -> None:
    """Run MCP tool discovery once on first site load, retrying until tools are found."""
    global _site_warmup_done
    if _site_warmup_done:
        return
    with _site_warmup_lock:
        if _site_warmup_done:
            return
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                # If tools already loaded (e.g. __init__ succeeded), we're done.
                if processor.mcp_tools_config:
                    _site_warmup_done = True
                    return
                # Tools not yet discovered — trigger discovery now.
                processor._discover_tools()
                if processor.mcp_tools_config:
                    _site_warmup_done = True
                    return
            except Exception:
                traceback.print_exc()

            if attempt >= max_attempts:
                # Do not block initial page load on warmup failure.
                app.logger.warning("Warmup tool discovery failed after %s attempts.", max_attempts)
                return
            wait_seconds = attempt * 5
            app.logger.warning(
                "Warmup tool discovery returned no tools (attempt %s/%s). Retrying in %ss.",
                attempt,
                max_attempts,
                wait_seconds,
            )
            time.sleep(wait_seconds)


def save_llm_history_to_mongo(username: str, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save LLM conversation history to MongoDB via the MCP server API.

    Args:
        username: Username of the user
        prompt: The prompt/question sent to the LLM
        response: The LLM's response
        metadata: Optional metadata to include
    """
    if not settings.SAVE_LLM_HISTORY:
        return

    if not username or not prompt or not response:
        app.logger.debug("Skipping LLM history save - missing required fields")
        return

    try:
        # Build the payload
        payload = {
            "username": username,
            "prompt": prompt,
            "response": response
        }

        if metadata:
            payload["metadata"] = metadata

        # Get the auth token
        auth_token = settings.get_auth_token()

        # Call the MCP server's /llm_history/save endpoint
        url = f"{settings.mongo_mcp_root}/llm_history/save"

        response = requests.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {auth_token}"
            },
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            app.logger.info(f"LLM history saved for user {username}, id: {data.get('id')}")
        else:
            app.logger.warning(f"Failed to save LLM history: HTTP {response.status_code}")

    except requests.exceptions.Timeout:
        app.logger.warning("Timeout saving LLM history to MongoDB")
    except Exception as e:
        app.logger.error(f"Error saving LLM history: {e}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "version": MCP_VERSION,
        "processor_ready": processor.init_error is None,
    }), 200


@app.route('/query', methods=['POST'])
def api_query():
    #add logging
    resp = '{"status": "error", "message": "Unknown error"}'  # Default error response
    code = 500
    username = None
    prompt = None
    reasoning_steps = []  # Collect reasoning steps

    try:
        payload = request.get_json(force=True)

        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")

        q = (payload.get("input", "") or "").strip()
        if not q:
            raise ValueError("Empty input")

        username = payload.get("username", "anonymous")
        prompt = q

        # Create a custom emit handler to capture reasoning steps
        def capture_emit(message, status="Processing"):
            msg = str(message) if not isinstance(message, Exception) else str(message)
            skip_patterns = ['LLM is still thinking', 'Querying Claude']
            if not any(p in msg for p in skip_patterns):
                reasoning_steps.append({"status": status, "message": msg})

        req = QueryRequest(input=q, history=payload.get("history", []), user_id=payload.get("user_id"), username=username, session_id=payload.get("session_id"))
        result = processor.query_with_mcp_tools(req, emit=capture_emit)
        resp = result.json()
        code = 200

        # Save LLM history if successful (in background thread to avoid blocking)
        if code == 200 and result.content:
            response_text = result.content.get("text", "")
            if response_text:
                threading.Thread(
                    target=save_llm_history_to_mongo,
                    args=(
                        username,
                        prompt,
                        response_text,
                        {
                            "session_id": payload.get("session_id"),
                            "user_id": payload.get("user_id"),
                            "endpoint": "/query",
                            "reasoning_steps": reasoning_steps  # Include processing log
                        }
                    ),
                    daemon=True
                ).start()

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


@app.route('/reset', methods=['POST'])
def full_reset():
    """Full application reset: clear state, drop cached MCP clients, reload tool discovery."""
    try:
        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")
        resp = processor.reset().json()
        return jsonify(resp), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify(QueryResponse(status="Error", error=str(e)).json()), 500


@app.route('/pattern/save', methods=['POST'])
def save_pattern():
    """Save the current interaction as a reusable query pattern."""
    try:
        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")
        payload = request.get_json(force=True) or {}
        user_id = (payload.get("user_id") or "").strip()
        session_id = (payload.get("session_id") or "").strip()
        history = payload.get("history") or []
        resp = processor.save_pattern(user_id, session_id, history)
        return jsonify(resp.json()), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify(QueryResponse(status="Error", error=str(e)).json()), 500


@app.route('/feedback', methods=['POST'])
def record_feedback():
    """Record user feedback (positive/negative) on the last interaction."""
    try:
        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")
        payload = request.get_json(force=True)
        user_id = (payload.get("user_id") or "").strip()
        session_id = (payload.get("session_id") or "").strip()
        feedback = (payload.get("feedback") or "").strip()
        history = payload.get("history") or []
        if not user_id or not session_id or feedback not in ("positive", "negative"):
            return jsonify({"error": "user_id, session_id, and feedback (positive|negative) required"}), 400
        resp = processor.record_feedback(user_id, session_id, feedback, history)
        return jsonify(resp.json()), 200
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

        local_queue = queue.Queue()
        reasoning_steps = []  # Collect reasoning steps for history

        def emit_local(message, status="Processing"):
            if isinstance(message, Exception):
                resp = QueryResponse(status="Error", error=str(message), message=str(message))
            else:
                resp = QueryResponse(status=status, message=str(message))
                # Capture reasoning steps (skip heartbeat noise)
                msg = str(message)
                skip_patterns = ['LLM is still thinking', 'Querying Claude']
                if not any(p in msg for p in skip_patterns):
                    reasoning_steps.append({"status": status, "message": msg})
            local_queue.put(resp.json())

        def read_local_stream(timeout=0.1):
            while True:
                try:
                    yield local_queue.get(timeout=timeout)
                except queue.Empty:
                    break

        def pop_local_messages() -> List[str]:
            msgs = []
            while True:
                try:
                    msgs.append(local_queue.get_nowait())
                except queue.Empty:
                    break
            return msgs

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

            HEARTBEAT_INTERVAL = 5.0
            last_heartbeat = time.monotonic()

            # Stream messages as they arrive from the handler
            while thread.is_alive():
                for msg in read_local_stream(timeout=0.1):
                    yield msg + '\n'
                    last_heartbeat = time.monotonic()

                if time.monotonic() - last_heartbeat >= HEARTBEAT_INTERVAL:
                    # Drain queue first, then emit heartbeat
                    for msg in pop_local_messages():
                        yield msg + '\n'
                    yield QueryResponse(status='LLM Thinking...', message='LLM is still thinking...').json() + '\n'
                    last_heartbeat = time.monotonic()

            # Drain any remaining messages after thread completes
            for msg in pop_local_messages():
                yield msg + '\n'

            # Yield the final result and exception as a tuple
            yield (result[0], exception[0])

        result = None
        exception = None
        username = payload.get("username", "anonymous")

        yield QueryResponse(status='querying', message='Querying Claude with MCP tools...').json() + '\n'
        req = QueryRequest(input=q, history=payload.get("history", []), user_id=payload.get("user_id"), username=username, session_id=payload.get("session_id"))
        for item in execute_in_thread(lambda: processor.query_with_mcp_tools(req, emit=emit_local)):
            if isinstance(item, tuple):
                result, exception = item
            else:
                yield item

        # Yield final result
        if exception:
            yield QueryResponse(error=str(exception)).json() + '\n'
        elif result:
            yield result.json() + '\n'

            # Save LLM history if successful
            if result and result.content:
                response_text = result.content.get("text", "")
                if response_text:
                    # Save in a separate thread to not block the response
                    threading.Thread(
                        target=save_llm_history_to_mongo,
                        args=(
                            username,
                            q,
                            response_text,
                            {
                                "session_id": payload.get("session_id"),
                                "user_id": payload.get("user_id"),
                                "endpoint": "/query/stream",
                                "reasoning_steps": reasoning_steps  # Include processing log
                            }
                        ),
                        daemon=True
                    ).start()

    except Exception as e:
        traceback.print_exc()
        yield QueryResponse(error=str(e)).json() + '\n'

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder = app.static_folder
    app.logger.debug("serve: static_folder=%s", static_folder)
    if path != "" and os.path.exists(os.path.join(static_folder, path)):
        return send_from_directory(static_folder, path)
    index_path = os.path.join(static_folder, 'index.html')
    if os.path.exists(index_path):
        _warmup_tool_discovery_once()
        return send_from_directory(static_folder, 'index.html')
    return "Frontend not found. you must first build the project.", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
