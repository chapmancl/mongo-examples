import os
import logging
import time
from flask import Flask, send_from_directory, request, jsonify, abort, Response
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
from mcp_processor import APIQueryProcessor, QueryResponse, QueryRequest
from mongomcp import __version__ as MCP_VERSION
from auth import init_auth, apply_identity, current_identity
import mimetypes
import traceback
from typing import Optional, List, Any
import threading
import json
import queue

logging.basicConfig(level=logging.INFO)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

mimetypes.add_type('application/javascript', '.js')

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'frontend', 'dist'))
# Honor X-Forwarded-For/Proto/Host from the fronting proxy/ALB so request.remote_addr
# reflects the real client IP rather than the proxy/container address.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
CORS(app)
# Install the pluggable auth guard. No-op unless AUTH_MODE (or AUTH_PROVIDER) is set.
init_auth(app)

# Optional, untracked local shim (see webui/redirect_shim.py) — absent in the
# public repo, so this is a no-op unless the file has been created locally.
try:
    from redirect_shim import install_redirect_shim
    install_redirect_shim(app)
except ImportError:
    pass

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

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "version": MCP_VERSION,
        "processor_ready": processor.init_error is None,
    }), 200


@app.route('/me', methods=['GET'])
def whoami():
    """Report the verified identity (if any) so the frontend can lock the
    username field when running behind an auth provider."""
    identity = current_identity()
    if identity is None:
        return jsonify({"authenticated": False}), 200
    return jsonify({
        "authenticated": True,
        "user_id": identity.sub,
        "username": identity.email or identity.sub,
        "email": identity.email,
        "groups": identity.groups,
    }), 200


@app.route('/query', methods=['POST'])
def api_query():
    resp = '{"status": "error", "message": "Unknown error"}'  # Default error response
    code = 500
    try:
        payload = request.get_json(force=True)
        apply_identity(payload)

        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")

        q = (payload.get("input", "") or "").strip()
        if not q:
            raise ValueError("Empty input")

        req = QueryRequest(input=q, history=payload.get("history", []), user_id=payload.get("user_id"), username=payload.get("username"), session_id=payload.get("session_id"), function_stage=payload.get("function_stage"), ip=request.remote_addr, identity_verified=payload.get("identity_verified", False))
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
        # Capture the client IP while the request context is active; the generator
        # runs outside it and cannot access `request`.
        if isinstance(payload, dict):
            payload['client_ip'] = request.remote_addr
        # Bake the verified identity into the payload while the request context
        # (and g.identity) is still available; generate() runs outside it.
        apply_identity(payload)
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


@app.route('/function_stage', methods=['POST'])
def set_function_stage():
    """Toggle which dynamic-function stage is surfaced to the model: 'dev' or 'prod'.

    'dev' also surfaces in-development functions so they can be iterated before
    publishing; 'prod' (default) shows only published functions.
    """
    try:
        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")
        payload = request.get_json(force=True) or {}
        stage = (payload.get("stage") or "prod").strip()
        resp = processor.set_function_stage(stage)
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
        apply_identity(payload)
        user_id = (payload.get("user_id") or "").strip()
        session_id = (payload.get("session_id") or "").strip()
        history = payload.get("history") or []
        resp = processor.save_pattern(user_id, session_id, history)
        return jsonify(resp.json()), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify(QueryResponse(status="Error", error=str(e)).json()), 500


@app.route('/checkpoint/finalize', methods=['POST'])
def finalize_checkpoint():
    """Fold any remaining live turns into the session's conversation checkpoint
    before the thread is abandoned (e.g. on 'clear chat'). Best-effort, fast."""
    try:
        if processor.init_error:
            raise ValueError(f"Processor initialization failed: {processor.init_error}")
        payload = request.get_json(force=True) or {}
        apply_identity(payload)
        user_id = (payload.get("user_id") or "").strip()
        username = (payload.get("username") or "").strip()
        session_id = (payload.get("session_id") or "").strip()
        history = payload.get("history") or []
        resp = processor.finalize_checkpoint(session_id, user_id, username, history)
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
        apply_identity(payload)
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

        def emit_local(message, status="Processing"):
            if isinstance(message, Exception):
                resp = QueryResponse(status="Error", error=str(message), message=str(message))
            else:
                resp = QueryResponse(status=status, message=str(message))
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

        yield QueryResponse(status='querying', message='Querying Claude with MCP tools...').json() + '\n'
        req = QueryRequest(input=q, history=payload.get("history", []), user_id=payload.get("user_id"), username=payload.get("username"), session_id=payload.get("session_id"), function_stage=payload.get("function_stage"), ip=payload.get("client_ip"), identity_verified=payload.get("identity_verified", False))
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
