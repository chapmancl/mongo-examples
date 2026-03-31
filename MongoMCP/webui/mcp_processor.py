from typing import Any, List, Optional, Tuple
import traceback
import json
import re
from pydantic import BaseModel
from typing import Optional, List, Any
#from local_settings import settings # change this to use AWS settings 
from aws_settings import settings
from mongomcp.agent.cached_query_processor import CachedQueryProcessor
import queue
import threading
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryResponse(BaseModel):
    content: Optional[dict] = None
    error: Optional[str] = None
    status: Optional[str] = None
    history: Optional[List[Any]] = None
    cache_stats: Optional[dict] = None
    message: Optional[str] = None

    def json(self):
        if self.message is not None:
            self.message = self.message.replace('\n', ' ').replace('\r', '')
        return self.model_dump_json()

class QueryRequest(BaseModel):
    input: str
    history: Optional[List[Any]] = None

class APIQueryProcessor:
    """Lightweight wrapper used by the Flask endpoint.

    Lazily instantiates the full `CachedQueryProcessor` to avoid heavy
    initialization at import time and to surface initialization errors cleanly.
    """

    def __init__(self):
        self._init_error: Optional[Exception] = None
        self._message_queue: queue.Queue = queue.Queue()
        self._history_cleared: bool = False
        logging.info(f"Loading CachedQueryProcessor with endpoint: {settings.mongo_mcp_root}")
        self._impl: CachedQueryProcessor = CachedQueryProcessor(settings, self._handle_message)

    def _ensure_impl(self) -> None:
        if self._impl is None and self._init_error is None:
            try:
                self._impl = CachedQueryProcessor(settings, self._handle_message)
                self._init_error = None                
            except Exception as e:
                self._init_error = e
                raise RuntimeError(f"Initialization failed: {self._init_error}")


    @property
    def init_error(self) -> Optional[Exception]:
        self._ensure_impl()
        return self._init_error

    def _handle_message(self, message, status="Processing") -> None:
        """Handle incoming messages from the server and queue them for streaming."""
        if isinstance(message, Exception):
            resp = QueryResponse(status="Error", error="Error from server. see message for details", message=str(message))
        else:
            resp = QueryResponse(status=status, message=str(message))
        self._message_queue.put(resp.json())

    def pop_queued_messages(self) -> List[str]:
        """Extract and clear all queued messages (non-blocking)."""
        messages = []
        try:
            while True:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
        except queue.Empty:
            pass
        return messages
    
    def read_message_stream(self, timeout=0.1):
        """Generator that yields queued messages with optional timeout."""
        try:
            while True:
                try:
                    msg = self._message_queue.get(timeout=timeout)
                    yield msg
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error reading message stream: {e}")

    def clear_all_caches(self) -> None:
        self._ensure_impl()            
        str_resp = self._impl.clear_all_caches()
        return QueryResponse(status="Clear Caches", message="Completed", history=self._impl.history)

    def clear_history(self) -> QueryResponse:
        self._ensure_impl()
        self._impl.history = None
        self._history_cleared = True
        # Drain any stale messages so they don't replay on the next query
        try:
            while True:
                self._message_queue.get_nowait()
        except queue.Empty:
            pass
        return QueryResponse(status="Clear History", message="History cleared", history=[])

    def get_history(self) -> QueryResponse:
        self._ensure_impl()
        ui_history = self._trim_history_for_ui(self._impl.history)
        return QueryResponse(status="Get History", message="Completed", history=ui_history)

    def get_cache_stats(self) -> QueryResponse:
        self._ensure_impl()
        return QueryResponse(status="Get Cache stats", message="Completed", cache_stats=self._impl.get_cache_stats(), history=self._impl.history)

    def get_mcp_config(self) -> dict:
        self._ensure_impl()
        return {
            "endpoints": self._impl.mcp_endpoints or [],
            "collection_info": self._impl.mongo_collection_info or {},
        }

    def query_with_mcp_tools(self, request: QueryRequest) -> QueryResponse:
        """Forward the question to the underlying processor and return (answer, history)."""
        self._ensure_impl()
        # If history was explicitly cleared, do not let the client re-send stale history.
        history = None if self._history_cleared else request.history
        self._history_cleared = False
        answer, jsondata, history = self._impl.query_with_mcp_tools(request.input, history)
        content = {"text": answer, "jsondata": jsondata}
        return QueryResponse(status="Query Completed", message="Completed", content=content, history=history)

    @staticmethod
    def _trim_history_for_ui(history: Optional[List[Any]], max_text_len: int = 2000) -> Optional[List[Any]]:
        """Return a shallow copy of history with oversized tool-result text truncated.

        Only affects toolResult content blocks — user/assistant text is left intact.
        The server-side history (used for Bedrock context) is untouched.
        """
        if not history:
            return history
        trimmed = []
        for msg in history:
            content = msg.get("content")
            if not isinstance(content, list):
                trimmed.append(msg)
                continue
            needs_trim = any(
                isinstance(block, dict)
                and "toolResult" in block
                and any(
                    isinstance(c, dict) and len(c.get("text", "")) > max_text_len
                    for c in block["toolResult"].get("content", [])
                )
                for block in content
            )
            if not needs_trim:
                trimmed.append(msg)
                continue
            new_content = []
            for block in content:
                if isinstance(block, dict) and "toolResult" in block:
                    tr = block["toolResult"]
                    new_parts = []
                    for c in tr.get("content", []):
                        if isinstance(c, dict) and "text" in c and len(c["text"]) > max_text_len:
                            new_parts.append({"text": c["text"][:max_text_len] + f"... [truncated {len(c['text'])} chars]"})
                        else:
                            new_parts.append(c)
                    new_content.append({"toolResult": {**tr, "content": new_parts}})
                else:
                    new_content.append(block)
            trimmed.append({**msg, "content": new_content})
        return trimmed

    def save_pattern(self) -> QueryResponse:
        """Persist the routing pattern from the last query into the pattern cache."""
        self._ensure_impl()
        saved = self._impl.save_pattern()
        if saved:
            return QueryResponse(status="Pattern Saved", message="Pattern saved successfully")
        return QueryResponse(status="No Pattern", message="No pattern available to save")
