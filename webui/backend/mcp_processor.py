from typing import Any, List, Optional, Tuple
import traceback
from pydantic import BaseModel
from typing import Optional, List, Any
import mcp_client
import queue
import threading

class QueryResponse(BaseModel):
    answer: Optional[str] = None
    error: Optional[str] = None
    status: Optional[str] = None
    history: Optional[List[Any]] = None
    cache_stats: Optional[dict] = None
    message: Optional[str] = None

    def json(self):
        if self.answer is not None:
            self.answer = self._sanitize_obj(self.answer)
        if self.history is not None:
            self.history = self._sanitize_obj(self.history)
        if self.cache_stats is not None:
            self.cache_stats = self._sanitize_obj(self.cache_stats)
        if self.message is not None:
            print(self.message)
            self.message = self._sanitize_obj(self.message)
        return self.model_dump_json()

    def _sanitize_obj(self, o):
        """Recursively remove newline characters from string fields in an object."""
        if isinstance(o, dict):
            return {k: self._sanitize_obj(v) for k, v in o.items()}
        if isinstance(o, list):
            return [self._sanitize_obj(v) for v in o]
        if isinstance(o, str):
            return o.replace('\\n', '').replace('\\r', '')
        return o

class QueryRequest(BaseModel):
    input: str
    history: Optional[List[Any]] = None

class StreamingProcessor(mcp_client.CachedQueryProcessor):
    def __init__(self, handler: callable):
        super().__init__()
        self.message_handler = handler

class APIQueryProcessor:
    """Lightweight wrapper used by the Flask endpoint.

    Lazily instantiates the full `CachedQueryProcessor` to avoid heavy
    initialization at import time and to surface initialization errors cleanly.
    """

    def __init__(self):
        self._impl: Optional[mcp_client.CachedQueryProcessor] = None
        self._init_error: Optional[Exception] = None
        self._message_queue: queue.Queue = queue.Queue()

    def _ensure_impl(self) -> None:
        if self._impl is None and self._init_error is None:
            try:
                self._impl = StreamingProcessor(self._handle_message)
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
        return QueryResponse(status="Clear History", message="History cleared", history=[])

    def get_history(self) -> QueryResponse:
        self._ensure_impl()
        return QueryResponse(status="Get History", message="Completed", history=self._impl.history)

    def get_cache_stats(self) -> QueryResponse:
        self._ensure_impl()
        return QueryResponse(status="Get Cache stats", message="Completed", cache_stats=self._impl.get_cache_stats(), history=self._impl.history)

    def query_claude_with_mcp_tools(self, request: QueryRequest) -> QueryResponse:
        """Forward the question to the underlying processor and return (answer, history)."""
        self._ensure_impl()
        answer, history = self._impl.query_claude_with_mcp_tools(request.input, request.history)
        return QueryResponse(status="Query Completed",message="Completed", answer=answer, history=history)
