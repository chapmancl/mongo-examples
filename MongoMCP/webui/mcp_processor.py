from typing import Any, List, Optional, Tuple
import traceback
import json
import re
from pydantic import BaseModel
from typing import Optional, List, Any
#from local_settings import settings # change this to use AWS settings 
from aws_settings import settings
from mongoagent.cached_query_processor import CachedQueryProcessor
import queue
import threading

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
        return QueryResponse(status="Clear History", message="History cleared", history=[])

    def get_history(self) -> QueryResponse:
        self._ensure_impl()
        return QueryResponse(status="Get History", message="Completed", history=self._impl.history)

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
        answer, jsondata, history = self._impl.query_with_mcp_tools(request.input, request.history)
        content = {"text": answer, "jsondata": jsondata}
        return QueryResponse(status="Query Completed", message="Completed", content=content, history=history)
