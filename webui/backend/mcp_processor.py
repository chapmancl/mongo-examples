from typing import Any, List, Optional, Tuple
import traceback

import mcp_client


class APIQueryProcessor:
    """Lightweight wrapper used by the Flask endpoint.

    Lazily instantiates the full `CachedQueryProcessor` to avoid heavy
    initialization at import time and to surface initialization errors cleanly.
    """

    def __init__(self):
        self._impl: Optional[mcp_client.CachedQueryProcessor] = None
        self._init_error: Optional[Exception] = None

    def _ensure_impl(self) -> None:
        if self._impl is None and self._init_error is None:
            try:
                self._impl = mcp_client.CachedQueryProcessor()
            except Exception as e:
                self._init_error = e

    @property
    def init_error(self) -> Optional[Exception]:
        self._ensure_impl()
        return self._init_error

    def clear_all_caches(self) -> None:
        self._ensure_impl()
        if self._init_error:
            raise RuntimeError(f"Initialization failed: {self._init_error}")
        return self._impl.clear_all_caches()

    def clear_history(self) -> None:
        self._ensure_impl()
        if self._init_error:
            raise RuntimeError(f"Initialization failed: {self._init_error}")
        self._impl.history = None

    def get_history(self) -> Optional[List[Any]]:
        self._ensure_impl()
        if self._init_error:
            return None
        return self._impl.history

    def get_cache_stats(self) -> dict:
        self._ensure_impl()
        if self._init_error:
            raise RuntimeError(f"Initialization failed: {self._init_error}")
        return self._impl.get_cache_stats()

    def query_claude_with_mcp_tools(self, question: str, history: Optional[List[Any]] = None) -> Tuple[str, List[Any]]:
        """Forward the question to the underlying processor and return (answer, history)."""
        self._ensure_impl()
        if self._init_error:
            raise RuntimeError(f"Initialization failed: {self._init_error}")
        return self._impl.query_claude_with_mcp_tools(question, history)
