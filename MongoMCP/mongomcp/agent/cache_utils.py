import hashlib
import json
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional


class SimpleCache:
    """In-memory cache with TTL support. Mirrors the MongoSessionCache async interface
    so the two classes are interchangeable in CachedQueryProcessor."""

    def __init__(
        self,
        settings: Any,
        username: str,
        session_id: str,
        cache_object_name: str = "tool_discovery",
    ):
        self.cache_object_name = cache_object_name
        self.username = username
        self.session_id = session_id
        self._default_ttl: int = getattr(settings, "CACHE_TTL", 300)
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def reset_connection(self) -> None:
        """No-op — satisfies the MongoSessionCache interface."""

    async def get(self, key: str) -> Any:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, timestamp, ttl = entry
            if time.time() - timestamp < ttl:
                return value
            del self._cache[key]
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        cache_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            self._cache[key] = (value, time.time(), cache_ttl)

    async def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    async def remove_pattern(self, pattern: str) -> int:
        """Remove keys containing pattern. Returns count of removed entries."""
        with self._lock:
            keys = [k for k in self._cache if pattern in k]
            for k in keys:
                del self._cache[k]
            return len(keys)

    @staticmethod
    def create_cache_key(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Create a deterministic cache key from tool name and input."""
        sorted_input = json.dumps(tool_input, sort_keys=True, default=str)
        input_hash = hashlib.md5(sorted_input.encode("utf-8")).hexdigest()
        return f"{tool_name}:{input_hash}"

    async def get_or_compute(self, cache_key: str, compute: Callable[[], Awaitable[Any]],
                             on_cache_hit: Optional[Callable[[], None]] = None,
                             on_cache_miss: Optional[Callable[[], None]] = None) -> Any:
        """Resolve a cached value or compute and store it."""
        cached_result = await self.get(cache_key)
        if cached_result is not None:
            if on_cache_hit:
                on_cache_hit()
            return cached_result
        if on_cache_miss:
            on_cache_miss()
        result = await compute()
        await self.set(cache_key, result)
        return result


def create_cache_key(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Module-level alias for SimpleCache.create_cache_key — kept for backward compatibility."""
    return SimpleCache.create_cache_key(tool_name, tool_input)
