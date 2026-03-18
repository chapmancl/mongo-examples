import hashlib
import json
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional


class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, Any] = {}
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._cache:
                data, timestamp, ttl = self._cache[key]
                if time.time() - timestamp < ttl:
                    return data
                del self._cache[key]
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        cache_ttl = ttl or self._default_ttl
        with self._lock:
            self._cache[key] = (value, time.time(), cache_ttl)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def remove_pattern(self, pattern: str) -> None:
        """Remove keys containing a pattern."""
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]


def create_cache_key(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Create a deterministic cache key from tool name and input."""
    sorted_input = json.dumps(tool_input, sort_keys=True, default=str)
    input_hash = hashlib.md5(sorted_input.encode("utf-8")).hexdigest()
    return f"{tool_name}:{input_hash}"

def get_or_compute(
    cache: SimpleCache,
    cache_key: str,
    compute: Callable[[], Any],
    on_cache_hit: Optional[Callable[[], None]] = None,
    on_cache_miss: Optional[Callable[[], None]] = None,
) -> Any:
    """Resolve a cached value or compute and store it."""
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        if on_cache_hit:
            on_cache_hit()
        return cached_result

    if on_cache_miss:
        on_cache_miss()
    result = compute()
    cache.set(cache_key, result)
    return result


async def get_or_compute_async(
    cache: SimpleCache,
    cache_key: str,
    compute: Callable[[], Awaitable[Any]],
    on_cache_hit: Optional[Callable[[], None]] = None,
    on_cache_miss: Optional[Callable[[], None]] = None,
) -> Any:
    """Async variant of get_or_compute."""
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        if on_cache_hit:
            on_cache_hit()
        return cached_result

    if on_cache_miss:
        on_cache_miss()
    result = await compute()
    cache.set(cache_key, result)
    return result
