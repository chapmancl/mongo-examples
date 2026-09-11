import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional
from .mongodb_client import MongoDBClient

class MongoSessionCache:
    """MongoDB-backed cache scoped to a username and session ID.

    Two storage modes controlled by ``per_document``:

    * ``per_document=False`` (default) — **embedded mode**.  All entries for a
      ``(username, session_id)`` pair live as subdocuments inside a single
      MongoDB document.  Efficient for caches with a small number of entries
      (e.g. tool discovery, which stores exactly one entry).

    * ``per_document=True`` — **flat mode**.  Each cache entry is its own
      document, keyed by ``(username, session_id, cache_key)``.  Required for
      caches whose individual values can be large (e.g. tool responses from
      vector search or aggregation) to avoid hitting the 16 MB document limit.
    """

    def __init__(
        self,
        settings: Any,
        username: str,
        session_id: str,
        cache_object_name: str = "tool_discovery",
        per_document: bool = False,
    ):
        if not username:
            raise ValueError("username is required")
        if not session_id:
            raise ValueError("session_id is required")

        self.cache_object_name = cache_object_name
        self.username = username
        self.session_id = session_id
        self.per_document = per_document
        self._collection_name = "mcp_cache"
        local_settings = settings
        local_settings.mcp_config_col = "mcp_cache" # Override collection name for cache
        self._mongo_client = MongoDBClient(settings=local_settings)
        self._default_ttl = getattr(local_settings, "CACHE_TTL", 300)
        self._indexes_initialized = False

    @property
    def _session_filter(self) -> Dict[str, str]:
        """Filter for embedded-mode docs: scoped to (username, session_id, cache_object_name, doc_type)."""
        return {
            "doc_type": "embedded",
            "username": self.username,
            "session_id": self.session_id,
            "cache_object_name": self.cache_object_name,
        }

    def _entry_filter(self, key: str) -> Dict[str, str]:
        """Filter for a single entry in flat (per_document) mode."""
        return {
            "username": self.username,
            "session_id": self.session_id,
            "cache_object_name": self.cache_object_name,
            "cache_key": key,
        }

    @staticmethod
    def _cache_slot(key: str) -> str:
        """Store cache entries under hash slots to keep subdocument field names valid."""
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def reset_connection(self) -> None:
        """Drop the current Motor client so the next _get_collection() call
        creates a fresh AsyncIOMotorClient bound to the current event loop.
        Must be called at the start of any new asyncio.run() that uses this cache.
        """
        self._mongo_client._connection_initialized = False
        self._mongo_client.client = {}
        self._mongo_client.db = {}
        self._mongo_client.collections = {}
        self._indexes_initialized = False

    async def _get_collection(self):
        await self._mongo_client.ensure_connection()
        collection = self._mongo_client.get_collection(self._collection_name)

        if not self._indexes_initialized:
            if self.per_document:
                # One document per cache entry — index on the entry key.
                await collection.create_index(
                    [("username", 1), ("session_id", 1),
                     ("cache_object_name", 1), ("cache_key", 1)],
                    unique=True,
                    name="mcp_cache_entry_unique",
                )
                # TTL index — MongoDB auto-deletes expired entries.
                await collection.create_index(
                    [("expires_at", 1)],
                    expireAfterSeconds=0,
                    name="mcp_cache_entry_ttl",
                )
            else:
                # One document per (username, session_id, cache_object_name) — all entries embedded.
                # Partial filter uses doc_type="embedded" discriminator (Atlas supports $eq in partial filters).
                await collection.create_index(
                    [("username", 1), ("session_id", 1), ("cache_object_name", 1)],
                    unique=True,
                    partialFilterExpression={"doc_type": {"$eq": "embedded"}},
                    name="mcp_cache_embedded_unique",
                )
            self._indexes_initialized = True

        return collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any:
        collection = await self._get_collection()

        if self.per_document:
            doc = await collection.find_one(self._entry_filter(key), {"value": 1, "timestamp": 1, "ttl": 1, "_id": 0})
            if not doc:
                return None
            timestamp = float(doc.get("timestamp", 0))
            ttl = int(doc.get("ttl", self._default_ttl))
            if time.time() - timestamp >= ttl:
                await self.delete(key)
                return None
            return doc.get("value")

        # Embedded mode
        slot = self._cache_slot(key)
        projection = {f"{self.cache_object_name}.cache.{slot}": 1, "_id": 0}
        doc = await collection.find_one(self._session_filter, projection)
        if not doc:
            return None

        cache_root = doc.get(self.cache_object_name, {})
        cache_doc = cache_root.get("cache", {})
        entry = cache_doc.get(slot)
        if not isinstance(entry, dict):
            return None

        if entry.get("key") != key:
            # Extremely unlikely hash collision safety check.
            return None

        timestamp = float(entry.get("timestamp", 0))
        ttl = int(entry.get("ttl", self._default_ttl))
        if time.time() - timestamp >= ttl:
            await self.delete(key)
            return None

        return entry.get("value")

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        collection = await self._get_collection()
        cache_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()

        if self.per_document:
            expires_at = datetime.fromtimestamp(now + cache_ttl, tz=timezone.utc)
            await collection.update_one(
                self._entry_filter(key),
                {
                    "$set": {
                        "value": value,
                        "timestamp": now,
                        "ttl": cache_ttl,
                        "expires_at": expires_at,
                        "updated_at": datetime.now(timezone.utc),
                    },
                    "$setOnInsert": {
                        "doc_type": "entry",
                        "username": self.username,
                        "session_id": self.session_id,
                        "cache_object_name": self.cache_object_name,
                        "cache_key": key,
                        "created_at": datetime.now(timezone.utc),
                    },
                },
                upsert=True,
            )
            return

        # Embedded mode
        slot = self._cache_slot(key)
        entry = {"key": key, "value": value, "timestamp": now, "ttl": cache_ttl}
        await collection.update_one(
            self._session_filter,
            {
                "$setOnInsert": {
                    "doc_type": "embedded",
                    "username": self.username,
                    "session_id": self.session_id,
                    "cache_object_name": self.cache_object_name,
                    "started_at": datetime.now(timezone.utc),
                },
                "$set": {
                    f"{self.cache_object_name}.cache.{slot}": entry,
                    f"{self.cache_object_name}.updated_at": datetime.now(timezone.utc),
                },
            },
            upsert=True,
        )

    async def delete(self, key: str) -> None:
        collection = await self._get_collection()

        if self.per_document:
            await collection.delete_one(self._entry_filter(key))
            return

        slot = self._cache_slot(key)
        await collection.update_one(
            self._session_filter,
            {
                "$unset": {f"{self.cache_object_name}.cache.{slot}": ""},
                "$set": {f"{self.cache_object_name}.updated_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )

    async def clear(self) -> None:
        collection = await self._get_collection()

        if self.per_document:
            await collection.delete_many({
                "username": self.username,
                "session_id": self.session_id,
                "cache_object_name": self.cache_object_name,
            })
            return

        await collection.update_one(
            self._session_filter,
            {
                "$setOnInsert": {
                    "doc_type": "embedded",
                    "username": self.username,
                    "session_id": self.session_id,
                    "cache_object_name": self.cache_object_name,
                },
                "$set": {
                    f"{self.cache_object_name}.cache": {},
                    f"{self.cache_object_name}.updated_at": datetime.now(timezone.utc),
                },
            },
            upsert=True,
        )

    async def remove_pattern(self, pattern: str) -> int:
        """Remove cached entries where the original key contains pattern."""
        collection = await self._get_collection()

        if self.per_document:
            result = await collection.delete_many({
                "username": self.username,
                "session_id": self.session_id,
                "cache_object_name": self.cache_object_name,
                "cache_key": {"$regex": pattern},
            })
            return result.deleted_count

        doc = await collection.find_one(self._session_filter, {f"{self.cache_object_name}.cache": 1, "_id": 0})
        if not doc:
            return 0

        cache_root = doc.get(self.cache_object_name, {})
        cache_doc = cache_root.get("cache", {})
        fields_to_unset: Dict[str, str] = {}
        removed = 0

        for slot, entry in cache_doc.items():
            original_key = ""
            if isinstance(entry, dict):
                original_key = str(entry.get("key", ""))
            if pattern in original_key:
                fields_to_unset[f"{self.cache_object_name}.cache.{slot}"] = ""
                removed += 1

        if fields_to_unset:
            await collection.update_one(
                self._session_filter,
                {
                    "$unset": fields_to_unset,
                    "$set": {f"{self.cache_object_name}.updated_at": datetime.now(timezone.utc)},
                },
                upsert=True,
            )

        return removed

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
    """Module-level alias for MongoSessionCache.create_cache_key — kept for backward compatibility."""
    return MongoSessionCache.create_cache_key(tool_name, tool_input)
