"""
MemoryService: business logic for the memory subpackage.

Implements: Intake, Recall, Reflect, Query, ListSessions,
            SchemaDeclare, StrategyStore, StrategyRecall.

Design notes:
  - All methods are async; motor (async pymongo) is used throughout.
  - No TurboQuant compression or shard-key routing — those are Go-only concerns.
  - Auto-link on intake: cosine threshold 0.75 via Atlas $vectorSearch.
  - Recall uses composite score: 0.6*vector + 0.3*importance_decayed + 0.1*recency.
  - One-hop graph expansion on recall.
  - Atlas vector search index names are defined as module-level constants
    (must match the indexes created on the cluster).
"""

import uuid
import re
import math
import asyncio
import logging
import datetime
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId

from .mongo_helpers import (
    now_ms, to_ms, format_object_id, normalize_oid_str, get_collection, strip_embedding,
    split_regex_filters, apply_regex_post_filters,
)
from .scope import (
    build_scope_filter, collection_for_scope, shard_key_for_scope,
    SCOPE_USER_SESSION,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Collection / index constants
# ------------------------------------------------------------------
COLLECTION_EPISODIC = "memory_episodic"
COLLECTION_SEMANTIC = "memory_semantic"
COLLECTION_STRATEGIES = "memory_semantic"  # strategies stored as memory_type='strategy'
COLLECTION_SCHEMAS = "memory_semantic"     # schemas stored as memory_type='schema' (flat)

VECTOR_IDX_EPISODIC = "memory_episodic_vector_index"
VECTOR_IDX_STRATEGIES = "memory_semantic_vector_index"
FULLTEXT_IDX_STRATEGIES = "memory_semantic_fulltext_index"

NEAR_DUPLICATE_THRESHOLD = 0.92  # cosine similarity at which we warn of near-duplicate

# Built-in schema for routing patterns stored in memory_semantic (memory_type='strategy').
ROUTING_PATTERN_SCHEMA = {
    "tools":       {"required": True,  "type": "list",   "description": "Tool names selected for this pattern"},
    "query_hints": {"required": False, "type": "list",   "description": "Per-tool input templates"},
    "output_hint": {"required": False, "type": "string", "description": "Expected output JSON skeleton"},
    "playbook":    {"required": False, "type": "string", "description": "Free-text guidance for the LLM"},
    "extends":     {"required": False, "type": "string", "description": "strategy_key of parent pattern to inherit playbook from"},
}


class MemoryService:
    """
    Business logic layer for the memory MCP tools.

    Parameters
    ----------
    db_client:
        A MongoDBClient instance (motor-backed). Its .client attribute
        must be an AsyncIOMotorClient. ensure_connection() is called
        before every operation.
    llm_client:
        A BedrockClient instance that exposes generate_embedding() and
        invoke_bedrock_text().
    memory_db_name:
        The MongoDB database that holds the memory collections.
        Typically comes from settings.memory_db (default "mcp_config").
    """

    def __init__(self, db_client, llm_client, memory_db_name: str, query_embedding_model_id: Optional[str] = None, agent_instructions: str = ""):
        self.db_client = db_client
        self.llm_client = llm_client
        self.memory_db_name = memory_db_name
        self.agent_instructions = agent_instructions
        # Model used for query-side embeddings (recall, strategy_recall).
        # Voyage supports separate query and document models; falls back to
        # the client default (EMBEDDING_MODEL_ID) when not specified.
        self.query_embedding_model_id = query_embedding_model_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _col(self, name: str):
        return get_collection(self.db_client.client, self.memory_db_name, name)

    async def _ensure_connected(self):
        await self.db_client.ensure_connection()

    async def _bump_access_for_docs(
        self,
        docs: List[dict],
        default_collection: str = COLLECTION_EPISODIC,
    ) -> None:
        """Batch bump access metadata for docs, grouped by source collection."""
        if not docs:
            return

        _accessed_at = datetime.datetime.now(datetime.timezone.utc)
        ids_by_collection: Dict[str, List[ObjectId]] = {}
        for doc in docs:
            doc_id = doc.get("_id")
            if not isinstance(doc_id, ObjectId):
                continue
            coll_name = doc.get("_src_col", default_collection)
            ids_by_collection.setdefault(coll_name, []).append(doc_id)

        for coll_name, ids in ids_by_collection.items():
            unique_ids = list(set(ids))
            if not unique_ids:
                continue
            try:
                await self._col(coll_name).update_many(
                    {"_id": {"$in": unique_ids}},
                    {
                        "$inc": {"access_count": 1},
                        "$set": {"last_accessed": _accessed_at},
                    },
                )
            except Exception as exc:
                logger.debug("_bump_access_for_docs skipped for %s: %s", coll_name, exc)

    def _schedule_bump_access(
        self,
        docs: List[dict],
        default_collection: str = COLLECTION_EPISODIC,
    ) -> None:
        """Schedule fire-and-forget access bump for returned docs."""
        if not docs:
            return
        try:
            asyncio.create_task(self._bump_access_for_docs(docs, default_collection=default_collection))
        except RuntimeError:
            pass  # no event loop running (e.g. tests) — skip bump

    def _composite_score(
        self,
        vector_score: float,
        importance: float,
        created_at,
        decay_rate: float = 0.01,
    ) -> float:
        """
        Composite relevance score used for recall ranking.

        Formula:
            0.6 * vector_score
          + 0.3 * (importance * exp(-decay_rate * age_days))
          + 0.1 * (1 / (1 + age_days))

        decay_rate is read per-document so each memory can decay at its own pace.
        created_at may be an int (ms, Python intake) or datetime (Go/BSON intake).
        """
        age_ms = max(0, now_ms() - to_ms(created_at))
        age_days = age_ms / (24.0 * 3600.0 * 1000.0)
        effective_imp = importance * math.exp(-decay_rate * age_days)
        recency = 1.0 / (1.0 + age_days)
        return 0.6 * vector_score + 0.3 * effective_imp + 0.1 * recency

    # ------------------------------------------------------------------
    # Scope visibility helper
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_visible(doc: dict, agent_id: str = "", username: str = "") -> bool:
        """Return True if *doc* is visible to the caller identified by agent_id/username.

        Evaluates the scope int field (new model) with fallback to legacy is_isolated bool.
        """
        from .scope import (
            SCOPE_SHARED, SCOPE_AGENT, SCOPE_USER,
            SCOPE_USER_SESSION, SCOPE_USER_SESSION_AGENT,
        )
        scope = doc.get("scope")
        if scope is None:
            # Legacy doc: visible unless explicitly isolated to a different owner.
            return not doc.get("is_isolated", False)
        if scope == SCOPE_SHARED:
            return True
        if scope == SCOPE_AGENT:
            return bool(agent_id) and doc.get("agent_id") == agent_id
        if scope == SCOPE_USER:
            return bool(username) and doc.get("username") == username
        if scope == SCOPE_USER_SESSION:
            return bool(username) and doc.get("username") == username
        if scope == SCOPE_USER_SESSION_AGENT:
            return (
                bool(username) and doc.get("username") == username
                and bool(agent_id) and doc.get("agent_id") == agent_id
            )
        # Unknown scope: default visible.
        return True

    # ------------------------------------------------------------------
    # Internal schema validation helper
    # ------------------------------------------------------------------

    async def _validate_payload_against_schema(
        self,
        schema_version: str,
        payload: Optional[Dict[str, Any]],
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        Validate payload against a declared schema's required fields.
        Returns a (possibly empty) list of warning strings.
        The write always proceeds regardless of warnings.
        """
        col = self._col(COLLECTION_SCHEMAS)
        # Support both flat 'schema' (legacy) and namespaced 'schema:<name>' (current) format.
        schema_filter: Dict[str, Any] = {
            "memory_type": {"$in": [f"schema:{schema_version}", "schema"]},
            "payload.type_name": schema_version,
        }

        schema_doc = await col.find_one(schema_filter)
        if not schema_doc:
            return [f"schema_version '{schema_version}' not declared; skipping validation"]

        fields = (schema_doc.get("payload") or {}).get("fields", schema_doc.get("fields", {}))
        warnings: List[str] = []
        for field_name, meta in fields.items():
            if isinstance(meta, dict) and meta.get("required"):
                if not payload or field_name not in payload:
                    warnings.append(f"required field '{field_name}' missing from payload")
        return warnings

    # ------------------------------------------------------------------
    # Recall helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collections_for_scope(scope: str) -> List[tuple]:
        """Return [(collection_name, vector_index_name)] pairs for the given scope."""
        if scope == "episodic":
            return [(COLLECTION_EPISODIC, VECTOR_IDX_EPISODIC)]
        if scope == "semantic":
            return [(COLLECTION_SEMANTIC, VECTOR_IDX_STRATEGIES)]
        # "all" (default) — search both
        return [(COLLECTION_EPISODIC, VECTOR_IDX_EPISODIC), (COLLECTION_SEMANTIC, VECTOR_IDX_STRATEGIES)]

    @staticmethod
    def _build_atlas_filter(
        collection: str,
        session_id: Optional[str],
        agent_id: Optional[str],
        memory_types: Optional[List[str]],
        tags: Optional[List[str]],
        importance_threshold: float,
    ) -> Dict[str, Any]:
        """Build Atlas $vectorSearch pre-filter using only declared filterable fields.

        Episodic filterable fields:  session_id, memory_type, importance, tags, is_isolated
        Semantic filterable fields:  agent_id, memory_type, tags, strategy_key, is_isolated

        Cross-agent sharing: memories without is_isolated (or is_isolated=false) are visible
        to all agents. Only explicitly isolated memories are restricted to their owner.
        We do NOT add an agent_id pre-filter on episodic — the Python post-filter in recall()
        handles ownership using the $or [own | not isolated] pattern.
        """
        f: Dict[str, Any] = {}
        if collection == COLLECTION_EPISODIC:
            if session_id:
                f["session_id"] = {"$eq": session_id}
            if memory_types:
                f["memory_type"] = {"$in": memory_types}
            if tags:
                f["tags"] = {"$in": tags}
            if importance_threshold > 0:
                f["importance"] = {"$gte": importance_threshold}
        else:  # memory_semantic
            if memory_types:
                f["memory_type"] = {"$in": memory_types}
            if tags:
                f["tags"] = {"$in": tags}
            # Do NOT add agent_id filter on semantic — cross-agent sharing uses post-filter.
        return f

    async def _run_vector_search(
        self,
        col,
        index_name: str,
        query_vec: List[float],
        limit: int,
        num_candidates: int = 0,
        atlas_filter: Optional[Dict[str, Any]] = None,
        src_col: Optional[str] = None,
        extra_match: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """Run a $vectorSearch pipeline on *col* and return docs with vs_score set.

        Shared by recall(), query(), _check_similar_existing(), and _strategy_vector_only().
        Raises on MongoDB error — callers are responsible for try/except.
        """
        if num_candidates <= 0:
            num_candidates = max(50, limit * 10)
        vs_stage: Dict[str, Any] = {
            "index": index_name,
            "path": "embedding",
            "queryVector": query_vec,
            "numCandidates": num_candidates,
            "limit": limit,
        }
        if atlas_filter:
            vs_stage["filter"] = atlas_filter
        add_fields: Dict[str, Any] = {"vs_score": {"$meta": "vectorSearchScore"}}
        if src_col:
            add_fields["_src_col"] = src_col
        pipeline: List[Dict[str, Any]] = [
            {"$vectorSearch": vs_stage},
            *([{"$match": extra_match}] if extra_match else []),
            {"$addFields": add_fields},
            {"$project": {"embedding": 0}},
        ]
        docs: List[dict] = []
        async for doc in col.aggregate(pipeline):
            docs.append(doc)
        return docs

    async def _run_rerank(
        self,
        query: str,
        docs: List[dict],
        top_k: Optional[int] = None,
        content_field: str = "content",
    ) -> List[dict]:
        """Rerank *docs* against *query* using the Voyage AI reranker.

        Extracts *content_field* from each doc as the document string, calls the
        Voyage AI reranker, and returns docs reordered by descending relevance_score
        with ``rerank_score`` set on each.

        Raises on API error — callers are responsible for try/except.
        """
        if not docs:
            return docs

        texts = [str(doc.get(content_field, "")) for doc in docs]
        results = await self.llm_client.rerank(
            query=query,
            documents=texts,
            top_k=top_k,
        )
        # Map relevance scores back to the original docs by index.
        scored: List[dict] = []
        for result in results:
            idx = result["index"]
            doc = dict(docs[idx])
            doc["rerank_score"] = result.get("relevance_score", 0.0)
            scored.append(doc)
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    async def intake(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        decay_rate: float = 0.01,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        payload: Optional[Dict[str, Any]] = None,
        payload_push: Optional[Dict[str, Any]] = None,
        username: Optional[str] = None,
        agent_id: Optional[str] = None,
        schema_version: Optional[str] = None,
        scope: int = -1,
        related_docs: Optional[List[Dict[str, Any]]] = None,
        _id: Optional[str] = None,
    ) -> dict:
        """Store a memory with optional explicit related_docs links and near-duplicate warning.

        If _id is provided and matches an existing document, the memory is
        updated in-place: content is re-embedded, and content/embedding/tags/
        entities/payload/importance/decay_rate/schema_version are $set on the
        existing document. The _id and related_docs links are preserved so all
        graph connections remain valid.

        payload_push: dict mapping payload field names to values (or lists) to
        append to existing payload arrays using $push. Only used in update
        (_id) mode. Each key must reference an array field; the value is the
        item (or list of items via $each) to append.
        Example: payload_push={"batches_array": {"batch_id": 61, "score": 0.9}}
        """
        await self._ensure_connected()
        effective_scope_for_col = scope if scope >= 0 else SCOPE_USER_SESSION
        target_coll_name = collection_for_scope(effective_scope_for_col)
        col = self._col(target_coll_name)

        embedding = (await self.llm_client.generate_embedding(content))["vector"]

        # --- Replace-in-place when _id is provided ---
        if _id:
            # Accept both bare hex and Go-compatible ObjectID("hex") wrapper.
            raw = _id.strip()
            if len(raw) > 12 and raw[:10].upper() == 'OBJECTID("' and raw[-2:] == '")':
                raw = raw[10:-2]
            try:
                oid = ObjectId(raw)
            except Exception:
                return {"error": f"Invalid _id format: {_id!r}"}

            update_fields: Dict[str, Any] = {
                "content": content,
                "embedding": embedding,
                "memory_type": memory_type,
                "importance": importance,
                "decay_rate": decay_rate,
            }
            if tags is not None:
                update_fields["tags"] = tags
            if entities is not None:
                update_fields["entities"] = entities
            if payload is not None:
                update_fields["payload"] = payload
            if schema_version is not None:
                update_fields["schema_version"] = schema_version

            mongo_update: Dict[str, Any] = {"$set": update_fields}

            # payload_push: append items to payload sub-arrays without reading the doc first.
            # Each key becomes payload.<key> with $push semantics.
            if payload_push:
                push_ops: Dict[str, Any] = {}
                for field, value in payload_push.items():
                    mongo_key = f"payload.{field}"
                    if isinstance(value, list):
                        push_ops[mongo_key] = {"$each": value}
                    else:
                        push_ops[mongo_key] = value
                mongo_update["$push"] = push_ops

            result = await col.update_one({"_id": oid}, mongo_update)
            if result.matched_count == 0:
                # Try the alternate collection — doc may be in semantic even when scope maps to episodic.
                alt_col_name = COLLECTION_SEMANTIC if target_coll_name == COLLECTION_EPISODIC else COLLECTION_EPISODIC
                alt_col = self._col(alt_col_name)
                result = await alt_col.update_one({"_id": oid}, mongo_update)
                if result.matched_count == 0:
                    return {"error": f"No document found with _id: {_id}"}
                target_coll_name = alt_col_name

            schema_warnings: List[str] = []
            if schema_version:
                schema_warnings = await self._validate_payload_against_schema(
                    schema_version=schema_version,
                    payload=payload,
                    agent_id=agent_id or username,
                )

            return {
                "id": format_object_id(oid),
                "collection": target_coll_name,
                "memory_type": memory_type,
                "updated": True,
                "schema_warnings": schema_warnings,
            }

        _now = datetime.datetime.now(datetime.timezone.utc)

        # Caller-supplied explicit links — stored as-is; no automatic vector-search linking.
        explicit_related_docs = list(related_docs) if related_docs else []

        effective_scope = scope if scope >= 0 else SCOPE_USER_SESSION

        doc = {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "decay_rate": decay_rate,
            "tags": tags or [],
            "entities": entities,
            "payload": payload,
            "session_id": session_id,
            "embedding": embedding,
            "created_at": _now,
            "last_accessed": _now,
            "access_count": 0,
            "related_docs": explicit_related_docs,
            "username": username,
            "agent_id": agent_id,
            "schema_version": schema_version,
            "scope": effective_scope,
        }

        result = await col.insert_one(doc)
        new_id = result.inserted_id

        # Parent auto-backlink: if payload contains a parent_task_id that looks like
        # an ObjectId, find that parent document and add this child to its linked_ids
        # and related_docs — so the parent can discover its children without a separate
        # memory_reflect(link) call. Partial failure is acceptable.
        parent_task_id = (payload or {}).get("parent_task_id")
        if parent_task_id:
            try:
                parent_oid = ObjectId(normalize_oid_str(str(parent_task_id)))
                child_link = {"id": str(new_id), "relation": "child_of_parent", "explicit": True}
                parent_link = {"id": str(parent_oid), "relation": "parent_of", "explicit": True}
                for coll_name in [COLLECTION_EPISODIC, COLLECTION_SEMANTIC]:
                    try:
                        await self._col(coll_name).update_one(
                            {"_id": parent_oid},
                            {"$addToSet": {"related_docs": child_link}},
                        )
                    except Exception:
                        pass
                # Also record back-link on the new child doc itself.
                await col.update_one(
                    {"_id": new_id},
                    {"$addToSet": {"related_docs": parent_link}},
                )
                logger.debug("Auto-backlinked child %s → parent %s", str(new_id), str(parent_oid))
            except Exception as exc:
                logger.debug("Parent auto-backlink skipped (parent_task_id not a valid ObjectId or not found): %s", exc)

        # Schema validation (warnings only — write already succeeded).
        schema_warnings: List[str] = []
        if schema_version:
            schema_warnings = await self._validate_payload_against_schema(
                schema_version=schema_version,
                payload=payload,
                agent_id=agent_id or username,
            )

        # Surface near-duplicates so the caller can decide whether to deduplicate.
        similar_existing, has_near_duplicates = await self._check_similar_existing(
            new_doc_id=str(new_id),
            vector=embedding,
            agent_id=agent_id,
            username=username,
        )

        return {
            "id": format_object_id(new_id),
            "collection": target_coll_name,
            "memory_type": memory_type,
            "schema_warnings": schema_warnings,
            "similar_existing": similar_existing,
            "has_near_duplicates": has_near_duplicates,
        }

    async def recall(
        self,
        query: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        username: Optional[str] = None,
        scope: str = "all",
        limit: int = 5,
        num_candidates: int = 150,
        score_threshold: float = 0.0,
        importance_threshold: float = 0.0,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        depth: int = 1,
        depth_relations: Optional[List[str]] = None,
        output_format: str = "default",
    ) -> dict:
        """
        Semantic recall across episodic and/or semantic collections with
        one-hop graph expansion and composite scoring.

        Routing:
          query + entities  → vector search, then entity post-filter (existing path)
          query only        → pure vector search
          entities only     → direct find({entities:{$in:[...]}}) — no vector scan;
                              auto-fallback to vector if zero results (requires query text —
                              returns error if neither query nor entities provided)
          neither           → error

        Steps (vector path):
          1. Embed the query.
          2. Run $vectorSearch on each collection for the given scope, with
             per-collection Atlas pre-filters (only declared filterable fields).
          3. Backstop: for session-scoped queries also fetch all docs directly.
          4. Python post-filter: scope ownership check; entities post-filter.
          5. One-hop graph expansion via related_docs links.
          6. Composite score, filter by threshold, return top-limit.
          7. Fire-and-forget access count bump.
        """
        await self._ensure_connected()

        # --- Routing decision ---
        has_query = bool(query and query.strip())
        has_entities = bool(entities)

        if not has_query and not has_entities:
            return {"error": "At least one of 'query' or 'entities' is required"}

        # Auto-scope schema/strategy queries to semantic — avoids wasteful episodic scans.
        if memory_types and all(mt.startswith(("schema", "strategy")) for mt in memory_types):
            if scope == "all":
                scope = "semantic"

        # --- Entities-only path: direct find, no vector scan ---
        if has_entities and not has_query:
            entity_set = set(entities)  # type: ignore[arg-type]
            owner = agent_id or username
            collections = self._collections_for_scope(scope)
            initial_results: List[dict] = []
            seen_ids: set = set()
            for coll_name, _ in collections:
                col = self._col(coll_name)
                find_filter: Dict[str, Any] = {"entities": {"$in": list(entity_set)}}
                if memory_types:
                    find_filter["memory_type"] = {"$in": memory_types}
                if tags:
                    find_filter["tags"] = {"$in": tags}
                if importance_threshold > 0:
                    find_filter["importance"] = {"$gte": importance_threshold}
                if session_id:
                    find_filter["session_id"] = session_id
                async for doc in col.find(
                    find_filter,
                    projection={"embedding": 0},
                    sort=[("importance", -1)],
                    limit=limit * 4,
                ):
                    if doc["_id"] not in seen_ids:
                        doc["vs_score"] = 0.7
                        doc["_src_col"] = coll_name
                        seen_ids.add(doc["_id"])
                        initial_results.append(doc)
            if owner:
                initial_results = [
                    d for d in initial_results
                    if self._doc_visible(d, agent_id=agent_id or "", username=username or "")
                ]
            if initial_results:
                # Score and return — skip graph expansion on entities-only path.
                scored: List[dict] = []
                for doc in initial_results:
                    composite = self._composite_score(
                        doc.get("vs_score", 0.0),
                        doc.get("importance", 0.5),
                        doc.get("created_at", now_ms()),
                        doc.get("decay_rate", 0.01),
                    )
                    if composite >= score_threshold:
                        doc["score"] = round(composite, 4)
                        scored.append(doc)
                scored.sort(key=lambda d: d["score"], reverse=True)
                top = scored[:limit]
                overflow = scored[limit:]
                results = [strip_embedding(doc, doc.pop("_src_col", COLLECTION_EPISODIC)) for doc in top]
                entity_stubs = [
                    {
                        "id": format_object_id(doc["_id"]),
                        "memory_type": doc.get("memory_type"),
                        "importance": doc.get("importance"),
                        "entities": doc.get("entities") or [],
                        "tags": doc.get("tags") or [],
                        "score": doc.get("score"),
                    }
                    for doc in overflow
                ]
                resp: Dict[str, Any] = {
                    "results": results,
                    "count": len(results),
                    "paths_used": ["entities_filter"],
                    "fallback_used": False,
                }
                if entity_stubs:
                    resp["entity_match_stubs"] = entity_stubs
                    resp["entity_match_stub_count"] = len(entity_stubs)
                return resp
            # Zero results — no query text to fall back with.
            return {
                "results": [],
                "count": 0,
                "paths_used": ["entities_filter"],
                "fallback_used": False,
            }

        # --- Vector path (query present) ---
        query_vec = (await self.llm_client.generate_embedding(query, model_id=self.query_embedding_model_id))["vector"]  # type: ignore[index]
        collections = self._collections_for_scope(scope)

        initial_results = []
        seen_ids = set()
        fallback_used = False
        fallback_reason = None

        for coll_name, idx_name in collections:
            col = self._col(coll_name)
            atlas_filter = self._build_atlas_filter(
                collection=coll_name,
                session_id=session_id,
                agent_id=agent_id,
                memory_types=memory_types,
                tags=tags,
                importance_threshold=importance_threshold,
            )
            try:
                for doc in await self._run_vector_search(
                    col, idx_name, query_vec,
                    limit=limit * 4,
                    num_candidates=max(num_candidates, limit * 10),
                    atlas_filter=atlas_filter or None,
                    src_col=coll_name,
                ):
                    if doc["_id"] not in seen_ids:
                        seen_ids.add(doc["_id"])
                        initial_results.append(doc)
            except Exception as exc:
                logger.warning("vectorSearch on %s failed: %s", coll_name, exc)

        # Backstop: for session-scoped queries, also fetch episodic docs for that
        # session directly so they are included even if they scored below the ANN cutoff.
        # Capped to avoid loading an unbounded number of session docs into memory.
        if session_id:
            _backstop_cap = max(limit * 20, 200)
            col = self._col(COLLECTION_EPISODIC)
            _backstop_count = 0
            async for doc in col.find(
                {"session_id": session_id},
                projection={"embedding": 0},
                sort=[("importance", -1)],  # prefer high-importance docs when cap is hit
                limit=_backstop_cap,
            ):
                if doc["_id"] not in seen_ids:
                    doc["vs_score"] = 0.6  # baseline for direct hits
                    doc["_src_col"] = COLLECTION_EPISODIC
                    seen_ids.add(doc["_id"])
                    initial_results.append(doc)
                    _backstop_count += 1

        # Smart fallback: if memory_types filter returned zero results, automatically
        # retry vector search WITHOUT the type filter in same request.
        if memory_types and not initial_results:
            logger.info(
                "recall: memory_types filter %s returned zero results. Retrying without type filter.",
                memory_types
            )
            initial_results = []
            seen_ids = set()

            for coll_name, idx_name in collections:
                col = self._col(coll_name)
                # Rebuild atlas_filter WITHOUT memory_types constraint
                atlas_filter = self._build_atlas_filter(
                    collection=coll_name,
                    session_id=session_id,
                    agent_id=agent_id,
                    memory_types=None,  # <-- retry without type filter
                    tags=tags,
                    importance_threshold=importance_threshold,
                )
                try:
                    for doc in await self._run_vector_search(
                        col, idx_name, query_vec,
                        limit=limit * 4,
                        num_candidates=max(num_candidates, limit * 10),
                        atlas_filter=atlas_filter or None,
                        src_col=coll_name,
                    ):
                        if doc["_id"] not in seen_ids:
                            seen_ids.add(doc["_id"])
                            initial_results.append(doc)
                except Exception as exc:
                    logger.warning("vectorSearch fallback on %s failed: %s", coll_name, exc)

            fallback_used = True
            fallback_reason = (
                f"No memories found with memory_type filter: {memory_types}. "
                "Returned results from all types. Call memory_get_instructions() to see valid memory_type values."
            )

        # Python post-filter: scope-aware ownership check.
        # A doc is visible based on its scope field; legacy docs fall back to is_isolated.
        owner = agent_id or username
        if owner:
            initial_results = [
                d for d in initial_results
                if self._doc_visible(d, agent_id=agent_id or "", username=username or "")
            ]

        # Python post-filter: entities (not a filterable field in any Atlas index).
        if entities:
            entity_set = set(entities)
            initial_results = [
                d for d in initial_results
                if entity_set.intersection(d.get("entities") or [])
            ]

        # Mark direct hits with hop_depth=0 so results always carry the field.
        for doc in initial_results:
            doc.setdefault("hop_depth", 0)

        # --- Multi-hop BFS graph expansion ---
        # Iterate up to min(depth, 5) hops.  At each hop we follow related_docs edges
        # from the current frontier.  A shared hydration budget (EXPANSION_HYDRATE_LIMIT)
        # limits full document fetches across all hops; overflow docs become lightweight
        # stubs in the response envelope.
        _depth = min(max(depth, 1), 5)
        _expansion_oid_cap = max(limit * 10, 50)
        _expand_seed_cap = limit * 4
        EXPANSION_HYDRATE_LIMIT = 5

        # Frontier for hop-1 = top seeds from initial results.
        frontier_docs = sorted(
            initial_results,
            key=lambda d: d.get("vs_score", 0.0),
            reverse=True,
        )[:_expand_seed_cap]

        total_hydrated = 0
        graph_neighbor_stubs: List[dict] = []
        all_neighbor_oids: List[ObjectId] = []  # global discovery order (for cap)
        bfs_edges: List[dict] = []  # collected for output_format='graph'

        for hop in range(1, _depth + 1):
            if not frontier_docs:
                break

            # Collect neighbor OIDs from this frontier, respecting the global cap.
            hop_oids: List[ObjectId] = []
            oid_to_edge: Dict[ObjectId, dict] = {}  # oid → {source, relation}
            for doc in frontier_docs:
                if len(all_neighbor_oids) + len(hop_oids) >= _expansion_oid_cap:
                    break
                parent_id = format_object_id(doc["_id"])
                for r in (doc.get("related_docs") or []):
                    if len(all_neighbor_oids) + len(hop_oids) >= _expansion_oid_cap:
                        break
                    rel = (r.get("relation", "linked") if isinstance(r, dict) else "linked")
                    if depth_relations and rel not in depth_relations:
                        continue
                    lid = (r.get("id", r) if isinstance(r, dict) else r)
                    try:
                        oid = ObjectId(normalize_oid_str(lid))
                        if oid not in seen_ids:
                            hop_oids.append(oid)
                            seen_ids.add(oid)  # mark immediately — prevents duplicates within hop
                            oid_to_edge[oid] = {"source": parent_id, "relation": rel}
                    except Exception:
                        pass
            # Record edges for graph output
            for oid in hop_oids:
                edge_info = oid_to_edge.get(oid, {})
                bfs_edges.append({
                    "source": edge_info.get("source", ""),
                    "target": format_object_id(oid),
                    "relation": edge_info.get("relation", "linked"),
                    "hop": hop - 1,
                })

            if not hop_oids:
                break

            all_neighbor_oids.extend(hop_oids)

            # Split: hydrate as many as the remaining budget allows; stub the rest.
            remaining_budget = EXPANSION_HYDRATE_LIMIT - total_hydrated
            hydrate_oids = hop_oids[:max(remaining_budget, 0)]
            stub_oids = hop_oids[max(remaining_budget, 0):]

            # Stub the overflow — metadata only, no embedding fetch.
            if stub_oids:
                _stub_projection = {"_id": 1, "memory_type": 1, "importance": 1, "entities": 1, "tags": 1}
                for coll_name, _ in collections:
                    col = self._col(coll_name)
                    async for doc in col.find(
                        {"_id": {"$in": stub_oids}},
                        projection=_stub_projection,
                    ):
                        if owner and not self._doc_visible(doc, agent_id=agent_id or "", username=username or ""):
                            continue
                        graph_neighbor_stubs.append({
                            "id": format_object_id(doc["_id"]),
                            "memory_type": doc.get("memory_type"),
                            "importance": doc.get("importance"),
                            "entities": doc.get("entities") or [],
                            "tags": doc.get("tags") or [],
                            "hop_depth": hop,
                        })

            # Hydrate the budgeted OIDs fully; these extend the frontier for the next hop.
            next_frontier: List[dict] = []
            if hydrate_oids:
                for coll_name, _ in collections:
                    col = self._col(coll_name)
                    async for doc in col.find(
                        {"_id": {"$in": hydrate_oids}},
                        projection={"embedding": 0},
                    ):
                        if owner and not self._doc_visible(doc, agent_id=agent_id or "", username=username or ""):
                            continue
                        # Apply a per-hop score penalty so deeper neighbors rank below
                        # closer ones when they compete for top slots.
                        doc["vs_score"] = max(0.3, 0.5 - (hop - 1) * 0.1)
                        doc["_src_col"] = coll_name
                        doc["hop_depth"] = hop
                        total_hydrated += 1
                        initial_results.append(doc)
                        next_frontier.append(doc)

            # Only hydrated docs can serve as the frontier for the next hop
            # (stubs lack related_docs content).
            frontier_docs = next_frontier

        # --- Composite scoring and filtering ---
        scored: List[dict] = []
        for doc in initial_results:
            composite = self._composite_score(
                doc.get("vs_score", 0.0),
                doc.get("importance", 0.5),
                doc.get("created_at", now_ms()),
                doc.get("decay_rate", 0.01),
            )
            if composite >= score_threshold:
                doc["score"] = round(composite, 4)
                scored.append(doc)

        scored.sort(key=lambda d: d["score"], reverse=True)
        top = scored[:limit]

        # Fire-and-forget access metadata update for returned results.
        self._schedule_bump_access(top)

        results = [strip_embedding(doc, doc.pop("_src_col", COLLECTION_EPISODIC)) for doc in top]
        paths_used = [c for c, _ in collections]

        # --- Graph output format ---
        if output_format == "graph":
            nodes: List[dict] = []
            seen_node_ids: set = set()
            for doc in results:
                nid = doc.get("id") or format_object_id(doc.get("_id", ""))
                if nid in seen_node_ids:
                    continue
                seen_node_ids.add(nid)
                content = doc.get("content", "")
                nodes.append({
                    "id": nid,
                    "label": (doc.get("payload", {}) or {}).get("name") or doc.get("strategy_key") or content[:40] or nid,
                    "hop_depth": doc.get("hop_depth", 0),
                    "memory_type": doc.get("memory_type"),
                    "importance": doc.get("importance", 0.5),
                    "score": doc.get("score", 0.0),
                    "tags": doc.get("tags") or [],
                    "entities": doc.get("entities") or [],
                    "hydrated": True,
                    "content_preview": content[:120],
                })
            for stub in graph_neighbor_stubs:
                nid = stub["id"]
                if nid in seen_node_ids:
                    continue
                seen_node_ids.add(nid)
                nodes.append({
                    "id": nid,
                    "label": stub.get("memory_type") or nid,
                    "hop_depth": stub.get("hop_depth", 1),
                    "memory_type": stub.get("memory_type"),
                    "importance": stub.get("importance", 0.5),
                    "score": 0.0,
                    "tags": stub.get("tags") or [],
                    "entities": stub.get("entities") or [],
                    "hydrated": False,
                    "content_preview": "",
                })
            # Filter edges to only reference nodes we actually have
            valid_ids = seen_node_ids
            edges_out = [e for e in bfs_edges if e["source"] in valid_ids and e["target"] in valid_ids]
            return {
                "jsonDataType": "memory_graph",
                "nodes": nodes,
                "edges": edges_out,
                "query": query,
                "count": len([n for n in nodes if n["hydrated"]]),
                "depth": _depth,
                "paths_used": paths_used,
                "fallback_used": fallback_used,
                "graph_neighbor_count": len(graph_neighbor_stubs),
            }

        response: Dict[str, Any] = {
            "results": results,
            "count": len(results),
            "paths_used": paths_used,
            "fallback_used": fallback_used,
        }
        if fallback_reason:
            response["fallback_reason"] = fallback_reason
        if graph_neighbor_stubs:
            response["graph_neighbors"] = graph_neighbor_stubs
            response["graph_neighbor_count"] = len(graph_neighbor_stubs)
        return response

    async def _check_similar_existing(
        self,
        new_doc_id: str,
        vector: list,
        agent_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> tuple:
        """
        Run a top-5 vector search on both collections to surface near-duplicates.
        Excludes the newly inserted doc by _id.
        Returns (similar_existing_list, has_near_duplicates).
        """
        out: List[dict] = []
        has_near_dup = False
        collections = [
            (COLLECTION_EPISODIC, VECTOR_IDX_EPISODIC),
            (COLLECTION_SEMANTIC, VECTOR_IDX_STRATEGIES),
        ]
        for coll_name, idx_name in collections:
            if len(out) >= 5:
                break
            col = self._col(coll_name)
            try:
                docs = await self._run_vector_search(
                    col, idx_name, vector, limit=6, num_candidates=50,
                )
                for doc in docs:
                    if str(doc["_id"]) == new_doc_id:
                        continue
                    score = doc.get("vs_score", 0.0)
                    out.append({
                        "id": str(doc["_id"]),
                        "content": doc.get("content", ""),
                        "memory_type": doc.get("memory_type", ""),
                        "score": score,
                    })
                    if score >= NEAR_DUPLICATE_THRESHOLD:
                        has_near_dup = True
                    if len(out) >= 5:
                        break
            except Exception as exc:
                logger.warning("_check_similar_existing skipped for %s: %s", coll_name, exc)
        return out, has_near_dup

    async def _link_memories(
        self,
        source_ids: List[str],
        target_ids: Optional[List[str]],
        relation: str = "linked",
        inverse_relation: Optional[str] = None,
    ) -> dict:
        """
        Create explicit directional links between memories, mirroring Go's linkMemories.

        - source docs get: related_docs += [{id: target, relation: relation, explicit: True}]
        - target docs get: related_docs += [{id: source, relation: inverse_relation, explicit: True}]

        If target_ids is empty/None, falls back to symmetric all-pairs on source_ids
        (both directions use the same relation label).
        Sources and targets within the same group are never cross-linked.
        Searches both memory_episodic and memory_semantic.
        """
        if not source_ids:
            return {"error": "link requires at least one memory_id (source)"}
        if not relation:
            relation = "linked"
        if not inverse_relation:
            inverse_relation = relation

        # Symmetric all-pairs fallback.
        symmetric = not target_ids
        if symmetric:
            if len(source_ids) < 2:
                return {"error": "link with no target_ids requires at least 2 memory_ids"}
            target_ids = source_ids
            inverse_relation = relation

        # Validate all IDs up-front.
        def to_oid(s: str) -> Optional[ObjectId]:
            raw = s.strip()
            if raw.upper().startswith('OBJECTID("') and raw.endswith('")'):
                raw = raw[10:-2]
            try:
                return ObjectId(raw)
            except Exception:
                return None

        src_oids = [oid for oid in (to_oid(s) for s in source_ids) if oid]
        tgt_oids = [oid for oid in (to_oid(t) for t in target_ids) if oid]

        collections = [COLLECTION_EPISODIC, COLLECTION_SEMANTIC]

        # Build all update coroutines upfront, then fire them all concurrently.
        async def _do_update(coll_name: str, filter_q: dict, update_q: dict) -> int:
            try:
                res = await self._col(coll_name).update_many(filter_q, update_q)
                return res.modified_count
            except Exception as exc:
                logger.warning("link update failed for %s: %s", coll_name, exc)
                return 0

        tasks = []

        # Forward links: each source → all targets (skip self).
        for src in src_oids:
            fwd_docs = [
                {"id": str(tgt), "relation": relation, "explicit": True}
                for tgt in tgt_oids if tgt != src
            ]
            if not fwd_docs:
                continue
            update = {"$addToSet": {"related_docs": {"$each": fwd_docs}}}
            for coll_name in collections:
                tasks.append(_do_update(coll_name, {"_id": src}, update))

        # Back links: each target → all sources (skip self).
        for tgt in tgt_oids:
            back_docs = [
                {"id": str(src), "relation": inverse_relation, "explicit": True}
                for src in src_oids if src != tgt
            ]
            if not back_docs:
                continue
            update = {"$addToSet": {"related_docs": {"$each": back_docs}}}
            for coll_name in collections:
                tasks.append(_do_update(coll_name, {"_id": tgt}, update))

        counts = await asyncio.gather(*tasks)
        linked = sum(counts)

        return {
            "operation": "link",
            "linked": linked,
            "relation": relation,
            "inverse_relation": inverse_relation,
            "source_count": len(src_oids),
            "target_count": len(tgt_oids),
        }

    async def reflect(
        self,
        session_id: Optional[str] = None,
        operation: str = "summarise",
        memory_ids: Optional[List[str]] = None,
        target_ids: Optional[List[str]] = None,
        link_relation: str = "linked",
        inverse_relation: Optional[str] = None,
        entities: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> dict:
        """
        Multi-operation memory maintenance tool, mirroring Go's Reflect.

        operation="summarise" (default): Summarise all memories for a session via
            LLM and store the result as a 'session:summary' memory.  The summary
            always retains session_id — scope controls visibility, not session_id absence.

        operation="link": Create explicit bidirectional links between memories.
            memory_ids: source IDs; target_ids: target IDs (optional — if omitted,
            symmetric all-pairs linking on memory_ids). Uses related_docs field with
            {id, relation, explicit: True} entries, matching Go format.

        operation="set_entities": Overwrite the top-level entities[] field on the
            documents identified by memory_ids.  Searches both collections.
            entities param is required.
        """
        await self._ensure_connected()

        if operation == "link":
            return await self._link_memories(
                source_ids=memory_ids or [],
                target_ids=target_ids,
                relation=link_relation,
                inverse_relation=inverse_relation,
            )

        if operation == "set_entities":
            if not memory_ids:
                return {"error": "memory_ids is required for operation='set_entities'"}
            if entities is None:
                return {"error": "entities is required for operation='set_entities'"}
            oids = []
            for raw in memory_ids:
                try:
                    oids.append(ObjectId(normalize_oid_str(raw)))
                except Exception:
                    pass
            if not oids:
                return {"error": "No valid ObjectIDs in memory_ids"}
            updated = 0
            for coll_name in [COLLECTION_EPISODIC, COLLECTION_SEMANTIC]:
                try:
                    res = await self._col(coll_name).update_many(
                        {"_id": {"$in": oids}},
                        {"$set": {"entities": entities}},
                    )
                    updated += res.modified_count
                except Exception as exc:
                    logger.warning("set_entities update failed for %s: %s", coll_name, exc)
            return {
                "operation": "set_entities",
                "updated": updated,
                "memory_ids": memory_ids,
                "entities": entities,
            }

        # --- Default: summarise ---
        if not session_id:
            return {"error": "session_id is required for operation='summarise'"}

        col = self._col(COLLECTION_EPISODIC)

        docs: List[dict] = []
        async for doc in col.find(
            {"session_id": session_id},
            projection={"embedding": 0},
            sort=[("created_at", 1)],
            limit=100,
        ):
            docs.append(doc)

        if not docs:
            return {"summary": "", "session_id": session_id, "memories_reflected": 0}

        memory_text = "\n".join(
            f"[{d.get('memory_type', 'memory')}] {d.get('content', '')}" for d in docs
        )
        prompt = (
            f"Summarize the following session memories for session '{session_id}' "
            f"into a concise session summary capturing key facts, decisions, and progress:\n\n"
            f"{memory_text}"
        )

        summary_text = await self.llm_client.invoke_bedrock_text(prompt)

        if not summary_text:
            return {"error": "LLM summarisation failed — empty response", "session_id": session_id, "memories_reflected": len(docs)}

        source_ids_for_link = [str(d["_id"]) for d in docs]
        intake_result = await self.intake(
            content=summary_text,
            memory_type="session:summary",
            importance=0.9,
            session_id=session_id,
            tags=["reflection", "summary"],
            agent_id=agent_id,
            username=username,
        )
        summary_id = intake_result["id"]

        # Bidirectionally link the summary to all source memories.
        if source_ids_for_link:
            try:
                await self._link_memories(
                    source_ids=[summary_id],
                    target_ids=source_ids_for_link,
                    relation="summarises",
                    inverse_relation="summarised_by",
                )
            except Exception as exc:
                logger.warning("reflect bidir link skipped: %s", exc)

        # Stamp promoted_at on source docs when they exist.
        if source_ids_for_link:
            _promoted_now = datetime.datetime.now(datetime.timezone.utc)
            src_oids = []
            for sid in source_ids_for_link:
                try:
                    src_oids.append(ObjectId(normalize_oid_str(sid)))
                except Exception:
                    pass
            if src_oids:
                try:
                    await col.update_many(
                        {
                            "_id": {"$in": src_oids},
                            "$or": [{"promoted_at": {"$exists": False}}, {"promoted_at": None}],
                        },
                        {"$set": {"promoted_at": _promoted_now}},
                    )
                except Exception as exc:
                    logger.warning("promoted_at stamp skipped: %s", exc)

        return {
            "summary": summary_text,
            "session_id": session_id,
            "memories_reflected": len(docs),
            "summary_id": summary_id,
        }

    async def query(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        scope: str = "episodic",
        sort_by: str = "created_at",
        sort_dir: str = "desc",
        agent_id: Optional[str] = None,
        username: Optional[str] = None,
        query: Optional[str] = None,
        ids: Optional[List[str]] = None,
    ) -> dict:
        """Query memories by filter, scope (episodic|strategies), and sort.

        When ids=[...] is provided, bypasses all scope/ownership filters and
        fetches by _id only — caller already knows the exact documents.
        When 'query' is provided, uses $rankFusion (vector + fulltext) for
        strategies or $vectorSearch for episodic. Falls back to plain find
        when no query string is given.
        Cross-agent sharing: results include owned docs AND any non-isolated doc.
        """
        await self._ensure_connected()

        # --- Direct ID fetch: bypass scope filter entirely ---
        if ids:
            oid_list: List[ObjectId] = []
            for raw_id in ids:
                try:
                    oid_list.append(ObjectId(normalize_oid_str(raw_id)))
                except Exception:
                    pass
            if not oid_list:
                return {"results": [], "count": 0}
            extra_filter = {k: v for k, v in (filter or {}).items() if k != "_id"}
            id_filter = {"_id": {"$in": oid_list}, **extra_filter}
            raw_results: List[dict] = []
            for coll_name in [COLLECTION_EPISODIC, COLLECTION_SEMANTIC]:
                col = self._col(coll_name)
                async for doc in col.find(id_filter, projection={"embedding": 0}):
                    doc["_src_col"] = coll_name
                    raw_results.append(doc)
            self._schedule_bump_access(raw_results)
            results = [strip_embedding(doc, doc.get("_src_col", COLLECTION_EPISODIC)) for doc in raw_results]
            return {"results": results, "count": len(results)}

        mongo_filter = filter or {}
        owner = agent_id or username

        # Determine target collection.
        if "session_id" in mongo_filter:
            col_name = COLLECTION_EPISODIC
        elif scope == "strategies":
            col_name = COLLECTION_STRATEGIES
        else:
            col_name = COLLECTION_EPISODIC

        col = self._col(col_name)

        # --- Semantic search path ---
        if query:
            query_vec = (await self.llm_client.generate_embedding(query, model_id=self.query_embedding_model_id))["vector"]

            if scope == "strategies":
                candidates = await self._strategy_rank_fusion(col, query, query_vec, limit, tags=None)
                if candidates is None:
                    candidates = await self._strategy_vector_only(col, query_vec, limit, tags=None)
            else:
                candidates = []
                try:
                    candidates = await self._run_vector_search(
                        col, VECTOR_IDX_EPISODIC, query_vec,
                        limit=limit * 3,
                    )
                except Exception as exc:
                    logger.warning("vectorSearch in query() failed: %s", exc)

            if owner:
                candidates = [
                    d for d in candidates
                    if self._doc_visible(d, agent_id=agent_id or "", username=username or "")
                ]

            top_docs = candidates[:limit]
            for d in top_docs:
                d.setdefault("_src_col", col_name)
            self._schedule_bump_access(top_docs, default_collection=col_name)
            results = [strip_embedding(d, col_name) for d in top_docs]
            response = {"results": results, "count": len(results)}
            if not results and scope != "strategies":
                response["info"] = (
                    "No results found in episodic memory. "
                    "memory_query only searches episodic memory by default — "
                    "try memory_recall to search semantic (long-term) memories."
                )
            return response

        # --- Plain find path ---
        if owner and "agent_id" not in mongo_filter and "username" not in mongo_filter:
            mongo_filter = {
                **mongo_filter,
                "$or": build_scope_filter(
                    agent_id=agent_id or "",
                    username=username or "",
                    session_id="",
                ),
            }

        if col_name == COLLECTION_STRATEGIES and "memory_type" not in mongo_filter:
            mongo_filter = {**mongo_filter, "memory_type": "strategy"}

        sort_direction = -1 if sort_dir == "desc" else 1
        raw_results: List[dict] = []
        async for doc in col.find(
            mongo_filter,
            projection={"embedding": 0},
            sort=[(sort_by, sort_direction)],
            limit=limit,
        ):
            doc["_src_col"] = col_name
            raw_results.append(doc)

        self._schedule_bump_access(raw_results, default_collection=col_name)
        results = [strip_embedding(doc, col_name) for doc in raw_results]

        response = {"results": results, "count": len(results)}
        if not results and scope != "strategies":
            response["info"] = (
                "No results found in episodic memory. "
                "memory_query only searches episodic memory by default — "
                "try memory_recall to search semantic (long-term) memories."
            )
        return response

    async def list_sessions(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> dict:
        """
        List sessions using a $group aggregation that returns memory_count
        and last_updated_at per session_id.

        Any $regex conditions in the filter are stripped before the aggregation
        (Atlas does not support $regex in $match stages on certain indexes) and
        applied as a Python post-filter on the result set.
        """
        await self._ensure_connected()
        col = self._col(COLLECTION_EPISODIC)

        # Separate $regex conditions so they don't reach Atlas.
        base_filter, regex_filters = split_regex_filters(filter or {})
        session_filter = {**base_filter, "session_id": {"$ne": None}}

        pipeline = [
            {"$match": session_filter},
            {
                "$group": {
                    "_id": "$session_id",
                    "memory_count": {"$sum": 1},
                    "last_updated_at": {"$max": "$created_at"},
                    "last_memory_type": {"$last": "$memory_type"},
                }
            },
            {"$sort": {"last_updated_at": -1}},
            {"$limit": limit},
        ]

        sessions: List[dict] = []
        async for doc in col.aggregate(pipeline):
            sessions.append({
                "session_id": doc["_id"],
                "memory_count": doc["memory_count"],
                "last_updated_at": doc["last_updated_at"],
                "last_memory_type": doc.get("last_memory_type"),
                "collection": COLLECTION_EPISODIC,
            })

        # Apply in-memory regex post-filters stripped above.
        if regex_filters:
            sessions = apply_regex_post_filters(sessions, regex_filters)

        return {"sessions": sessions, "count": len(sessions)}

    async def schema_declare(
        self,
        schema_name: str,
        fields: Dict[str, Any],
        content: Optional[str] = None,
        version: str = "1.0",
        agent_id: Optional[str] = None,
    ) -> dict:
        """
        Upsert a schema definition in 'memory_semantic' (memory_type='schema' flat).

        Matches GoMCP flat format: memory_type='schema', lookup by payload.type_name.
        Fields stored inside payload.fields alongside type_name and version.
        The content string is embedded for semantic search.
        """
        await self._ensure_connected()
        col = self._col(COLLECTION_SCHEMAS)
        embed_content = content or f"Schema definition for {schema_name} (version {version})"

        embedding = (await self.llm_client.generate_embedding(embed_content))["vector"]
        memory_type_value = f"schema:{schema_name}"
        doc = {
            "schema_name": schema_name,
            "memory_type": memory_type_value,
            "content": embed_content,
            "embedding": embedding,
            "payload": {
                "type_name": schema_name,
                "version": version,
                "fields": fields,
            },
            "agent_id": agent_id,
            "tags": ["schema", schema_name, f"v{version}"],
            "created_at": datetime.datetime.now(datetime.timezone.utc),
        }
        result = await col.update_one(
            {"memory_type": {"$in": [memory_type_value, "schema"]}, "payload.type_name": schema_name},
            {"$set": doc},
            upsert=True,
        )
        # Retrieve the _id whether this was an insert or an update.
        inserted_id = result.upserted_id
        if inserted_id is None:
            existing = await col.find_one(
                {"memory_type": memory_type_value, "payload.type_name": schema_name},
                projection={"_id": 1},
            )
            inserted_id = existing["_id"] if existing else None
        return {
            "id": format_object_id(inserted_id) if inserted_id else None,
            "schema_name": schema_name,
            "memory_type": memory_type_value,
            "agent_id": agent_id,
            "upserted": result.upserted_id is not None,
            "modified": result.modified_count > 0,
        }

    async def _get_most_recent_strategy(self, name: str, limit: int = 1) -> Union[Optional[dict], List[dict]]:
        """Return the most recent strategy document(s) for the given name.

        limit=1 (default): uses find_one — returns Optional[dict].
        limit>1: uses find cursor — returns List[dict], sorted by version_seq desc.
        """
        col = self._col(COLLECTION_STRATEGIES)
        if limit == 1:
            return await col.find_one({"strategy_key": name}, sort=[("version_seq", -1)], projection={"embedding": 0})
        else:
            docs: List[dict] = []
            async for doc in col.find(
                {"strategy_key": name},
                projection={"embedding": 0},
                sort=[("version_seq", -1)],
                limit=limit,
            ):
                docs.append(doc)
            return docs

    async def strategy_store(
        self,
        name: str,
        context: str,
        _id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        username: Optional[str] = None,
        agent_id: Optional[str] = None,
        scope: int = -1,
        session_id: Optional[str] = None,
        importance: float = 0.90,
        decay_rate: float = 0.001,
        memory_type: str = "strategy",
        payload: Optional[Dict[str, Any]] = None,
        schema_version: Optional[str] = None,
        superseded_by: Optional[str] = None,
        related_doc_ids: Optional[List[str]] = None,
        link_relation: str = "related",
    ) -> dict:
        """
        Store a new strategy version in memory_semantic, or update an existing one.

        _id supplied  → in-place update: re-embeds context into content/embedding,
                        replaces payload if provided. version_seq and history unchanged.
        _id absent    → always inserts a new document. version_seq is auto-incremented
                        from the most recent doc with this name. The previous doc's
                        superseded_by is updated to point to the new document
                        (best-effort; version_seq is authoritative).
        superseded_by: manual override only — normally managed by code.
        """
        await self._ensure_connected()
        col = self._col(COLLECTION_STRATEGIES)

        # --- In-place update path ---
        if _id:
            try:
                oid = ObjectId(normalize_oid_str(_id))
            except Exception:
                return {"error": f"Invalid _id format: {_id!r}"}
            # Promote schema_version from payload if not passed at top level.
            if schema_version is None and payload and isinstance(payload.get("schema_version"), str):
                schema_version = payload["schema_version"]
            embedding = (await self.llm_client.generate_embedding(context))["vector"]
            update_fields: Dict[str, Any] = {
                "content": context,
                "embedding": embedding,
            }
            if payload is not None:
                update_fields["payload"] = payload
            if tags is not None:
                update_fields["tags"] = tags
            if entities is not None:
                update_fields["entities"] = entities
            if username is not None:
                update_fields["username"] = username
            if agent_id is not None:
                update_fields["agent_id"] = agent_id
            if scope >= 0:
                update_fields["scope"] = scope
            if superseded_by is not None:
                update_fields["superseded_by"] = superseded_by
            if schema_version is not None:
                update_fields["schema_version"] = schema_version
            result = await col.update_one({"_id": oid}, {"$set": update_fields})
            if result.matched_count == 0:
                return {"error": f"No strategy found with _id: {_id}"}
            response: Dict[str, Any] = {"id": format_object_id(oid), "name": name, "action": "updated"}
            if related_doc_ids:
                link_result = await self._link_memories(
                    source_ids=[str(oid)],
                    target_ids=related_doc_ids,
                    relation=link_relation,
                )
                response["links"] = link_result
            return response

        # --- Insert new version path ---
        # Find the current most recent version for version_seq and backlink update.
        prev_doc = await self._get_most_recent_strategy(name)
        prev_version_seq = prev_doc.get("version_seq") if prev_doc else None
        new_version_seq = (prev_version_seq or 0) + 1
        # Carry forward the previous version's hit_count so recall history is preserved.
        inherited_hit_count = int(prev_doc.get("hit_count") or 0) if prev_doc else 0

        # Prepend name to tags so BM25 can match by strategy name.
        caller_tags = list(tags or [])
        merged_tags: List[str] = [name] + [t for t in caller_tags if t != name]

        # Merge tool names from payload into tags for the fulltext/BM25 leg.
        if payload and isinstance(payload.get("tools"), list):
            for t in payload["tools"]:
                if isinstance(t, str) and t not in merged_tags:
                    merged_tags.append(t)

        # If schema_version wasn't passed as a top-level param, promote it from payload.
        if schema_version is None and payload and isinstance(payload.get("schema_version"), str):
            schema_version = payload["schema_version"]

        effective_scope = scope if scope >= 0 else SCOPE_USER_SESSION
        # Shared-scope strategies are permanent reference docs — boost importance, no decay.
        if effective_scope == 0:  # SCOPE_SHARED
            importance = 0.98
            decay_rate = 0.0
        embedding = (await self.llm_client.generate_embedding(context))["vector"]
        new_doc = {
            "strategy_key": name,            # top-level canonical key
            "version_seq": new_version_seq,  # auto-incremented; highest = most recent
            "superseded_by": None,            # always None on insert; code sets after insert
            "memory_type": memory_type,
            "importance": importance,
            "decay_rate": decay_rate,
            "content": context,
            "tags": merged_tags,
            "entities": entities or [],
            "username": username,
            "agent_id": agent_id,
            "scope": effective_scope,
            "embedding": embedding,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "session_id": session_id,
            "payload": payload,
            "schema_version": schema_version,
            "hit_count": inherited_hit_count,
        }
        result = await col.insert_one(new_doc)
        new_id = result.inserted_id

        # Best-effort: stamp previous doc's superseded_by and reset its hit_count to 0.
        if prev_doc is not None:
            prev_oid = prev_doc.get("_id")
            if prev_oid:
                try:
                    await col.update_one(
                        {"_id": prev_oid},
                        {"$set": {"superseded_by": str(new_id), "hit_count": 0}},
                    )
                except Exception as exc:
                    logger.warning("strategy_store: superseded_by update failed for %s: %s", prev_oid, exc)

        #schema_warnings: List[str] = []
        #if schema_version == "routing_pattern" and merged_payload:
        #    for field, meta in ROUTING_PATTERN_SCHEMA.items():
        #        if meta.get("required") and field not in merged_payload:
        #            schema_warnings.append(f"required field '{field}' missing from payload")

        insert_response: Dict[str, Any] = {
            "id": format_object_id(new_id),
            "name": name,
            "version_seq": new_version_seq,
            "tags": merged_tags,
            "action": "inserted",
            "previous_id": format_object_id(prev_doc["_id"]) if prev_doc else None,
        }
        if related_doc_ids:
            link_result = await self._link_memories(
                source_ids=[str(new_id)],
                target_ids=related_doc_ids,
                relation=link_relation,
            )
            insert_response["links"] = link_result
        return insert_response

    async def strategy_recall(
        self,
        query: Optional[str] = None,
        name: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.5,
        tags: Optional[List[str]] = None,
        include_history: bool = False,
    ) -> dict:
        """
        Recall strategies by exact name lookup or semantic search.

        Matches GoMCP flat interface: memory_type='strategy' (no prefix).

        Exact name lookup: returns the matching strategy document directly.

        Semantic search:
          1. Try $rankFusion (vector 0.7 + BM25 on content+tags 0.3).
          2. Fall back to $vectorSearch-only if $rankFusion is unavailable.
          3. Post-filter: only docs with memory_type == 'strategy'.
          4. Scoring: 0.6*position_score + 0.4*tag_overlap_ratio + hit_count log-norm boost.
          5. Skip documents with hit_count < 0 (negative suppression).
          6. If a result has payload.extends, fetch the parent strategy and
             merge its playbook into the result as parent_playbook.
          7. Increment hit_count on the top result.
        """
        await self._ensure_connected()
        col = self._col(COLLECTION_STRATEGIES)

        if name:
            if include_history:
                # Return all versions sorted by version_seq desc.
                results: List[dict] = []
                raw_results: List[dict] = []
                for doc in await self._get_most_recent_strategy(name, limit=limit):
                    doc["_src_col"] = COLLECTION_STRATEGIES
                    raw_results.append(doc)
                    results.append(strip_embedding(doc, COLLECTION_STRATEGIES))
                self._schedule_bump_access(raw_results, default_collection=COLLECTION_STRATEGIES)
                return {"strategies": results, "results": results, "count": len(results)}
            else:
                # Return only the most recent version (highest version_seq).
                doc = await self._get_most_recent_strategy(name)
                if not doc:
                    return {"strategies": [], "results": [], "count": 0}
                doc["_src_col"] = COLLECTION_STRATEGIES
                self._schedule_bump_access([doc], default_collection=COLLECTION_STRATEGIES)
                stripped = strip_embedding(doc, COLLECTION_STRATEGIES)
                return {"strategies": [stripped], "results": [stripped], "count": 1}

        if not query:
            # No query text — exact tag match (or plain find if no tags).
            # BM25 ($search) is avoided here: hyphens and other punctuation in tag
            # values are tokenized as delimiters, making exact tag lookup unreliable.
            tag_filter: Dict[str, Any] = {}
            if tags:
                tag_filter["tags"] = {"$in": tags}
            results: List[dict] = []
            raw_results: List[dict] = []
            async for doc in col.find(
                tag_filter,
                projection={"embedding": 0},
                sort=[("hit_count", -1), ("created_at", -1)],
                limit=limit,
            ):
                doc["_src_col"] = COLLECTION_STRATEGIES
                raw_results.append(doc)
                results.append(strip_embedding(doc, COLLECTION_STRATEGIES))
            self._schedule_bump_access(raw_results, default_collection=COLLECTION_STRATEGIES)
            return {"strategies": results, "results": results, "count": len(results)}

        query_vec = (await self.llm_client.generate_embedding(query, model_id=self.query_embedding_model_id))["vector"]
        candidates = await self._strategy_rank_fusion(col, query, query_vec, limit, tags)
        if candidates is None or len(candidates) == 0:
            logger.info("$rankFusion unavailable for strategy_recall, falling back to vector-only search")
            # $rankFusion unavailable — fall back to vector-only
            candidates = await self._strategy_vector_only(col, query_vec, limit, tags)

        # Post-filter: accept flat 'strategy'
        # Scoring: 0.6*position + 0.4*tag_overlap_ratio + hit_count log-norm boost.
        n = len(candidates)
        query_tags_set = set(tags) if tags else set()
        scored: List[dict] = []
        for rank, doc in enumerate(candidates):
            #mt = doc.get("memory_type", "")
            #if mt != "strategy":
            #    continue
            hit_count = doc.get("hit_count", 0)
            if hit_count < 0:
                continue
            position_score = 1.0 - (rank / max(n, 1))
            tag_ratio = 0.0
            if query_tags_set:
                doc_tags = set(doc.get("tags") or [])
                tag_ratio = len(query_tags_set & doc_tags) / len(query_tags_set)
            boost = (math.log(hit_count + 1) / math.log(101)) * 0.2
            doc["boosted_score"] = round(0.6 * position_score + 0.4 * tag_ratio + boost, 4)
            if doc["boosted_score"] >= similarity_threshold:
                scored.append(doc)

        scored.sort(key=lambda d: d["boosted_score"], reverse=True)
        top = scored[:limit]
        for doc in top:
            doc.setdefault("_src_col", COLLECTION_STRATEGIES)

        # --- Resolve each unique strategy_key to its latest version (Option C) ---
        # Candidates are already sorted by score desc, so first occurrence of each key
        # carries the highest score for that key.
        key_scores: Dict[str, float] = {}
        key_order: List[str] = []
        for doc in top:
            key = doc.get("strategy_key")
            if key and key not in key_scores:
                key_scores[key] = doc.get("boosted_score", 0.0)
                key_order.append(key)

        if key_scores:
            async def _resolve_tip(key: str, score: float) -> Optional[dict]:
                tip = await self._get_most_recent_strategy(key)
                if tip:
                    tip["_src_col"] = COLLECTION_STRATEGIES
                    tip["boosted_score"] = score
                return tip

            tip_results = await asyncio.gather(
                *[_resolve_tip(k, key_scores[k]) for k in key_order]
            )
            resolved = [t for t in tip_results if t is not None]
            top = sorted(
                resolved,
                key=lambda d: d.get("boosted_score", 0.0),
                reverse=True,
            )[:limit]

        # Increment hit_count on the top result's tip doc (non-fatal).
        if top:
            try:
                top_id = top[0].get("_id")
                if top_id:
                    await col.update_one(
                        {"_id": top_id},
                        {"$inc": {"hit_count": 1}},
                    )
            except Exception as exc:
                logger.warning("hit_count increment failed: %s", exc)

        # Resolve parent playbook for results with payload.extends.
        # This whole section needs to change - use related docs and some relation there...
        results: List[dict] = []
        for doc in top:
            stripped = strip_embedding(doc, COLLECTION_STRATEGIES)
            # strategy_id: prefer top-level strategy_key, fall back to payload.name (GoMCP) or legacy name.
            stripped["strategy_id"] = doc.get("strategy_key", "")
            extends_key = (doc.get("payload") or {}).get("extends")
            if extends_key:
                parent = await col.find_one(
                    {"strategy_key": extends_key},
                    projection={"payload": 1, "_id": 0},
                )
                if parent:
                    parent_playbook = (parent.get("payload") or {}).get("playbook")
                    if parent_playbook:
                        stripped["parent_playbook"] = parent_playbook
            results.append(stripped)

        self._schedule_bump_access(top, default_collection=COLLECTION_STRATEGIES)

        return {"strategies": results, "results": results, "count": len(results), "query": query}

    async def _strategy_rank_fusion(
        self,
        col,
        question: str,
        query_vec: List[float],
        limit: int,
        tags: Optional[List[str]] = None,
    ) -> Optional[List[dict]]:
        """$rankFusion combining vector (0.7) + BM25 on content+tags (0.3).

        No memory_type filter in vectorSearch (Atlas doesn't support $regex there).
        Post-filtering by memory_type == 'strategy' happens in the caller.
        Tags filter is applied as a $match stage after $vectorSearch (avoids requiring
        tags as a filterable field in the Atlas vector index).
        """
        vector_pipeline: List[Dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": VECTOR_IDX_STRATEGIES,
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": max(50, limit * 10),
                    "limit": limit * 5,
                    "filter": {"memory_type": "strategy"},
                }
            }
            #*,([{"$match": {"tags": {"$in": tags}}}] if tags else []),
        ]
        text_pipeline: List[Dict[str, Any]] = [
            {
                "$search": {
                    "index": FULLTEXT_IDX_STRATEGIES,
                    "text": {"query": question, "path": ["content", "tags"]},
                }
            },
            {"$match": {"memory_type": "strategy"}},
            {"$limit": limit * 5},
        ]
        pipeline: List[Dict[str, Any]] = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {"vector": vector_pipeline, "text": text_pipeline}
                    },
                    "combination": {"weights": {"vector": 0.7, "text": 0.3}},
                }
            },
            *([{"$addFields": {"_tag_match": {"$gt": [{"$size": {"$ifNull": [{"$setIntersection": ["$tags", tags]}, []]}}, 0]}}}] if tags else []),
            *([ {"$sort": {"_tag_match": -1}}] if tags else []),
            {"$limit": limit * 3},
            {"$project": {"embedding": 0}},
        ]
        try:
            docs: List[dict] = []
            async for doc in col.aggregate(pipeline):
                docs.append(doc)
            return docs
        except Exception as exc:
            err = str(exc)
            if "rankFusion" in err or "Unrecognized pipeline stage" in err:
                logger.warning("$rankFusion unavailable on strategies, falling back to vector-only: %s", exc)
                return None
            logger.warning("$rankFusion on strategies failed: %s", exc)
            return None

    async def _strategy_vector_only(
        self,
        col,
        query_vec: List[float],
        limit: int,
        tags: Optional[List[str]] = None,
    ) -> List[dict]:
        """Vector-only fallback for strategy recall."""
        return await self._run_vector_search(
            col, VECTOR_IDX_STRATEGIES, query_vec,
            limit=limit * 5,
            atlas_filter={"memory_type": "strategy"},
            extra_match={"tags": {"$in": tags}} if tags else None,
        )

    # HTML comment regex — mirrors Go's (?s)<!--.*?-->
    _HTML_COMMENT_RE = re.compile(r'(?s)<!--.*?-->')

    async def get_instructions(self) -> dict:
        """
        Assemble agent operating instructions from three sources (mirrors get_instructions.go):
          1. Static base (agent_instructions field) — HTML comments stripped.
          2. Latest agent:blueprint from memory_semantic — appended as a section.
          3. Live memory_type inventory — distinct types across both collections.
        All DB sources fail silently so a cold-start (empty DB) always returns something.
        """
        # 1. Base — strip HTML comments.
        base = self._HTML_COMMENT_RE.sub("", self.agent_instructions or "").strip()

        # 2. Blueprint section.
        blueprint_section = ""
        try:
            await self._ensure_connected()
            col = self._col(COLLECTION_SEMANTIC)
            doc = await self._get_most_recent_strategy("master_instructions")
            if doc:
                content = (doc.get("content") or "").strip()
                if content:
                    blueprint_section = "\n\n---\n\n## Agent Blueprint\n\n" + content
        except Exception as exc:
            logger.warning("get_instructions: failed to fetch blueprint: %s", exc)

        instructions = base + blueprint_section
        logger.debug("get_instructions called: returning %d chars", len(instructions))
        return {"instructions": instructions}
