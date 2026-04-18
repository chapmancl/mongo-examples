"""
MemoryService: business logic for the memory subpackage.

Implements: Intake, Recall, Reflect, Query, ListSessions,
            SchemaDeclare, StrategyStore, StrategyRecall.

Design notes:
  - All methods are async; motor (async pymongo) is used throughout.
  - No TurboQuant compression — compression_ratio is always 0.0.
  - Auto-link on intake: cosine threshold 0.75 via Atlas $vectorSearch.
  - Recall uses composite score: 0.6*vector + 0.3*importance_decayed + 0.1*recency.
  - One-hop graph expansion on recall.
  - Atlas vector search index names are defined as module-level constants
    (must match the indexes created on the cluster).
"""

import uuid
import math
import asyncio
import logging
from typing import Any, Dict, List, Optional

from bson import ObjectId

from .mongo_helpers import (
    now_ms, to_ms, format_object_id, get_collection, strip_embedding,
    split_regex_filters, apply_regex_post_filters,
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

AUTO_LINK_THRESHOLD = 0.75

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
        # Flat memory_type='schema'; lookup by payload.type_name — cross-agent shared.
        schema_filter: Dict[str, Any] = {
            "memory_type": "schema",
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
        username: Optional[str] = None,
        agent_id: Optional[str] = None,
        schema_version: Optional[str] = None,
        is_isolated: bool = False,
    ) -> dict:
        """Store a memory and auto-link to similar existing memories."""
        await self._ensure_connected()
        col = self._col(COLLECTION_EPISODIC)

        embedding = (await self.llm_client.generate_embedding(content))["vector"]
        shard_key = f"{memory_type}|{uuid.uuid4()}"

        # --- Auto-link: find similar existing memories via vector search ---
        linked_ids: List[str] = []
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": VECTOR_IDX_EPISODIC,
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": 50,
                        "limit": 10,
                    }
                },
                {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}}},
                {"$match": {"vs_score": {"$gte": AUTO_LINK_THRESHOLD}}},
                {"$project": {"_id": 1}},
            ]
            async for doc in col.aggregate(pipeline):
                linked_ids.append(str(doc["_id"]))
        except Exception as exc:
            # Index may not exist yet (e.g. first run); treat as no links.
            logger.warning("Auto-link vectorSearch skipped: %s", exc)

        shard_key = f"{memory_type}|{uuid.uuid4()}"

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
            "created_at": now_ms(),
            "shard_key": shard_key,
            "shard_appended": True,
            "linked_ids": linked_ids,
            "username": username,
            "agent_id": agent_id or username,  # mirror Go: agent_id is the ownership field
            "schema_version": schema_version,
            "is_isolated": is_isolated,
        }

        result = await col.insert_one(doc)
        new_id = result.inserted_id

        # Back-link: add new ID to the linked_ids of similar memories.
        if linked_ids:
            linked_oids = []
            for lid in linked_ids:
                try:
                    linked_oids.append(ObjectId(lid))
                except Exception:
                    pass
            if linked_oids:
                await col.update_many(
                    {"_id": {"$in": linked_oids}},
                    {"$addToSet": {"linked_ids": str(new_id)}},
                )

        # Schema validation (warnings only — write already succeeded).
        schema_warnings: List[str] = []
        if schema_version:
            schema_warnings = await self._validate_payload_against_schema(
                schema_version=schema_version,
                payload=payload,
                agent_id=agent_id or username,
            )

        return {
            "id": format_object_id(new_id),
            "collection": COLLECTION_EPISODIC,
            "memory_type": memory_type,
            "shard_key": shard_key,
            "shard_appended": True,
            "compression_ratio": 0.0,  # No TurboQuant in Python implementation
            "schema_warnings": schema_warnings,
        }

    async def recall(
        self,
        query: str,
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
    ) -> dict:
        """
        Semantic recall across episodic and/or semantic collections with
        one-hop graph expansion and composite scoring.

        Steps:
          1. Embed the query.
          2. Run $vectorSearch on each collection for the given scope, with
             per-collection Atlas pre-filters (only declared filterable fields).
          3. Backstop: for session-scoped queries also fetch all docs directly.
          4. Python post-filter: agent_id / username ownership check on episodic;
             entities filter (not Atlas-filterable).
          5. One-hop graph expansion via linked_ids or related_docs links.
          6. Composite score, filter by threshold, return top-limit.
          7. Fire-and-forget access count bump.
        """
        await self._ensure_connected()

        query_vec = (await self.llm_client.generate_embedding(query, model_id=self.query_embedding_model_id))["vector"]
        collections = self._collections_for_scope(scope)

        initial_results: List[dict] = []
        seen_ids: set = set()

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
            vs_stage: Dict[str, Any] = {
                "index": idx_name,
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": max(num_candidates, limit * 10),
                "limit": limit * 4,  # over-fetch for graph expansion + re-scoring
            }
            if atlas_filter:
                vs_stage["filter"] = atlas_filter

            pipeline = [
                {"$vectorSearch": vs_stage},
                {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}, "_src_col": coll_name}},
                {"$project": {"embedding": 0}},
            ]
            try:
                async for doc in col.aggregate(pipeline):
                    if doc["_id"] not in seen_ids:
                        seen_ids.add(doc["_id"])
                        initial_results.append(doc)
            except Exception as exc:
                logger.warning("vectorSearch on %s failed: %s", coll_name, exc)

        # Backstop: for session-scoped queries, also fetch all episodic docs for that
        # session directly so they are included even if they scored below the ANN cutoff.
        if session_id:
            col = self._col(COLLECTION_EPISODIC)
            async for doc in col.find(
                {"session_id": session_id},
                projection={"embedding": 0},
            ):
                if doc["_id"] not in seen_ids:
                    doc["vs_score"] = 0.6  # baseline for direct hits
                    doc["_src_col"] = COLLECTION_EPISODIC
                    seen_ids.add(doc["_id"])
                    initial_results.append(doc)

        # Python post-filter: cross-agent ownership scoping.
        # A doc is visible if EITHER:
        #   (a) it belongs to this agent/user, OR
        #   (b) it is not isolated (is_isolated absent or False) — shared by default.
        # This mirrors Go's buildMemoryFilters $or pattern.
        owner = agent_id or username
        if owner:
            initial_results = [
                d for d in initial_results
                if d.get("username") == owner
                or d.get("agent_id") == owner
                or not d.get("is_isolated", False)
            ]

        # Python post-filter: entities (not a filterable field in any Atlas index).
        if entities:
            entity_set = set(entities)
            initial_results = [
                d for d in initial_results
                if entity_set.intersection(d.get("entities") or [])
            ]

        # --- One-hop graph expansion ---
        expansion_oids: List[ObjectId] = []
        for doc in initial_results:
            # Support both field names: linked_ids (Python intake) and related_docs (Go intake)
            link_ids = doc.get("linked_ids") or []
            related = doc.get("related_docs") or []
            for r in related:
                link_ids.append(r.get("id", r) if isinstance(r, dict) else r)
            for lid in link_ids:
                try:
                    oid = ObjectId(str(lid))
                    if oid not in seen_ids:
                        expansion_oids.append(oid)
                except Exception:
                    pass

        if expansion_oids:
            for coll_name, _ in collections:
                col = self._col(coll_name)
                async for doc in col.find(
                    {"_id": {"$in": expansion_oids}},
                    projection={"embedding": 0},
                ):
                    if doc["_id"] not in seen_ids:
                        doc["vs_score"] = 0.5  # lower baseline for graph neighbors
                        doc["_src_col"] = coll_name
                        seen_ids.add(doc["_id"])
                        initial_results.append(doc)

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

        # Fire-and-forget: bump access_count for returned results.
        async def _bump_access():
            for doc in top:
                coll_name = doc.get("_src_col", COLLECTION_EPISODIC)
                col = self._col(coll_name)
                try:
                    await col.update_one({"_id": doc["_id"]}, {"$inc": {"access_count": 1}})
                except Exception:
                    pass

        try:
            asyncio.ensure_future(_bump_access())
        except RuntimeError:
            pass  # no event loop running (e.g. tests) — skip bump

        results = [strip_embedding(doc, doc.pop("_src_col", COLLECTION_EPISODIC)) for doc in top]
        paths_used = [c for c, _ in collections]
        return {"results": results, "count": len(results), "paths_used": paths_used}

    async def reflect(self, session_id: str, keep_session: bool = False) -> dict:
        """
        Summarise all memories for a session via LLM and store the summary
        as a 'session:summary' memory.

        keep_session: when True, the promoted summary retains the session_id
                      (useful for multi-session continuity); when False the
                      summary is stored without session_id so it acts as a
                      long-term semantic memory independent of any session.
        """
        await self._ensure_connected()
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

        intake_result = await self.intake(
            content=summary_text,
            memory_type="session:summary",
            importance=0.9,
            session_id=session_id if keep_session else None,
            tags=["reflection", "summary"],
        )
        return {
            "summary": summary_text,
            "session_id": session_id,
            "memories_reflected": len(docs),
            "summary_id": intake_result["id"],
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
    ) -> dict:
        """Query memories by filter, scope (episodic|strategies), and sort.

        When 'query' is provided, uses $rankFusion (vector + fulltext) for
        strategies or $vectorSearch for episodic. Falls back to plain find
        when no query string is given.
        Cross-agent sharing: results include owned docs AND any non-isolated doc.
        """
        await self._ensure_connected()

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
                pipeline: List[Dict[str, Any]] = [
                    {
                        "$vectorSearch": {
                            "index": VECTOR_IDX_EPISODIC,
                            "path": "embedding",
                            "queryVector": query_vec,
                            "numCandidates": max(50, limit * 10),
                            "limit": limit * 3,
                        }
                    },
                    {"$project": {"embedding": 0}},
                ]
                candidates = []
                try:
                    async for doc in col.aggregate(pipeline):
                        candidates.append(doc)
                except Exception as exc:
                    logger.warning("vectorSearch in query() failed: %s", exc)

            if owner:
                candidates = [
                    d for d in candidates
                    if d.get("agent_id") == owner
                    or d.get("username") == owner
                    or not d.get("is_isolated", False)
                ]

            results = [strip_embedding(d, col_name) for d in candidates[:limit]]
            return {"results": results, "count": len(results)}

        # --- Plain find path ---
        if owner and "agent_id" not in mongo_filter and "username" not in mongo_filter:
            mongo_filter = {
                **mongo_filter,
                "$or": [
                    {"agent_id": owner},
                    {"username": owner},
                    {"is_isolated": {"$ne": True}},
                ],
            }

        if col_name == COLLECTION_STRATEGIES and "memory_type" not in mongo_filter:
            mongo_filter = {**mongo_filter, "memory_type": "strategy"}

        sort_direction = -1 if sort_dir == "desc" else 1
        results: List[dict] = []
        async for doc in col.find(
            mongo_filter,
            projection={"embedding": 0},
            sort=[(sort_by, sort_direction)],
            limit=limit,
        ):
            results.append(strip_embedding(doc, col_name))

        return {"results": results, "count": len(results)}

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
        doc = {
            "schema_name": schema_name,
            "memory_type": "schema",
            "content": embed_content,
            "embedding": embedding,
            "payload": {
                "type_name": schema_name,
                "version": version,
                "fields": fields,
            },
            "agent_id": agent_id,
            "tags": ["schema", schema_name, f"v{version}"],
            "created_at": now_ms(),
        }
        result = await col.update_one(
            {"memory_type": "schema", "payload.type_name": schema_name},
            {"$set": doc},
            upsert=True,
        )
        return {
            "schema_name": schema_name,
            "agent_id": agent_id,
            "upserted": result.upserted_id is not None,
            "modified": result.modified_count > 0,
        }

    async def strategy_store(
        self,
        name: str,
        description: str,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        pipeline_template: Optional[Dict[str, Any]] = None,
        filter_template: Optional[Dict[str, Any]] = None,
        shard_keys: Optional[List[str]] = None,
        metadata_requirements: Optional[List[str]] = None,
        payload: Optional[Dict[str, Any]] = None,
        schema_version: Optional[str] = None,
    ) -> dict:
        """
        Upsert a strategy/pattern by name into memory_semantic.

        memory_type is flat 'strategy' — GoMCP flat convention.
        Name is prepended to tags for BM25 searchability.
        GoMCP pipeline/filter/shard fields are merged into payload alongside
        any caller-supplied payload (e.g. routing_pattern fields: tools, playbook).
        hit_count is initialised to 0 on first insert and never reset on update.
        """
        await self._ensure_connected()
        col = self._col(COLLECTION_STRATEGIES)

        # Prepend name to tags so BM25 can match by strategy name.
        caller_tags = list(tags or [])
        merged_tags: List[str] = [name] + [t for t in caller_tags if t != name]

        # Merge tool names from payload into tags for the fulltext/BM25 leg.
        if payload and isinstance(payload.get("tools"), list):
            for t in payload["tools"]:
                if isinstance(t, str) and t not in merged_tags:
                    merged_tags.append(t)

        # Build GoMCP-compatible payload, merging caller extras on top.
        merged_payload: Dict[str, Any] = {
            "name": name,
            "pipeline_template": pipeline_template or {},
            "filter_template": filter_template or {},
            "shard_keys": shard_keys or [],
            "metadata_requirements": metadata_requirements or [],
        }
        if payload:
            merged_payload.update(payload)

        embedding = (await self.llm_client.generate_embedding(description))["vector"]
        set_doc = {
            "strategy_key": name,        # internal field for upsert + exact lookup
            "memory_type": "strategy",   # flat — GoMCP flat convention
            "content": description,
            "tags": merged_tags,
            "embedding": embedding,
            "created_at": now_ms(),
            "session_id": session_id,
            "payload": merged_payload,
            "schema_version": schema_version,
        }
        # hit_count: initialise to 0 on insert only ($setOnInsert), never overwrite.
        result = await col.update_one(
            {"strategy_key": name},
            {
                "$set": set_doc,
                "$setOnInsert": {"hit_count": 0},
            },
            upsert=True,
        )

        if result.upserted_id:
            oid = result.upserted_id
        else:
            found = await col.find_one({"strategy_key": name}, projection={"_id": 1})
            oid = found["_id"] if found else None

        schema_warnings: List[str] = []
        if schema_version == "routing_pattern" and merged_payload:
            for field, meta in ROUTING_PATTERN_SCHEMA.items():
                if meta.get("required") and field not in merged_payload:
                    schema_warnings.append(f"required field '{field}' missing from payload")

        return {
            "id": format_object_id(oid) if oid else None,
            "name": name,
            "shard_keys": shard_keys or [],
            "tags": merged_tags,
            "action": "upserted" if result.upserted_id else "updated",
            "schema_warnings": schema_warnings,
        }

    async def strategy_recall(
        self,
        query: Optional[str] = None,
        name: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.0,
        tags: Optional[List[str]] = None,
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
            results: List[dict] = []
            # Name lookup: check strategy_key (Python-written) OR payload.name (Go-written).
            async for doc in col.find(
                {"$or": [{"strategy_key": name}, {"payload.name": name}]},
                projection={"embedding": 0},
                limit=limit,
            ):
                results.append(strip_embedding(doc, COLLECTION_STRATEGIES))
            return {"strategies": results, "results": results, "count": len(results)}

        if not query:
            # No query text — exact tag match (or plain find if no tags).
            # BM25 ($search) is avoided here: hyphens and other punctuation in tag
            # values are tokenized as delimiters, making exact tag lookup unreliable.
            tag_filter: Dict[str, Any] = {"memory_type": "strategy"}
            if tags:
                tag_filter["tags"] = {"$in": tags}
            results: List[dict] = []
            async for doc in col.find(
                tag_filter,
                projection={"embedding": 0},
                sort=[("hit_count", -1), ("created_at", -1)],
                limit=limit,
            ):
                results.append(strip_embedding(doc, COLLECTION_STRATEGIES))
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
            mt = doc.get("memory_type", "")
            if mt != "strategy":
                continue
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

        # Increment hit_count on the top result (non-fatal).
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
        results: List[dict] = []
        for doc in top:
            stripped = strip_embedding(doc, COLLECTION_STRATEGIES)
            # strategy_id: use payload.name (GoMCP) or strategy_key fallback.
            stripped["strategy_id"] = (doc.get("payload") or {}).get("name") or doc.get("strategy_key", "")
            extends_key = (doc.get("payload") or {}).get("extends")
            if extends_key:
                parent = await col.find_one(
                    {"$or": [{"strategy_key": extends_key}, {"payload.name": extends_key}]},
                    projection={"payload": 1, "_id": 0},
                )
                if parent:
                    parent_playbook = (parent.get("payload") or {}).get("playbook")
                    if parent_playbook:
                        stripped["parent_playbook"] = parent_playbook
            results.append(stripped)

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
            },
            *([{"$match": {"tags": {"$in": tags}}}] if tags else []),
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
            *([{"$match": {"tags": {"$in": tags}}}] if tags else []),
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
        """Vector-only fallback for strategy recall.

        No memory_type filter in vectorSearch (Atlas doesn't support $regex there).
        Tags filter applied as $match after $vectorSearch (avoids requiring tags as a
        filterable field in the Atlas vector index). Post-filtering by memory_type prefix in caller.
        """
        print(VECTOR_IDX_STRATEGIES)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_IDX_STRATEGIES,
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": max(50, limit * 10),
                    "limit": limit * 5,
                    "filter": {"memory_type": "strategy"},
                }
            },
            *([{"$match": {"tags": {"$in": tags}}}] if tags else []),
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
        ]
        docs: List[dict] = []
        
        async for doc in col.aggregate(pipeline):
            docs.append(doc)
        return docs

    async def get_instructions(self) -> dict:
        """
        Return the agent operating instructions string.

        Equivalent to the agent_prompt field in memory_endpoint.json on the
        Go server. The instructions are configured at service construction
        time via the agent_instructions parameter.
        """
        logger.debug("get_instructions called: returning %d chars", len(self.agent_instructions))
        return {"instructions": self.agent_instructions}
