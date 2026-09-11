"""Pattern-based routing cache with query/output hints.

Parked for future use.  The RoutingCache stores pattern → tool mapping plus
enriched hints (query_hints, output_hint) extracted from successful LLM
interactions so that future calls matching the same abstract pattern can skip
the "teach the LLM" step.

Usage (deferred — not wired into the main flow yet)::

    from mongomcp.agent.pattern_cache import PatternCache

    cache = PatternCache(settings)
    cached = await cache.get("Find [entity] near [location]")
    await cache.set("Find [entity] near [location]", tool_names, query_hints, output_hint)
    await cache.record(pattern, history, response_text)
    hint_text = PatternCache.format_hints(cached)
"""

import asyncio
import hashlib
import json
import logging
import math
import time
import requests as _requests
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PatternCache:
    """Permanent MongoDB store for pattern → tool routing decisions + query/output hints.

    Collection: ``mcp_patterns`` (in the same database as the config collection).

    Document schema per entry::

        {
            "_id":           <ObjectId>,
            "pattern_hash":  "<md5 of normalised pattern>",
            "pattern":       "Find [entity] near [location] with [attributes]",
            "tools":         ["tool_a", "tool_b"],
            "query_hints":   [{"tool_name": "...", "tool_input": {...}}, ...],
            "output_hint":   "<JSON template skeleton>",
            "embedding":     [0.1, 0.2, ...],
            "timestamp":     <unix float>,
            "hit_count":     <int>,
            "last_used":     <unix float>,
        }

    Unique index on ``pattern_hash`` ensures one entry per unique pattern.
    """

    _COLLECTION = "mcp_patterns"
    _LOG_COLLECTION = "mcp_interaction_log"

    def __init__(self, settings: Any, tool_scope: Optional[str] = None):
        from ..mongodb_client import MongoDBClient  # noqa: PLC0415
        from ..bedrock_client import BedrockClient  # noqa: PLC0415
        import copy

        local_settings = copy.copy(settings)
        local_settings.mcp_config_col = self._COLLECTION
        self._mongo = MongoDBClient(settings=local_settings)
        self._bedrock = BedrockClient(settings)
        self._mcp_root: str = getattr(settings, "mongo_mcp_root", "")
        self._auth_token: str = getattr(settings, "AUTH_TOKEN", "")
        self._headers = {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json",
        }
        self._indexes_initialized = False
        self._log_indexes_initialized = False
        self._tool_scope = tool_scope  # scopes patterns to a specific tool config

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(text: str) -> str:
        return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

    def reset_connection(self) -> None:
        self._mongo._connection_initialized = False
        self._mongo.client = {}
        self._mongo.db = {}
        self._mongo.collections = {}
        self._indexes_initialized = False

    async def _collection(self):
        await self._mongo.ensure_connection()
        col = self._mongo.get_collection(self._COLLECTION)
        if not self._indexes_initialized:
            try:
                existing = await col.index_information()
                if "mcp_patterns_phash_unique" not in existing:
                    await col.create_index(
                        "pattern_hash", unique=True, name="mcp_patterns_phash_unique"
                    )
            except Exception as e:
                logger.warning(f"Could not create pattern_hash index: {e}")
            self._indexes_initialized = True
        return col

    async def _vectorize_OLD(self, text: str) -> Optional[List[float]]:
        """Call the MCP /vectorize endpoint and return the embedding vector."""
        if not self._mcp_root:
            return None
        try:
            url = f"{self._mcp_root}/vectorize"
            resp = await asyncio.to_thread(
                _requests.post,
                url,
                json={"textChunk": text},
                headers=self._headers,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("vector")
        except Exception as e:
            logger.warning(f"PatternCache vectorize failed (embedding skipped): {e}")
            return None

    async def _vectorize(self, text: str) -> Optional[List[float]]:
        """Generate an embedding vector for text using Voyage AI via BedrockClient."""
        try:
            return await self._bedrock.generate_voyage_embeddings(text=text, is_query=True)
        except Exception as e:
            logger.warning(f"PatternCache vectorize failed (embedding skipped): {e}")
            return None

    # ------------------------------------------------------------------
    #  Embedding text builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_embedding_text(
        pattern: str,
        tool_names: Optional[List[str]] = None,
        example_queries: Optional[List[str]] = None,
    ) -> str:
        """Build a composite text for vectorisation.

        Combines the abstract pattern, tool names, and real example
        queries so the resulting embedding sits closer to natural
        language questions in vector space.
        """
        parts = [f"Pattern: {pattern}"]
        if tool_names:
            parts.append(f"Tools: {', '.join(tool_names)}")
        if example_queries:
            parts.append("Example queries:")
            for eq in example_queries:
                parts.append(f"- {eq}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    async def get(self, pattern: str) -> Optional[Dict[str, Any]]:
        """Return cached routing data for *pattern*, or None on a miss.

        Returns dict with keys: tools, query_hints, output_hint, playbook.
        """
        phash = self._make_key(pattern)
        col = await self._collection()
        query = {"pattern_hash": phash}
        if self._tool_scope:
            query["tool_scope"] = self._tool_scope
        doc = await col.find_one(
            query,
            {"tools": 1, "query_hints": 1, "output_hint": 1, "playbook": 1, "hit_count": 1},
        )
        if doc is None:
            return None
        try:
            await col.update_one(
                {"pattern_hash": phash},
                {"$inc": {"hit_count": 1}, "$set": {"last_used": time.time()}},
            )
        except Exception:
            pass
        return {
            "tools": doc.get("tools", []),
            "query_hints": doc.get("query_hints"),
            "output_hint": doc.get("output_hint"),
            "playbook": doc.get("playbook"),
        }

    async def find_best_match(
        self,
        question: str,
        similarity_threshold: float = 0.85,
        max_candidates: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find the best matching patterns for a question.

        Tries $rankFusion (vector + full-text) first; falls back to vector-only
        if the Atlas Search index is unavailable.  Each candidate is re-ranked
        with an additive hit_count boost so validated patterns float to the top.

        Returns up to *max_candidates* patterns above *similarity_threshold*,
        sorted by boosted score (descending).  Returns an empty list on a miss.
        """
        embedding = await self._vectorize(question)
        if embedding is not None:
            col = await self._collection()
            candidates = await self._find_with_rank_fusion(
                col, question, embedding, similarity_threshold, max_candidates
            )
            if candidates is None:
                # $rankFusion unavailable — fall back to vector-only
                candidates = await self._find_vector_only(
                    col, embedding, similarity_threshold, max_candidates
                )
                if candidates is not None:
                    logger.info(f"Pattern cache: $vectorSearch returned {len(candidates)} candidates")
            else:
                logger.info(f"Pattern cache: $rankFusion returned {len(candidates)} candidates")
            if candidates:
                logger.info(
                    f"Pattern cache: {len(candidates)} candidate(s) above threshold "
                    f"[best score={candidates[0].get('boosted_score', candidates[0].get('score', 0)):.3f}, "
                    f"hits={candidates[0]['hit_count']}]"
                )
            else:
                logger.info("Pattern cache MISS: no candidates above threshold")
            return candidates

        # No embedding — exact hash fallback
        exact = await self.get(question)
        if exact is not None:
            exact["pattern"] = question
            exact["hit_count"] = exact.get("hit_count", 0)
            exact["score"] = 1.0
            return [exact]
        return []

    async def _find_with_rank_fusion(
        self,
        col,
        question: str,
        embedding: List[float],
        similarity_threshold: float,
        max_candidates: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Run a $rankFusion pipeline combining vector and full-text search.

        Weights: vector=0.7, text=0.3.  Each result is further boosted by a
        log-normalised hit_count (weight 0.2) so user-validated patterns rank higher.

        Returns a candidate list, or None if $rankFusion is unavailable so the
        caller can fall back to vector-only search.
        """
        scope_filter = {"tool_scope": self._tool_scope} if self._tool_scope else None
        vector_pipeline: List[Dict] = [
            {
                "$vectorSearch": {
                    "index": "pattern_voyage_index", # "pattern_embedding_index",
                    "path": "voyage_embedding",  #"embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": max_candidates * 2,
                    **({"filter": scope_filter} if scope_filter else {}),
                }
            }
        ]
        text_pipeline: List[Dict] = [
            {
                "$search": {
                    "index": "fulltext_pattern_index",
                    "text": {
                        "query": question,
                        "path": ["pattern", "example_queries"],
                    },
                }
            },
            {"$limit": max_candidates * 2},
        ]
        if scope_filter and self._tool_scope:
            text_pipeline[0]["$search"]["filter"] = {
                "term": {"path": "tool_scope", "value": self._tool_scope}
            }
        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vector": vector_pipeline,
                            "text": text_pipeline,
                        }
                    },
                    "combination": {
                        "weights": {"vector": 0.7, "text": 0.3},
                    },
                }
            },
            {"$limit": max_candidates * 2},
            {
                "$project": {
                    "pattern": 1,
                    "tools": 1,
                    "query_hints": 1,
                    "output_hint": 1,
                    "playbook": 1,
                    "hit_count": 1,
                }
            },
        ]
        try:
            raw: List[Dict] = []
            async for doc in col.aggregate(pipeline):
                raw.append(doc)

            if not raw:
                return []

            # $rankFusion already returns results in best-first order.
            # Apply a Python-side hit_count boost and re-sort so user-validated
            # patterns can bubble above equally-ranked newcomers.
            n = len(raw)
            candidates = []
            for rank, doc in enumerate(raw):
                # Normalise position: rank 0 → 1.0, last → approaching 0.
                position_score = 1.0 - (rank / max(n, 1))
                hit_count = doc.get("hit_count", 0)
                # Patterns with negative hit_count have been flagged as bad by user feedback — skip them.
                if hit_count < 0:
                    logger.info(
                        f"Pattern cache: '{doc.get('pattern', '?')}' rejected (hit_count={hit_count})"
                    )
                    continue
                boost = (math.log(hit_count + 1) / math.log(101)) * 0.2
                boosted_score = position_score + boost
                if boosted_score >= similarity_threshold:
                    candidates.append({
                        "pattern": doc.get("pattern", "?"),
                        "tools": doc.get("tools", []),
                        "query_hints": doc.get("query_hints"),
                        "output_hint": doc.get("output_hint"),
                        "playbook": doc.get("playbook"),
                        "hit_count": hit_count,
                        "score": position_score,
                        "boosted_score": boosted_score,
                    })
                else:
                    logger.info(
                        f"Pattern cache: '{doc.get('pattern', '?')}' position_score={position_score:.3f}, "
                        f"boost={boost:.3f} (hits={hit_count}), "
                        f"boosted_score={boosted_score:.3f}, "
                        f"below threshold {similarity_threshold}"
                    )

            candidates.sort(key=lambda c: c["boosted_score"], reverse=True)
            return candidates[:max_candidates]
        except Exception as e:
            err = str(e)
            if "rankFusion" in err or "Unrecognized pipeline stage" in err:
                logger.warning(f"$rankFusion unavailable, falling back to vector-only: {e}")
                return None
            logger.warning(f"$rankFusion search failed: {e}")
            return None

    async def _find_vector_only(
        self,
        col,
        embedding: List[float],
        similarity_threshold: float,
        max_candidates: int,
    ) -> List[Dict[str, Any]]:
        """Vector-only fallback when $rankFusion is unavailable."""
        vs_query: Dict[str, Any] = {
            "index": "pattern_embedding_index",
            "path": "embedding",
            "queryVector": embedding,
            "numCandidates": 5,
            "limit": max_candidates,
        }
        if self._tool_scope:
            vs_query["filter"] = {"tool_scope": self._tool_scope}
        pipeline = [
            {"$vectorSearch": vs_query},
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {
                "$project": {
                    "pattern": 1,
                    "tools": 1,
                    "query_hints": 1,
                    "output_hint": 1,
                    "playbook": 1,
                    "hit_count": 1,
                    "score": 1,
                }
            },
        ]
        candidates = []
        try:
            async for doc in col.aggregate(pipeline):
                score = doc.get("score", 0)
                hit_count = doc.get("hit_count", 0)
                # Patterns with negative hit_count have been flagged as bad by user feedback — skip them.
                if hit_count < 0:
                    logger.info(
                        f"Pattern cache: '{doc.get('pattern', '?')}' rejected (hit_count={hit_count})"
                    )
                    continue
                if score >= similarity_threshold:
                    candidates.append({
                        "pattern": doc.get("pattern", "?"),
                        "tools": doc.get("tools", []),
                        "query_hints": doc.get("query_hints"),
                        "output_hint": doc.get("output_hint"),
                        "playbook": doc.get("playbook"),
                        "hit_count": doc.get("hit_count", 0),
                        "score": score,
                    })
                else:
                    logger.info(
                        f"Pattern cache: '{doc.get('pattern', '?')}' score={score:.3f}, "
                        f"below threshold {similarity_threshold}"
                    )
        except Exception as e:
            logger.warning(f"Vector search on patterns failed: {e}")
        return candidates

    async def find_similar_pattern(
        self,
        pattern: str,
        similarity_threshold: float = 0.75,
    ) -> Optional[str]:
        """Check if a semantically similar pattern already exists.

        Uses a *high* threshold (0.85) to only match near-duplicates.
        Returns the existing pattern string if found, or None.
        """
        embedding = await self._vectorize(pattern)
        if embedding is None:
            return None
        col = await self._collection()
        try:
            vs_query = {
                "index": "pattern_embedding_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 5,
                "limit": 1,
            }
            if self._tool_scope:
                vs_query["filter"] = {"tool_scope": self._tool_scope}
            pipeline = [
                {"$vectorSearch": vs_query},
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$project": {"pattern": 1, "score": 1}},
            ]
            async for doc in col.aggregate(pipeline):
                score = doc.get("score", 0)
                existing = doc.get("pattern", "")
                if score >= similarity_threshold:
                    logger.info(
                        f"Dedup: new pattern matches existing '{existing}' "
                        f"(score={score:.3f}), will merge"
                    )
                    return existing
                else:
                    logger.debug(
                        f"Dedup: best match '{existing}' scored {score:.3f}, "
                        f"below dedup threshold {similarity_threshold}"
                    )
        except Exception as e:
            logger.warning(f"Dedup vector search failed: {e}")
        return None

    async def set(
        self,
        pattern: str,
        tool_names: List[str],
        query_hints: Optional[List[Dict]] = None,
        output_hint: Optional[str] = None,
        playbook: Optional[str] = None,
        example_queries: Optional[List[str]] = None,
    ) -> None:
        """Upsert a routing entry keyed by pattern.

        Embeds a **composite text** (pattern + tools + example queries) so that
        the resulting vector sits close to real natural-language questions.
        New example queries are merged (deduplicated, capped at 10) with any
        existing ones, and the embedding is rebuilt from the full composite.
        """
        phash = self._make_key(pattern)
        col = await self._collection()

        # Merge example_queries with any already stored for this pattern
        merged_examples: List[str] = []
        if example_queries:
            existing_doc = await col.find_one(
                {"pattern_hash": phash}, {"example_queries": 1}
            )
            prior = existing_doc.get("example_queries", []) if existing_doc else []
            seen = set(q.strip().lower() for q in prior)
            merged_examples = list(prior)  # preserve order
            for eq in example_queries:
                if eq and eq.strip().lower() not in seen:
                    merged_examples.append(eq)
                    seen.add(eq.strip().lower())
            # Cap to 10 most recent examples
            merged_examples = merged_examples[-10:]

        # Build composite text for richer embedding
        composite = self._build_embedding_text(pattern, tool_names, merged_examples or None)
        embedding = await self._vectorize(composite)

        doc: Dict[str, Any] = {
            "pattern": pattern,
            "tools": tool_names,
            "timestamp": time.time(),
            "last_used": time.time(),
        }
        if self._tool_scope:
            doc["tool_scope"] = self._tool_scope
        if merged_examples:
            doc["example_queries"] = merged_examples
        if query_hints is not None:
            doc["query_hints"] = query_hints
        if output_hint is not None:
            doc["output_hint"] = output_hint
        if playbook is not None:
            doc["playbook"] = playbook
        if embedding is not None:
            doc["voyage_embedding"] = embedding
        await col.update_one(
            {"pattern_hash": phash},
            {
                "$setOnInsert": {"pattern_hash": phash, "hit_count": 0},
                "$set": doc,
            },
            upsert=True,
        )

    async def clear(self) -> None:
        """Drop all routing patterns (destructive — use with care)."""
        col = await self._collection()
        await col.delete_many({})

    # ------------------------------------------------------------------
    #  Interaction log — one document per browser user_id, interactions array
    # ------------------------------------------------------------------

    async def _log_collection(self):
        """Return the interaction log collection, ensuring the compound (user_id, session_id) index exists."""
        await self._mongo.ensure_connection()
        col = self._mongo.get_collection(self._LOG_COLLECTION)
        if not self._log_indexes_initialized:
            try:
                existing = await col.index_information()
                if "mcp_interaction_log_uid_sid" not in existing:
                    await col.create_index(
                        [("user_id", 1), ("session_id", 1)],
                        unique=True,
                        name="mcp_interaction_log_uid_sid",
                    )
            except Exception as e:
                logger.warning(f"Could not create interaction_log index: {e}")
            self._log_indexes_initialized = True
        return col

    @staticmethod
    def _extract_tool_calls_from_history(history: list) -> List[Dict[str, Any]]:
        """Pull tool name + input from Bedrock toolUse blocks in history."""
        calls = []
        for msg in history:
            if msg.get("role") != "assistant":
                continue
            for block in msg.get("content", []):
                if "toolUse" in block:
                    tu = block["toolUse"]
                    calls.append({"tool_name": tu.get("name", ""), "tool_input": tu.get("input", {})})
        return calls

    async def log_interaction(
        self,
        user_id: str,
        session_id: str,
        question: str,
        history: list,
        response_text: str,
        outcome: str,
        pattern_matched: Optional[str] = None,
        pattern_hash: Optional[str] = None,
        tools_selected: Optional[List[str]] = None,
    ) -> None:
        """Upsert the single per-session interaction document.

        One document per (user_id, session_id).  Each call appends the
        new question to the ``questions`` array and overwrites the tool,
        response, and pattern fields with the latest values — all in one
        atomic update so follow-up questions within a session stay together.
        """
        try:
            col = await self._log_collection()
            await col.update_one(
                {"user_id": user_id, "session_id": session_id},
                {
                    "$setOnInsert": {
                        "user_id": user_id,
                        "session_id": session_id,
                        "created_at": time.time(),
                        "feedback": None,
                    },
                    "$push": {
                        "questions": question,
                        "responses": response_text[:500] + ("\u2026" if len(response_text) > 500 else ""),
                    },
                    "$set": {
                        "tool_calls": self._extract_tool_calls_from_history(history),
                        "tools_selected": tools_selected or [],
                        "outcome": outcome,
                        "pattern_matched": pattern_matched,
                        "pattern_hash": pattern_hash,
                        "updated_at": time.time(),
                    },
                },
                upsert=True,
            )
        except Exception as e:
            logger.warning(f"log_interaction failed: {e}")

    async def record_feedback(self, user_id: str, session_id: str, feedback: str) -> bool:
        """Record user feedback on the session document and penalise the pattern on negative signal.

        Targets the session document directly via (user_id, session_id) —
        no positional index is needed since there is one document per session.
        """
        if feedback not in ("positive", "negative"):
            return False
        try:
            col = await self._log_collection()
            result = await col.find_one_and_update(
                {"user_id": user_id, "session_id": session_id},
                {"$set": {"feedback": feedback, "feedback_at": time.time()}},
                return_document=True,
                projection={"pattern_hash": 1},
            )
            if result is None:
                return False
            if feedback == "negative":
                ph = result.get("pattern_hash")
                if ph:
                    patterns_col = await self._collection()
                    await patterns_col.update_one(
                        {"pattern_hash": ph},
                        {"$inc": {"hit_count": -1}},
                    )
            return True
        except Exception as e:
            logger.warning(f"record_feedback failed: {e}")
            return False

    async def record(
        self,
        pattern: str,
        history: list,
        response_text: str,
        selected_tools: Optional[List[str]] = None,
        jsondata: Any = None,
        question: Optional[str] = None,
    ) -> None:
        """Extract tool calls + output format from a completed interaction and enrich the cached pattern.

        Args:
            pattern: The abstract pattern string from the router.
            history: Bedrock conversation history (may span multiple turns).
            response_text: The final LLM response text.
            selected_tools: Tool names the router selected for this pattern.
                Used to filter history to only relevant calls and preserved
                as the authoritative ``tools`` list.
            jsondata: Parsed JSON data object from the response (already
                extracted by normalize_bedrock_response).  Used to build
                the output_hint when the raw text no longer contains
                [JSON_DATA_START] tags.
            question: The original natural-language question.  Stored as an
                example query to enrich the embedding for future lookups.
        """
        query_hints = self.extract_tool_calls(history, allowed_tools=selected_tools)
        output_hint = self.extract_output_hint(response_text)
        if output_hint is None and jsondata is not None:
            output_hint = self._skeleton_from_jsondata(jsondata)
        if not query_hints and not output_hint:
            return

        # Use the router's selection as authoritative; fall back to extracted names
        if selected_tools:
            tool_names = selected_tools
        else:
            tool_names = list(dict.fromkeys(h["tool_name"] for h in query_hints))

        self.reset_connection()
        await self.set(
            pattern=pattern,
            tool_names=tool_names or [],
            query_hints=query_hints or None,
            output_hint=output_hint,
            example_queries=[question] if question else None,
        )
        logger.info(f"Recorded pattern '{pattern}' with {len(query_hints)} query hints")

    # ------------------------------------------------------------------
    #  Static extraction / formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_hints(hints: Dict[str, Any]) -> Optional[str]:
        """Build a text block from cached hints suitable for injection into a prompt.

        Prefers the LLM-generated playbook (PII-free, structured recipe).
        Falls back to legacy query_hints/output_hint if no playbook exists.
        Returns None if there are no meaningful hints to inject.
        """
        # Prefer playbook — it's a complete, PII-free instruction set
        if hints.get("playbook"):
            return hints["playbook"]

        # Legacy fallback for patterns saved before playbook was introduced
        if not hints.get("query_hints") and not hints.get("output_hint"):
            return None

        parts = ["[OUTPUT FORMAT — Use the following format for output data]"]

        if hints.get("query_hints"):
            parts.append("\nTool calls that worked for a similar question:")
            for qh in hints["query_hints"]:
                parts.append(f"  Tool: {qh['tool_name']}")
                parts.append(f"  Input: {json.dumps(qh['tool_input'], default=str)}")

        if hints.get("output_hint"):
            parts.append(
                "\nWrap your output data in [JSON_DATA_START] and [JSON_DATA_END] tags using the following format "
                "(adapt values to the current question):\n"
                f"[JSON_DATA_START]\n{hints['output_hint']}\n[JSON_DATA_END]"
            )

        return "\n".join(parts)

    @staticmethod
    def extract_tool_calls(
        history: list,
        allowed_tools: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Pull tool_name + tool_input from Bedrock toolUse blocks in conversation history.

        Args:
            history: Bedrock conversation messages.
            allowed_tools: If provided, only keep calls whose tool name is in
                this list. This prevents leaking hints from earlier unrelated
                conversation turns.

        Returns:
            Deduplicated list of {tool_name, tool_input} dicts.  When the same
            tool was called multiple times, only the last invocation is kept
            (the LLM typically refines parameters across retries).
        """
        allowed = set(allowed_tools) if allowed_tools else None
        # Collect all matching calls; later entries override earlier ones per tool_name
        seen: Dict[str, Dict[str, Any]] = {}
        for msg in history:
            if msg.get("role") != "assistant":
                continue
            for block in msg.get("content", []):
                if "toolUse" in block:
                    tu = block["toolUse"]
                    name = tu.get("name", "")
                    if allowed and name not in allowed:
                        continue
                    # Keep last call per tool_name (refined parameters)
                    seen[name] = {
                        "tool_name": name,
                        "tool_input": tu.get("input", {}),
                    }
        return list(seen.values())

    @staticmethod
    def extract_output_hint(response_text: str) -> Optional[str]:
        """Extract a sanitised JSON output skeleton from [JSON_DATA_START]…[JSON_DATA_END] blocks.

        Keeps structure and one example marker/row but strips bulk data so the
        hint is compact.
        """
        start_tag = "[JSON_DATA_START]"
        end_tag = "[JSON_DATA_END]"
        start = response_text.find(start_tag)
        end = response_text.find(end_tag)
        if start == -1 or end == -1 or end <= start:
            return None
        json_str = response_text[start + len(start_tag) : end].strip()
        try:
            data = json.loads(json_str)
            return json.dumps(PatternCache._make_skeleton(data), indent=2)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _skeleton_from_jsondata(jsondata) -> Optional[str]:
        """Build a compact output hint from the parsed jsondata object.

        This is the fallback when [JSON_DATA_START] tags have already been
        stripped from response_text by normalize_bedrock_response.
        """
        try:
            if isinstance(jsondata, str):
                jsondata = json.loads(jsondata)
            if isinstance(jsondata, dict):
                return json.dumps(PatternCache._make_skeleton(jsondata), indent=2)
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _make_skeleton(data: dict) -> dict:
        """Reduce a JSON dict to a compact skeleton — keep structure, trim bulk arrays to one example."""
        skeleton: Dict[str, Any] = {}
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 1:
                skeleton[key] = [val[0]]
            else:
                skeleton[key] = val
        return skeleton
