"""
mongomcp.memory — Python port of the go-mongo-memo memory layer.

Public API:  register_memory_tools(mcp, db_client, llm_client, settings)

Call this after creating the FastMCP instance and before calling
mcp.http_app() so that all memory tools are registered on the same
MCP instance and share the same auth provider.

Returns a dict of {tool_name: fn} suitable for merging into _TOOL_DISPATCH
in mongo_mcp.py.

Collections (all in settings.memory_db, default "mcp_config"):
  memory_episodic   — main episodic / semantic memories
  memory_strategies — reusable strategy / pattern entries
  memory_schemas    — declared memory schemas (schema_declare)

Atlas vector indexes required (see jsonartifacts/memory_indexes.json):
  memory_episodic_vector_index   on memory_episodic.embedding
  memory_strategies_vector_idx on memory_strategies.embedding
"""

import logging
from typing import Any, Dict, Annotated, List, Optional

from pydantic import Field
from fastapi import Depends
from fastmcp.server.dependencies import get_access_token, AccessToken

from .memservice import MemoryService
from .tools import build_memory_tool_fns
from ..mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


def register_memory_tools(mcp, db_client, llm_client, settings) -> Dict[str, Any]:
    """
    Register the 8 memory MCP tools on the given FastMCP instance.

    Parameters
    ----------
    mcp       : FastMCP instance (already configured with auth)
    db_client : MongoDBClient — motor client used to reach memory collections
    llm_client: BedrockClient — used for generate_embedding and invoke_bedrock_text
    settings  : AWSSettings / LocalSettings — must have .memory_db attribute

    Returns
    -------
    dict mapping tool_name -> .fn (unwrapped coroutine) for _TOOL_DISPATCH.
    """
    memory_db = getattr(settings, "memory_db", "mcp_config")
    query_model = getattr(settings, "QUERY_EMBEDDING_MODEL_ID", None)
    agent_instructions = getattr(settings, "agent_instructions", "")
    # Dedicated MongoDBClient for memory — avoids sharing Motor client lifecycle
    # with mongo_server (the tool's DB client). Fresh instance connects lazily
    # inside the running event loop on first use.
    memory_db_client = MongoDBClient(settings)
    svc = MemoryService(
        db_client=memory_db_client,
        llm_client=llm_client,
        memory_db_name=memory_db,
        query_embedding_model_id=query_model,
        agent_instructions=agent_instructions,
    )
    raw_fns = build_memory_tool_fns(svc)

    # ------------------------------------------------------------------
    # Register each tool on the mcp instance with Annotated+Field params
    # that match the go-mongo-memo MCP contract so agents can target
    # either implementation interchangeably.
    # ------------------------------------------------------------------

    @mcp.tool()
    async def intake(
        content: Annotated[str, Field(description="The memory content to store.")],
        memory_type: Annotated[str, Field(description="Memory type tag, e.g. 'episodic', 'task', 'step:execution', 'session:summary'.")] = "episodic",
        importance: Annotated[float, Field(default=0.5, description="Importance score 0.0-1.0.", ge=0.0, le=1.0)] = 0.5,
        decay_rate: Annotated[float, Field(default=0.01, description="How quickly importance decays per day (0=no decay, 0.1=fast).", ge=0.0, le=1.0)] = 0.01,
        session_id: Annotated[Optional[str], Field(default=None, description="Session identifier to group related memories.")] = None,
        tags: Annotated[Optional[List[str]], Field(default=None, description="List of string tags.")] = None,
        entities: Annotated[Optional[List[str]], Field(default=None, description="Named entities mentioned in the memory.")] = None,
        payload: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Arbitrary structured metadata. Do NOT put 'scope' here — use the top-level scope parameter.")] = None,
        payload_push: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Update-mode only (requires _id). Appends values to existing payload array fields using $push — without replacing the whole payload. Keys are payload field names; values are items to append. Pass a list as the value to append multiple items at once ($each). Example: {\"batches_array\": {\"batch_id\": 61, \"score\": 0.9}}.")] = None,
        username: Annotated[Optional[str], Field(default=None, description="Username storing the memory.")] = None,
        schema_version: Annotated[Optional[str], Field(default=None, description="Declared schema name to validate the payload against. Warnings returned but write still succeeds.")] = None,
        scope: Annotated[int, Field(default=-1, description="Visibility scope — MUST be a top-level parameter, not inside payload. 0=shared (all agents), 10=agent-only, 20=this user any session, 30=this user+session (default), 40=this user+session+agent. -1 = use default (30).")] = -1,
        related_docs: Annotated[Optional[List[Dict[str, Any]]], Field(default=None, description="Explicit graph links to store on this memory. Each entry: {id: ObjectID hex, relation: string, explicit: true}. No automatic vector-search linking is performed — use memory_reflect(operation='link') after the fact for bulk linking.")] = None,
        _id: Annotated[Optional[str], Field(default=None, description="Optional ObjectID hex string. When provided, updates the existing memory with this _id in-place rather than inserting a new document. Re-embeds the new content and updates all supplied fields. The _id, related_docs, and session_id are preserved so all graph links remain valid.")] = None,
        token: Annotated[AccessToken, Depends(get_access_token)] = None,
    ):
        """Store a memory with optional explicit related_docs links. Near-duplicate warning returned if cosine similarity >= 0.92 with an existing doc. No automatic links are created — use memory_reflect(operation='link') to add links explicitly."""
        agent_id = _agent_id_from_token(token)
        return await raw_fns["intake"](
            content=content, memory_type=memory_type, importance=importance,
            decay_rate=decay_rate, session_id=session_id, tags=tags, entities=entities,
            payload=payload, payload_push=payload_push, username=username,
            agent_id=agent_id, schema_version=schema_version,
            scope=scope, related_docs=related_docs, _id=_id,
        )

    @mcp.tool()
    async def recall(
        query: Annotated[Optional[str], Field(default=None, description="Natural language query for semantic vector search. Optional when 'entities' is provided — omitting query runs a direct entity filter with no embedding call.")] = None,
        session_id: Annotated[Optional[str], Field(default=None, description="Restrict recall to a specific session (episodic pre-filter).")] = None,
        username: Annotated[Optional[str], Field(default=None, description="Username to scope episodic results.")] = None,
        scope: Annotated[str, Field(default="all", description="Collection scope: 'all' (default), 'episodic', or 'semantic'. Use 'semantic' when filtering for strategy or schema memory_types.")] = "all",
        limit: Annotated[int, Field(default=5, description="Maximum number of fully-hydrated memories to return. Overflow matches are returned as lightweight stubs in entity_match_stubs.", ge=1, le=50)] = 5,
        num_candidates: Annotated[int, Field(default=150, description="ANN candidate pool size for vector search (ignored on entities-only path).", ge=10, le=1000)] = 150,
        score_threshold: Annotated[float, Field(default=0.0, description="Minimum composite score threshold (0.0-1.0).", ge=0.0, le=1.0)] = 0.0,
        importance_threshold: Annotated[float, Field(default=0.0, description="Minimum importance score pre-filter.", ge=0.0, le=1.0)] = 0.0,
        memory_types: Annotated[Optional[List[str]], Field(default=None, description="Filter by memory_type values. Schema docs use 'schema:<name>' (e.g. 'schema:uw-agent-output'). Always pair with scope='semantic' when filtering schemas or strategies.")] = None,
        tags: Annotated[Optional[List[str]], Field(default=None, description="Filter by tags (any match).")] = None,
        entities: Annotated[Optional[List[str]], Field(default=None, description="Filter by named entities. When query is omitted, runs a direct entity filter (no embedding call); first 'limit' results are fully hydrated, overflow returned as entity_match_stubs.")] = None,
        depth: Annotated[int, Field(default=1, description="Graph traversal depth for BFS expansion via related_docs edges. 1=one hop (default), up to 5. Each hop follows edges from hydrated results of the prior hop. Deeper docs receive a score penalty and appear as graph_neighbors or graph_neighbor_stubs.", ge=1, le=5)] = 1,
        depth_relations: Annotated[Optional[List[str]], Field(default=None, description="Restrict BFS traversal to edges with these relation labels (e.g. ['linked', 'derived_from']). Omit to follow all relation types.")] = None,
        output_format: Annotated[str, Field(default="default", description="Output format: 'default' returns the standard results list; 'graph' returns a jsonDataType=memory_graph object with nodes[] and edges[] for rendering as an interactive force-directed graph in the webui. Use 'graph' when you want to visualise the memory graph.")] = "default",
        token: Annotated[AccessToken, Depends(get_access_token)] = None,
    ):
        """Recall memories via semantic vector search or direct entity filter, with multi-hop BFS graph expansion and composite scoring. Routing: query+entities=vector+post-filter; entities-only=direct find (no embedding); query-only=pure vector; neither=error."""
        agent_id = _agent_id_from_token(token)
        return await raw_fns["recall"](
            query=query, session_id=session_id,
            agent_id=agent_id, username=username,
            scope=scope, limit=limit, num_candidates=num_candidates,
            score_threshold=score_threshold, importance_threshold=importance_threshold,
            memory_types=memory_types, tags=tags, entities=entities,
            depth=depth, depth_relations=depth_relations,
            output_format=output_format,
        )

    @mcp.tool()
    async def reflect(
        session_id: Annotated[Optional[str], Field(default=None, description="Session ID to summarise. Required for operation='summarise'; not needed for 'link' or 'set_entities'.")] = None,
        operation: Annotated[str, Field(default="summarise", description="Operation to perform: 'summarise' — LLM-summarise session and store as session:summary; 'link' — create explicit bidirectional related_docs links between memory_ids and target_ids; 'set_entities' — overwrite the entities[] field on documents in memory_ids.")] = "summarise",
        memory_ids: Annotated[Optional[List[str]], Field(default=None, description="Source memory ObjectID hex strings. Required for 'link' and 'set_entities' operations.")] = None,
        target_ids: Annotated[Optional[List[str]], Field(default=None, description="Target memory ObjectID hex strings for the 'link' operation. When omitted, falls back to symmetric all-pairs on memory_ids.")] = None,
        link_relation: Annotated[str, Field(default="linked", description="Relation label written on forward links (source → target).")] = "linked",
        inverse_relation: Annotated[Optional[str], Field(default=None, description="Relation label written on back links (target → source). Defaults to link_relation when omitted.")] = None,
        entities: Annotated[Optional[List[str]], Field(default=None, description="Entity list to set on documents for operation='set_entities'. Overwrites the existing entities[] field.")] = None,
        username: Annotated[Optional[str], Field(default=None, description="Username to attribute to the stored session:summary document (operation='summarise').")] = None,
        token: Annotated[AccessToken, Depends(get_access_token)] = None,
    ):
        """Multi-operation memory maintenance: summarise a session via LLM (stores agent_id from token + caller-supplied username on the summary), create explicit bidirectional related_docs links, or overwrite entities[] on existing memories."""
        agent_id = _agent_id_from_token(token)
        return await raw_fns["reflect"](
            session_id=session_id,
            operation=operation, memory_ids=memory_ids, target_ids=target_ids,
            link_relation=link_relation, inverse_relation=inverse_relation,
            entities=entities, agent_id=agent_id, username=username,
        )

    @mcp.tool()
    async def query(
        filter: Annotated[Optional[Dict[str, Any]], Field(default=None, description="MongoDB filter dict to narrow results.")] = None,
        limit: Annotated[int, Field(default=20, description="Maximum documents to return.", ge=1, le=500)] = 20,
        scope: Annotated[str, Field(default="episodic", description="Collection scope: 'episodic' or 'strategies'.")] = "episodic",
        sort_by: Annotated[str, Field(default="created_at", description="Field to sort by.")] = "created_at",
        sort_dir: Annotated[str, Field(default="desc", description="Sort direction: 'asc' or 'desc'.")] = "desc",
        ids: Annotated[Optional[List[str]], Field(default=None, description="List of ObjectID hex strings for direct ID fetch. Bypasses all scope/ownership filters — searches both collections. Any filter fields are applied on top of the ID match.")] = None,
    ):
        """Query memories by filter, scope, and sort. ids=[...] bypasses ALL scope filters and searches BOTH collections by _id — use this for exact retrieval when you have an ObjectId from a prior result. Do NOT use filter={'_id': '...'} — string _id is NOT cast to ObjectId."""
        return await raw_fns["query"](
            filter=filter, limit=limit, scope=scope,
            sort_by=sort_by, sort_dir=sort_dir, ids=ids,
        )

    @mcp.tool()
    async def list_sessions(
        filter: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Optional filter applied to session lookup.")] = None,
        limit: Annotated[int, Field(default=20, description="Maximum sessions to return.", ge=1, le=200)] = 20,
    ):
        """List sessions by finding session:summary memories, with fallback to distinct session_id grouping."""
        return await raw_fns["list_sessions"](filter=filter, limit=limit)

    @mcp.tool()
    async def schema_declare(
        schema_name: Annotated[str, Field(description="Unique name for the schema.")],
        fields: Annotated[Dict[str, Any], Field(description="Field definitions as a JSON object. Each field: {type, description, required}.")],
        content: Annotated[Optional[str], Field(default=None, description="Human-readable description of the schema, embedded for semantic search.")] = None,
        version: Annotated[str, Field(default="1.0", description="Schema version string (e.g. '1.0').")]  = "1.0",
        token: Annotated[AccessToken, Depends(get_access_token)] = None,
    ):
        """Declare or update a memory schema definition (upsert by schema_name). Stored in memory_semantic as memory_type='schema:<name>' with fields inside payload."""
        agent_id = _agent_id_from_token(token)
        return await raw_fns["schema_declare"](
            schema_name=schema_name, fields=fields,
            content=content, version=version, agent_id=agent_id,
        )

    @mcp.tool()
    async def strategy_store(
        name: Annotated[str, Field(description="Canonical name for this strategy (e.g. 'high_importance_facts').")],
        context: Annotated[str, Field(description="Natural-language description of what this strategy does. Gets embedded for semantic recall and stored as the content field.")],
        _id: Annotated[Optional[str], Field(default=None, description="ObjectId hex of an existing strategy to update in-place. When supplied, re-embeds context and replaces payload if provided. version_seq unchanged.")] = None,
        tags: Annotated[Optional[List[str]], Field(default=None, description="Tags for categorisation. Tool names from payload.tools are auto-added.")] = None,
        entities: Annotated[Optional[List[str]], Field(default=None, description="Entity labels for graph linking and entity-filter recall (e.g. tool names, project names).")] = None,
        username: Annotated[Optional[str], Field(default=None, description="Owner username. Stored for scope-based visibility filtering.")] = None,
        scope: Annotated[int, Field(default=-1, description="Visibility scope int (0=shared, 10=agent, 20=user, 30=user_session, 40=user_session_agent). -1 defaults to user_session.")] = -1,
        session_id: Annotated[Optional[str], Field(default=None, description="Optional session context.")] = None,
        importance: Annotated[float, Field(default=0.90, description="Importance score 0.0-1.0. Default 0.90. Auto-overridden to 0.98 when scope=0 (shared).", ge=0.0, le=1.0)] = 0.90,
        decay_rate: Annotated[float, Field(default=0.001, description="Decay rate for composite recall scoring. Default 0.001. Auto-overridden to 0.0 when scope=0 (shared).", ge=0.0)] = 0.001,
        memory_type: Annotated[str, Field(default="strategy", description="Memory type tag. Defaults to 'strategy'. Override for sub-types e.g. 'strategy:routing'.")] = "strategy",
        payload: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Extra structured metadata. For routing_pattern: {tools, playbook, output_hint, extends}.")] = None,
        schema_version: Annotated[Optional[str], Field(default=None, description="Schema name to validate payload against.")] = None,
        superseded_by: Annotated[Optional[str], Field(default=None, description="Only used with _id (update path). Sets superseded_by on the existing document. Ignored on insert — superseded_by is managed automatically by code.")] = None,
        related_doc_ids: Annotated[Optional[List[str]], Field(default=None, description="Memory _id hex strings to bidirectionally link to this strategy after store. Works on both insert and update paths.")] = None,
        link_relation: Annotated[str, Field(default="related", description="Relation label for the links created via related_doc_ids. Default 'related'.")] = "related",
        token: Annotated[AccessToken, Depends(get_access_token)] = None,
    ):
        """Store a new strategy version (insert, version_seq auto-incremented) or update an existing one (_id supplied, re-embeds context + replaces payload)."""
        agent_id = _agent_id_from_token(token)
        return await raw_fns["strategy_store"](
            name=name, context=context, _id=_id, tags=tags, entities=entities,
            username=username, agent_id=agent_id, scope=scope,
            session_id=session_id, importance=importance, decay_rate=decay_rate,
            memory_type=memory_type,
            payload=payload, schema_version=schema_version,
            superseded_by=superseded_by,
            related_doc_ids=related_doc_ids,
            link_relation=link_relation,
        )

    @mcp.tool()
    async def strategy_recall(
        query: Annotated[Optional[str], Field(default=None, description="Describe the retrieval goal. Semantic search via $rankFusion (vector+BM25).")] = None,
        name: Annotated[Optional[str], Field(default=None, description="Exact strategy name for direct lookup. Takes precedence over query.")] = None,
        limit: Annotated[int, Field(default=5, description="Maximum strategies to return.", ge=1, le=50)] = 5,
        similarity_threshold: Annotated[float, Field(default=0.5, description="Minimum boosted score (0.0-1.0).", ge=0.0, le=1.0)] = 0.5,
        tags: Annotated[Optional[List[str]], Field(default=None, description="Filter by tags. Without a query, performs exact tag match sorted by hit_count.")] = None,
        include_history: Annotated[bool, Field(default=False, description="When True and name is set, returns all versions of the strategy sorted by version_seq descending. Default False returns only the most recent version.")] = False,
    ):
        """Recall strategies. name → most recent version (include_history=True for full version history); query → semantic $rankFusion (vector+BM25); tags only → exact tag match sorted by hit_count; nothing → top by hit_count. Returns parent_playbook when extends is set."""
        return await raw_fns["strategy_recall"](
            query=query, name=name,
            limit=limit, similarity_threshold=similarity_threshold, tags=tags,
            include_history=include_history,
        )

    @mcp.tool()
    async def get_instructions():
        """Return the full agent operating instructions string configured at service startup."""
        result = await raw_fns["get_instructions"]()
        instr = result.get("instructions", "") if isinstance(result, dict) else ""
        logger.debug("[PIPELINE] __init__.get_instructions: returning instructions len=%d", len(instr))
        return result

    def _agent_id_from_token(token) -> Optional[str]:
        """Extract agent_name from an AccessToken (FastMCP path) or agent_rec dict (HTTP path)."""
        try:
            if token is None:
                token = get_access_token()
            if hasattr(token, "claims"):
                result = token.claims.get("agent_name") or None
                logger.info("[agent_id] AccessToken claims=%s → agent_id=%s", token.claims, result)
                return result
            if isinstance(token, dict):
                result = token.get("agent_name") or None
                logger.info("[agent_id] dict token keys=%s → agent_id=%s", list(token.keys()), result)
                return result
        except Exception as e:
            logger.warning("[agent_id] _agent_id_from_token failed: %s", e)
            raise e
        logger.warning("[agent_id] token type=%s, returning None", type(token))

        return None

    def _r(tool_obj):
        return getattr(tool_obj, "fn", tool_obj)

    # Return the dispatch table entries for _TOOL_DISPATCH in mongo_mcp.py
    return {
        "intake":           _r(intake),
        "recall":           _r(recall),
        "reflect":          _r(reflect),
        "query":            _r(query),
        "list_sessions":    _r(list_sessions),
        "schema_declare":   _r(schema_declare),
        "strategy_store":   _r(strategy_store),
        "strategy_recall":  _r(strategy_recall),
        "get_instructions": _r(get_instructions),
    }


def build_memory_dispatch(db_client, llm_client, settings) -> Dict[str, Any]:
    """
    Build a {tool_name: async_fn} dispatch table for direct (non-MCP) memory tool calls.

    Used by CachedQueryProcessor to dispatch memory_* tool calls without going through
    a FastMCP HTTP endpoint.  The returned functions are plain coroutines — call them as
    ``await dispatch[tool_name](**tool_input)``.
    """
    memory_db = getattr(settings, "memory_db", "mcp_config")
    query_model = getattr(settings, "QUERY_EMBEDDING_MODEL_ID", None)
    agent_instructions = getattr(settings, "agent_instructions", "")
    memory_db_client = MongoDBClient(settings)
    svc = MemoryService(
        db_client=memory_db_client,
        llm_client=llm_client,
        memory_db_name=memory_db,
        query_embedding_model_id=query_model,
        agent_instructions=agent_instructions,
    )
    return build_memory_tool_fns(svc)


def get_memory_bedrock_toolspecs() -> List[Dict[str, Any]]:
    """
    Return Bedrock-format toolSpec dicts for all 9 memory tools.

    These are injected directly into the LLM tool catalog by CachedQueryProcessor
    so memory tools are always available regardless of which MCP endpoints are
    configured.  Schema matches the go-mongo-memo MCP contract exactly.
    """
    def _spec(name: str, description: str, properties: dict, required: list) -> dict:
        return {
            "toolSpec": {
                "name": name,
                "description": description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                },
            }
        }

    return [
        _spec(
            "intake",
            "Store a memory with auto-linking to similar existing memories via vector similarity.",
            {
                "content":        {"type": "string", "description": "The memory content to store."},
                "memory_type":    {"type": "string", "description": "Memory type tag, e.g. 'episodic', 'task', 'step:execution', 'session:summary'.", "default": "episodic"},
                "importance":     {"type": "number", "description": "Importance score 0.0-1.0.", "default": 0.5},
                "decay_rate":     {"type": "number", "description": "How quickly importance decays per day (0=no decay, 0.1=fast).", "default": 0.01},
                "session_id":     {"type": "string", "description": "Session identifier to group related memories."},
                "tags":           {"type": "array",  "items": {"type": "string"}, "description": "List of string tags."},
                "entities":       {"type": "array",  "items": {"type": "string"}, "description": "Named entities mentioned in the memory."},
                "payload":        {"type": "object", "description": "Arbitrary structured metadata. Do NOT put 'scope' here — use the top-level scope parameter."},
                "username":       {"type": "string", "description": "Username storing the memory."},
                "schema_version": {"type": "string", "description": "Declared schema name to validate payload against."},
                "is_isolated":    {"type": "boolean", "description": "When True, memory is private to this agent only. When False (default), memory is shared and visible to all agents.", "default": False},
                "_id":            {"type": "string",  "description": "Optional ObjectID hex string. When provided, updates the existing memory with this _id in-place rather than inserting. Re-embeds content; updates tags/entities/payload/importance/decay_rate/schema_version if supplied. The _id, related_docs, and session_id are preserved."},
                "scope":          {"type": "integer", "description": "Visibility scope — MUST be a top-level parameter, not inside payload. 0=shared, 10=agent-only, 20=this user any session, 30=this user+session (default), 40=this user+session+agent. -1=default (30).", "default": -1},
            },
            ["content"],
        ),
        _spec(
            "recall",
            "Recall relevant memories via semantic search across episodic and semantic collections, with graph expansion and composite scoring.",
            {
                "query":                {"type": "string",  "description": "Natural language query to find relevant memories."},
                "session_id":           {"type": "string",  "description": "Restrict recall to a specific session (episodic pre-filter)."},
                "username":             {"type": "string",  "description": "Username to scope episodic results."},
                "scope":                {"type": "string",  "description": "Collection scope: 'all' (default), 'episodic', or 'semantic'.", "default": "all"},
                "limit":                {"type": "integer", "description": "Maximum memories to return.", "default": 5},
                "num_candidates":       {"type": "integer", "description": "ANN candidate pool size for vector search.", "default": 150},
                "score_threshold":      {"type": "number",  "description": "Minimum composite score threshold (0.0-1.0).", "default": 0.0},
                "importance_threshold": {"type": "number",  "description": "Minimum importance score pre-filter (episodic only).", "default": 0.0},
                "memory_types":         {"type": "array",   "items": {"type": "string"}, "description": "Filter by memory_type values."},
                "tags":                 {"type": "array",   "items": {"type": "string"}, "description": "Filter by tags (any match)."},
                "entities":             {"type": "array",   "items": {"type": "string"}, "description": "Filter by named entities (Python post-filter)."},
                "depth":                {"type": "integer", "description": "BFS graph expansion depth via related_docs. 1=one hop (default), up to 5.", "default": 1},
                "depth_relations":      {"type": "array",   "items": {"type": "string"}, "description": "Restrict BFS to these relation labels. Omit to follow all."},
                "output_format":        {"type": "string",  "description": "'default' returns results list; 'graph' returns jsonDataType=memory_graph for the webui force-directed graph widget.", "default": "default"},
            },
            [],
        ),
        _spec(
            "reflect",
            "Multi-operation memory maintenance: 'summarise' — LLM-summarise a session and store as session:summary with bidirectional links to source memories (agent_id always from auth token; pass username to attribute ownership); 'link' — create explicit directional related_docs links between memories; 'set_entities' — overwrite entities[] on existing memories.",
            {
                "session_id":       {"type": "string",  "description": "Session ID to summarise. Required for operation='summarise'; not needed for 'link' or 'set_entities'."},
                "operation":        {"type": "string",  "description": "'summarise' (default), 'link', or 'set_entities'.", "default": "summarise"},
                "username":         {"type": "string",  "description": "Username to store on the session:summary document (operation='summarise')."},
                "memory_ids":       {"type": "array",   "items": {"type": "string"}, "description": "Source memory ObjectID hex strings. Required for 'link' and 'set_entities'."},
                "target_ids":       {"type": "array",   "items": {"type": "string"}, "description": "Target memory ObjectID hex strings (for 'link'). Omit for symmetric all-pairs."},
                "link_relation":    {"type": "string",  "description": "Relation label on forward links (source → target).", "default": "linked"},
                "inverse_relation": {"type": "string",  "description": "Relation label on back links (target → source). Defaults to link_relation."},
                "entities":         {"type": "array",   "items": {"type": "string"}, "description": "Entity list to set for operation='set_entities'. Overwrites existing entities[]."},
            },
            [],
        ),
        _spec(
            "query",
            "Query memories by filter, scope, and sort without semantic search. "
            "DIRECT ID LOOKUP: pass ids=[\"<objectid_hex>\", ...] to bypass ALL scope/ownership filters "
            "and fetch by _id across BOTH memory_episodic AND memory_semantic. "
            "Use this when you have an _id from a prior result and want exact retrieval. "
            "Do NOT use filter={'_id': '...'} — string _id values are NOT cast to ObjectId. "
            "For semantic/long-term memories use memory_recall instead.",
            {
                "filter":   {"type": "object",  "description": "MongoDB filter dict to narrow results. Do NOT put _id here — use the ids parameter for ID lookup."},
                "limit":    {"type": "integer", "description": "Maximum documents to return.", "default": 20},
                "scope":    {"type": "string",  "description": "Collection scope: 'episodic' (default), 'strategies'. Ignored when ids is provided.", "default": "episodic"},
                "sort_by":  {"type": "string",  "description": "Field to sort by.", "default": "created_at"},
                "sort_dir": {"type": "string",  "description": "Sort direction: 'asc' or 'desc'.", "default": "desc"},
                "ids":      {"type": "array",   "items": {"type": "string"}, "description": "List of ObjectID hex strings for direct ID fetch. Bypasses ALL scope and ownership filters. Searches both memory_episodic and memory_semantic. Use when you have an _id from a prior result. Any filter fields are applied on top of the ID match."},
            },
            [],
        ),
        _spec(
            "list_sessions",
            "List sessions using $group aggregation returning memory_count and last_updated_at per session.",
            {
                "filter": {"type": "object",  "description": "Optional filter applied to session lookup."},
                "limit":  {"type": "integer", "description": "Maximum sessions to return.", "default": 20},
            },
            [],
        ),
        _spec(
            "schema_declare",
            "Declare or update a memory schema definition. Stored in memory_semantic (memory_type='schema:<name>') with fields inside payload, matching GoMCP format.",
            {
                "schema_name": {"type": "string", "description": "Unique name for the schema."},
                "fields":      {"type": "object", "description": "Field definitions. Each key is a field name; value is {type, description, required}."},
                "content":     {"type": "string", "description": "Human-readable schema description (embedded for semantic search)."},
                "version":     {"type": "string", "description": "Schema version string, e.g. '1.0'.", "default": "1.0"},
            },
            ["schema_name", "fields"],
        ),
        _spec(
            "strategy_store",
            "Store a new strategy version in memory_semantic (insert, version_seq auto-incremented) or update an existing one in-place (_id supplied). "
            "On insert: the previous doc's superseded_by is updated to point to the new one. "
            "Strategies teach the LLM how to remember, not what was stored.",
            {
                "name":                  {"type": "string", "description": "Canonical name shared across all versions of this strategy."},
                "context":               {"type": "string", "description": "Natural-language description of what this strategy does. Gets embedded and stored as the content field."},
                "_id":                   {"type": "string", "description": "ObjectId hex of an existing strategy to update in-place. Re-embeds context and replaces payload if provided. version_seq unchanged."},
                "tags":                  {"type": "array",  "items": {"type": "string"}, "description": "Tags for coarse filtering."},
                "entities":              {"type": "array",  "items": {"type": "string"}, "description": "Entity labels for graph linking and entity-filter recall."},
                "username":              {"type": "string", "description": "Owner username for scope-based visibility."},
                "session_id":            {"type": "string", "description": "Session identifier to group related memories."},
                "scope":                 {"type": "integer", "description": "Visibility scope (0=shared,10=agent,20=user,30=user_session,40=user_session_agent). -1 defaults to user_session."},
                "importance":            {"type": "number",  "description": "Importance score 0.0-1.0. Default 0.90. Auto-overridden to 0.98 when scope=0.", "default": 0.90},
                "decay_rate":            {"type": "number",  "description": "Decay rate for composite scoring. Default 0.001. Auto-overridden to 0.0 when scope=0.", "default": 0.001},
                "memory_type":           {"type": "string",  "description": "Memory type tag. Defaults to 'strategy'. Override for sub-types e.g. 'strategy:routing'."},
                "payload":               {"type": "object", "description": "Extra metadata. For routing_pattern: {tools, playbook, output_hint, extends}."},
                "schema_version":        {"type": "string", "description": "Schema name to validate payload against."},
                "superseded_by":         {"type": "string", "description": "Only applies when _id is supplied (update path). Sets superseded_by on the existing doc. Ignored on insert — managed by code."},
                "related_doc_ids":        {"type": "array",  "items": {"type": "string"}, "description": "Memory _id hex strings to bidirectionally link to this strategy after store. Works on both insert and update paths."},
                "link_relation":          {"type": "string", "description": "Relation label for the links. Default 'related'."},
            },
            ["name", "context"],
        ),
        _spec(
            "strategy_recall",
            "Recall strategies. name → most recent version by default (include_history=True for full version history sorted by version_seq desc); "
            "query → semantic $rankFusion (vector+BM25); tags only → exact tag match sorted by hit_count; nothing → top by hit_count. "
            "Returns parent_playbook when extends is set.",
            {
                "query":                {"type": "string",  "description": "Describe the retrieval goal in natural language. Triggers semantic $rankFusion search."},
                "name":                 {"type": "string",  "description": "Exact strategy name for direct lookup. Returns most recent version. Takes precedence over query."},
                "limit":                {"type": "integer", "description": "Number of candidate strategies to return.", "default": 5},
                "similarity_threshold": {"type": "number",  "description": "Minimum boosted score threshold (0.0-1.0).", "default": 0.0},
                "tags":                 {"type": "array",  "items": {"type": "string"}, "description": "Filter by tags. Without a query, performs exact tag match sorted by hit_count."},
                "include_history":      {"type": "boolean", "description": "When True and name is provided, returns all versions sorted by version_seq desc. Default False returns only the most recent version.", "default": False},
            },
            [],
        ),
        _spec(
            "get_instructions",
            "Return the full agent operating instructions string configured at service startup.",
            {},
            [],
        ),
    ]
