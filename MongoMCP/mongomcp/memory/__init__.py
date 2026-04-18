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
from typing import Any, Dict

from pydantic import Field
from typing import Annotated, List, Optional

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
        payload: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Arbitrary structured metadata.")] = None,
        username: Annotated[Optional[str], Field(default=None, description="Username storing the memory.")] = None,
        agent_id: Annotated[Optional[str], Field(default=None, description="Agent/owner ID for the memory. Used as the ownership field for recall scoping. Defaults to username if not set.")] = None,
        schema_version: Annotated[Optional[str], Field(default=None, description="Declared schema name to validate the payload against. Warnings returned but write still succeeds.")] = None,
        is_isolated: Annotated[bool, Field(default=False, description="When True, this memory is private to the storing agent and not visible to other agents. When False (default), the memory is shared and visible to all agents.")] = False,
    ):
        """Store a memory with auto-linking to similar existing memories via vector similarity."""
        return await raw_fns["intake"](
            content=content, memory_type=memory_type, importance=importance,
            decay_rate=decay_rate, session_id=session_id, tags=tags, entities=entities,
            payload=payload, username=username, agent_id=agent_id, schema_version=schema_version,
            is_isolated=is_isolated,
        )

    @mcp.tool()
    async def recall(
        query: Annotated[str, Field(description="Natural language query to find relevant memories.")],
        session_id: Annotated[Optional[str], Field(default=None, description="Restrict recall to a specific session (episodic pre-filter).")] = None,
        agent_id: Annotated[Optional[str], Field(default=None, description="Agent/owner ID to scope results. Filters agent_id on semantic, username on episodic.")] = None,
        username: Annotated[Optional[str], Field(default=None, description="Username to scope episodic results (alias for agent_id on episodic docs).")] = None,
        scope: Annotated[str, Field(default="all", description="Collection scope: 'all' (default), 'episodic', or 'semantic'.")] = "all",
        limit: Annotated[int, Field(default=5, description="Maximum number of memories to return.", ge=1, le=50)] = 5,
        num_candidates: Annotated[int, Field(default=150, description="ANN candidate pool size for vector search.", ge=10, le=1000)] = 150,
        score_threshold: Annotated[float, Field(default=0.0, description="Minimum composite score threshold (0.0-1.0).", ge=0.0, le=1.0)] = 0.0,
        importance_threshold: Annotated[float, Field(default=0.0, description="Minimum importance score pre-filter (episodic only).", ge=0.0, le=1.0)] = 0.0,
        memory_types: Annotated[Optional[List[str]], Field(default=None, description="Filter by memory_type values.")] = None,
        tags: Annotated[Optional[List[str]], Field(default=None, description="Filter by tags (any match).")] = None,
        entities: Annotated[Optional[List[str]], Field(default=None, description="Filter by named entities (Python post-filter).")] = None,
    ):
        """Recall relevant memories via semantic search across episodic and semantic collections, with graph expansion and composite scoring."""
        return await raw_fns["recall"](
            query=query, session_id=session_id,
            agent_id=agent_id, username=username,
            scope=scope, limit=limit, num_candidates=num_candidates,
            score_threshold=score_threshold, importance_threshold=importance_threshold,
            memory_types=memory_types, tags=tags, entities=entities,
        )

    @mcp.tool()
    async def reflect(
        session_id: Annotated[str, Field(description="Session ID to summarise and reflect on.")],
        keep_session: Annotated[bool, Field(default=False, description="When True, the promoted summary retains session_id (multi-session continuity). When False it is stored as a session-independent long-term memory.")] = False,
    ):
        """Summarise all memories for a session via LLM and store the result as a session:summary memory."""
        return await raw_fns["reflect"](session_id=session_id, keep_session=keep_session)

    @mcp.tool()
    async def query(
        filter: Annotated[Optional[Dict[str, Any]], Field(default=None, description="MongoDB filter dict to narrow results.")] = None,
        limit: Annotated[int, Field(default=20, description="Maximum documents to return.", ge=1, le=500)] = 20,
        scope: Annotated[str, Field(default="episodic", description="Collection scope: 'episodic' or 'strategies'.")] = "episodic",
        sort_by: Annotated[str, Field(default="created_at", description="Field to sort by.")] = "created_at",
        sort_dir: Annotated[str, Field(default="desc", description="Sort direction: 'asc' or 'desc'.")] = "desc",
    ):
        """Query memories by filter, scope, and sort without semantic search."""
        return await raw_fns["query"](
            filter=filter, limit=limit, scope=scope,
            sort_by=sort_by, sort_dir=sort_dir,
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
        agent_id: Annotated[Optional[str], Field(default=None, description="Agent or client ID declaring the schema.")] = None,
    ):
        """Declare or update a memory schema definition (upsert by schema_name). Stored in memory_semantic as memory_type='schema:<name>' with fields inside payload."""
        return await raw_fns["schema_declare"](
            schema_name=schema_name, fields=fields,
            content=content, version=version, agent_id=agent_id,
        )

    @mcp.tool()
    async def strategy_store(
        name: Annotated[str, Field(description="Unique name for this strategy (e.g. 'high_importance_facts').")],
        description: Annotated[str, Field(description="Natural-language description of what this strategy retrieves. Gets embedded for semantic recall.")],
        tags: Annotated[Optional[List[str]], Field(default=None, description="Tags for categorisation. Tool names from payload.tools are auto-added.")] = None,
        session_id: Annotated[Optional[str], Field(default=None, description="Optional session context.")] = None,
        pipeline_template: Annotated[Optional[Dict[str, Any]], Field(default=None, description="MongoDB aggregation pipeline with {{param}} placeholders.")] = None,
        filter_template: Annotated[Optional[Dict[str, Any]], Field(default=None, description="MongoDB filter with {{param}} placeholders applied at recall time.")] = None,
        shard_keys: Annotated[Optional[List[str]], Field(default=None, description="Shard keys this strategy routes to.")] = None,
        metadata_requirements: Annotated[Optional[List[str]], Field(default=None, description="Payload fields this strategy requires.")] = None,
        payload: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Extra structured metadata. For routing_pattern: {tools, playbook, output_hint, extends}.")] = None,
        schema_version: Annotated[Optional[str], Field(default=None, description="Schema name to validate payload against.")] = None,
    ):
        """Store or update a retrieval strategy as a vector-searchable document in memory_semantic (memory_type='strategy:<name>')."""
        return await raw_fns["strategy_store"](
            name=name, description=description, tags=tags, session_id=session_id,
            pipeline_template=pipeline_template, filter_template=filter_template,
            shard_keys=shard_keys, metadata_requirements=metadata_requirements,
            payload=payload, schema_version=schema_version,
        )

    @mcp.tool()
    async def strategy_recall(
        query: Annotated[Optional[str], Field(default=None, description="Describe the retrieval goal. Semantic search via $rankFusion (vector+BM25).")] = None,
        name: Annotated[Optional[str], Field(default=None, description="Exact strategy name for direct lookup. Takes precedence over query.")] = None,
        limit: Annotated[int, Field(default=5, description="Maximum strategies to return.", ge=1, le=50)] = 5,
        similarity_threshold: Annotated[float, Field(default=0.0, description="Minimum boosted score (0.0-1.0).", ge=0.0, le=1.0)] = 0.0,
        tags: Annotated[Optional[List[str]], Field(default=None, description="Filter by tags. Without a query, performs exact tag match sorted by hit_count.")] = None,
    ):
        """Recall strategies. name → exact lookup; query → semantic $rankFusion (vector+BM25); tags only → exact tag match sorted by hit_count; nothing → top by hit_count. Returns parent_playbook when extends is set."""
        return await raw_fns["strategy_recall"](
            query=query, name=name,
            limit=limit, similarity_threshold=similarity_threshold, tags=tags,
        )

    @mcp.tool()
    async def get_instructions():
        """Return the full agent operating instructions string configured at service startup."""
        result = await raw_fns["get_instructions"]()
        instr = result.get("instructions", "") if isinstance(result, dict) else ""
        logger.info("[PIPELINE] __init__.get_instructions: returning instructions len=%d", len(instr))
        return result

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
                "payload":        {"type": "object", "description": "Arbitrary structured metadata."},
                "username":       {"type": "string", "description": "Username storing the memory."},
                "agent_id":       {"type": "string", "description": "Agent/owner ID for the memory. Used for recall ownership scoping. Defaults to username if not set."},
                "schema_version": {"type": "string", "description": "Declared schema name to validate payload against."},
                "is_isolated":    {"type": "boolean", "description": "When True, memory is private to this agent only. When False (default), memory is shared and visible to all agents.", "default": False},
            },
            ["content"],
        ),
        _spec(
            "recall",
            "Recall relevant memories via semantic search across episodic and semantic collections, with graph expansion and composite scoring.",
            {
                "query":                {"type": "string",  "description": "Natural language query to find relevant memories."},
                "session_id":           {"type": "string",  "description": "Restrict recall to a specific session (episodic pre-filter)."},
                "agent_id":             {"type": "string",  "description": "Agent/owner ID to scope results."},
                "username":             {"type": "string",  "description": "Username to scope episodic results (alias for agent_id on episodic docs)."},
                "scope":                {"type": "string",  "description": "Collection scope: 'all' (default), 'episodic', or 'semantic'.", "default": "all"},
                "limit":                {"type": "integer", "description": "Maximum memories to return.", "default": 5},
                "num_candidates":       {"type": "integer", "description": "ANN candidate pool size for vector search.", "default": 150},
                "score_threshold":      {"type": "number",  "description": "Minimum composite score threshold (0.0-1.0).", "default": 0.0},
                "importance_threshold": {"type": "number",  "description": "Minimum importance score pre-filter (episodic only).", "default": 0.0},
                "memory_types":         {"type": "array",   "items": {"type": "string"}, "description": "Filter by memory_type values."},
                "tags":                 {"type": "array",   "items": {"type": "string"}, "description": "Filter by tags (any match)."},
                "entities":             {"type": "array",   "items": {"type": "string"}, "description": "Filter by named entities (Python post-filter)."},
            },
            ["query"],
        ),
        _spec(
            "reflect",
            "Summarise all memories for a session via LLM and store the result as a session:summary memory.",
            {
                "session_id":   {"type": "string", "description": "Session ID to summarise and reflect on."},
                "keep_session": {"type": "boolean", "description": "When True, the promoted summary retains session_id.", "default": False},
            },
            ["session_id"],
        ),
        _spec(
            "query",
            "Query memories by filter, scope, and sort without semantic search.",
            {
                "filter":   {"type": "object", "description": "MongoDB filter dict to narrow results."},
                "limit":    {"type": "integer", "description": "Maximum documents to return.", "default": 20},
                "scope":    {"type": "string",  "description": "Collection scope: 'episodic', 'strategies', or 'all'.", "default": "episodic"},
                "sort_by":  {"type": "string",  "description": "Field to sort by.", "default": "created_at"},
                "sort_dir": {"type": "string",  "description": "Sort direction: 'asc' or 'desc'.", "default": "desc"},
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
                "agent_id":    {"type": "string", "description": "Agent or client ID declaring the schema."},
            },
            ["schema_name", "fields"],
        ),
        _spec(
            "strategy_store",
            "Store a retrieval strategy as a vector-searchable document in memory_semantic (memory_type='strategy:<name>'). Strategies teach the LLM how to remember, not what was stored. Recalled by embedding similarity.",
            {
                "name":                  {"type": "string", "description": "Unique, human-readable name for this strategy."},
                "description":           {"type": "string", "description": "Natural-language description of what this strategy retrieves. Gets embedded."},
                "pipeline_template":     {"type": "object", "description": "Optional MongoDB aggregation pipeline with {{param}} placeholders."},
                "shard_keys":            {"type": "array",  "items": {"type": "string"}, "description": "Shard keys this strategy routes to."},
                "filter_template":       {"type": "object", "description": "Optional MongoDB filter with {{param}} placeholders."},
                "tags":                  {"type": "array",  "items": {"type": "string"}, "description": "Tags for coarse filtering."},
                "metadata_requirements": {"type": "array",  "items": {"type": "string"}, "description": "Payload fields this strategy requires."},
                "payload":               {"type": "object", "description": "Extra metadata. For routing_pattern: {tools, playbook, output_hint, extends}."},
                "schema_version":        {"type": "string", "description": "Schema name to validate payload against."},
            },
            ["name", "description"],
        ),
        _spec(
            "strategy_recall",
            "Recall strategies. name → exact lookup; query → semantic $rankFusion (vector+BM25); tags only (no query) → fulltext search on tag values; nothing → top strategies by hit_count. Returns parent_playbook when extends is set.",
            {
                "query":                {"type": "string",  "description": "Describe the retrieval goal in natural language. Triggers semantic $rankFusion search."},
                "name":                 {"type": "string",  "description": "Exact strategy name for direct lookup. Takes precedence over query."},
                "limit":                {"type": "integer", "description": "Number of candidate strategies to return.", "default": 5},
                "similarity_threshold": {"type": "number",  "description": "Minimum boosted score threshold (0.0-1.0).", "default": 0.0},
                "tags":                 {"type": "array",  "items": {"type": "string"}, "description": "Filter by tags. Without a query, performs exact tag match sorted by hit_count."},
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
