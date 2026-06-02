"""
MCP tool handler functions for the memory subpackage.

These are plain async functions (not decorated yet). Decoration with
@mcp.tool() happens inside register_memory_tools() so that the service
instance can be captured via closure.

Tool contracts are intentionally kept compatible with the go-mongo-memo
MCP server so agents can target either implementation.
"""

from typing import Any, Dict, List, Optional
import logging
import traceback

from .memservice import MemoryService

logger = logging.getLogger(__name__)


def build_memory_tool_fns(svc: MemoryService):
    """
    Return a dict of {tool_name: async_fn} referencing the given MemoryService.

    The caller (register_memory_tools) wraps each function with @mcp.tool()
    and collects the decorated .fn references for _TOOL_DISPATCH.
    """

    async def intake(
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
        """Store a memory in the episodic collection with auto-linking to similar memories."""
        try:
            return await svc.intake(
                content=content,
                memory_type=memory_type,
                importance=importance,
                decay_rate=decay_rate,
                session_id=session_id,
                tags=tags,
                entities=entities,
                payload=payload,
                username=username,
                agent_id=agent_id,
                schema_version=schema_version,
                is_isolated=is_isolated,
            )
        except Exception as exc:
            logger.error("intake failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"intake failed: {exc}"}

    async def recall(
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
        """Recall relevant memories using semantic vector search with one-hop graph expansion and composite scoring."""
        try:
            return await svc.recall(
                query=query,
                session_id=session_id,
                agent_id=agent_id,
                username=username,
                scope=scope,
                limit=limit,
                num_candidates=num_candidates,
                score_threshold=score_threshold,
                importance_threshold=importance_threshold,
                memory_types=memory_types,
                tags=tags,
                entities=entities,
            )
        except Exception as exc:
            logger.error("recall failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"recall failed: {exc}"}

    async def reflect(session_id: str, keep_session: bool = False) -> dict:
        """Summarise all session memories with the LLM and store the result as a session:summary memory."""
        try:
            return await svc.reflect(session_id=session_id, keep_session=keep_session)
        except Exception as exc:
            logger.error("reflect failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"reflect failed: {exc}"}

    async def query(
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        scope: str = "episodic",
        sort_by: str = "created_at",
        sort_dir: str = "desc",
    ) -> dict:
        """Query memories by filter dict, scope (episodic | strategies), and sort options."""
        try:
            return await svc.query(
                filter=filter,
                limit=limit,
                scope=scope,
                sort_by=sort_by,
                sort_dir=sort_dir,
            )
        except Exception as exc:
            logger.error("query failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"query failed: {exc}"}

    async def list_sessions(
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> dict:
        """List sessions by finding session:summary memories, with fallback to grouped session_id."""
        try:
            return await svc.list_sessions(filter=filter, limit=limit)
        except Exception as exc:
            logger.error("list_sessions failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"list_sessions failed: {exc}"}

    async def schema_declare(
        schema_name: str,
        fields: Dict[str, Any],
        content: Optional[str] = None,
        version: str = "1.0",
        agent_id: Optional[str] = None,
    ) -> dict:
        """Declare or update a memory schema definition (upsert by schema_name). Stored in memory_semantic with fields inside payload."""
        try:
            return await svc.schema_declare(
                schema_name=schema_name,
                fields=fields,
                content=content,
                version=version,
                agent_id=agent_id,
            )
        except Exception as exc:
            logger.error("schema_declare failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"schema_declare failed: {exc}"}

    async def strategy_store(
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
        """Store or update a reusable retrieval strategy. memory_type is set to 'strategy:<name>'."""
        try:
            return await svc.strategy_store(
                name=name,
                description=description,
                tags=tags,
                session_id=session_id,
                pipeline_template=pipeline_template,
                filter_template=filter_template,
                shard_keys=shard_keys,
                metadata_requirements=metadata_requirements,
                payload=payload,
                schema_version=schema_version,
            )
        except Exception as exc:
            logger.error("strategy_store failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"strategy_store failed: {exc}"}

    async def strategy_recall(
        query: Optional[str] = None,
        name: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> dict:
        """Recall strategies. name → exact lookup; query → semantic $rankFusion (vector+BM25); tags only (no query) → exact tag match sorted by hit_count; nothing → top strategies by hit_count."""
        try:
            return await svc.strategy_recall(
                query=query,
                name=name,
                limit=limit,
                similarity_threshold=similarity_threshold,
                tags=tags,
            )
        except Exception as exc:
            logger.error("strategy_recall failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"strategy_recall failed: {exc}"}

    async def get_instructions() -> dict:
        """Return the full agent operating instructions string configured at service startup."""
        try:
            result = await svc.get_instructions()
            instr = result.get("instructions", "")
            logger.info("[PIPELINE] tools.get_instructions: result instructions len=%d", len(instr))
            return result
        except Exception as exc:
            logger.error("get_instructions failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"get_instructions failed: {exc}"}

    return {
        "intake": intake,
        "recall": recall,
        "reflect": reflect,
        "query": query,
        "list_sessions": list_sessions,
        "schema_declare": schema_declare,
        "strategy_store": strategy_store,
        "strategy_recall": strategy_recall,
        "get_instructions": get_instructions,
    }
