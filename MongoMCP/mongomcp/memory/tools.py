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
        payload_push: Optional[Dict[str, Any]] = None,
        username: Optional[str] = None,
        agent_id: Optional[str] = None,
        schema_version: Optional[str] = None,
        scope: int = -1,
        related_docs: Optional[List[Dict[str, Any]]] = None,
        _id: Optional[str] = None,
    ) -> dict:
        """Store a memory in the appropriate collection with optional explicit related_docs links and near-duplicate warning."""
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
                payload_push=payload_push,
                username=username,
                agent_id=agent_id,
                schema_version=schema_version,
                scope=scope,
                related_docs=related_docs,
                _id=_id,
            )
        except Exception as exc:
            logger.error("intake failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"intake failed: {exc}"}

    async def recall(
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
        """Recall memories via semantic vector search or direct entity filter, with multi-hop BFS graph expansion and composite scoring."""
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
                depth=depth,
                depth_relations=depth_relations,
                output_format=output_format,
            )
        except Exception as exc:
            logger.error("recall failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"recall failed: {exc}"}

    async def reflect(
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
        """Multi-operation memory maintenance: summarise a session (storing agent_id + username on the summary), create explicit bidirectional links, or set entities on existing memories."""
        try:
            return await svc.reflect(
                session_id=session_id,
                operation=operation,
                memory_ids=memory_ids,
                target_ids=target_ids,
                link_relation=link_relation,
                inverse_relation=inverse_relation,
                entities=entities,
                agent_id=agent_id,
                username=username,
            )
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
        ids: Optional[List[str]] = None,
    ) -> dict:
        """Query memories by filter dict, scope (episodic | strategies), sort options, or direct ID list."""
        try:
            return await svc.query(
                filter=filter,
                limit=limit,
                scope=scope,
                sort_by=sort_by,
                sort_dir=sort_dir,
                ids=ids,
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
        """Store a new strategy version (insert) or update an existing one (_id supplied). context is embedded and stored as content."""
        try:
            return await svc.strategy_store(
                name=name,
                context=context,
                _id=_id,
                tags=tags,
                entities=entities,
                username=username,
                agent_id=agent_id,
                scope=scope,
                session_id=session_id,
                importance=importance,
                decay_rate=decay_rate,
                memory_type=memory_type,
                payload=payload,
                schema_version=schema_version,
                superseded_by=superseded_by,
                related_doc_ids=related_doc_ids,
                link_relation=link_relation,
            )
        except Exception as exc:
            logger.error("strategy_store failed: %s", exc)
            logger.debug("".join(traceback.format_exception(None, exc, exc.__traceback__)))
            return {"error": f"strategy_store failed: {exc}"}

    async def strategy_recall(
        query: Optional[str] = None,
        name: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.5,
        tags: Optional[List[str]] = None,
        include_history: bool = False,
    ) -> dict:
        """Recall strategies. name → most recent version (include_history=True for all versions); query → semantic $rankFusion; tags only → exact tag match sorted by hit_count; nothing → top by hit_count."""
        try:
            return await svc.strategy_recall(
                query=query,
                name=name,
                limit=limit,
                similarity_threshold=similarity_threshold,
                tags=tags,
                include_history=include_history,
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
            logger.debug("[PIPELINE] tools.get_instructions: result instructions len=%d", len(instr))
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
