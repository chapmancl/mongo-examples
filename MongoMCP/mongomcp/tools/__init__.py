"""
mongomcp.tools — Query tool registration for the MongoDB MCP server.

Public API: register_query_tools(mcp, mongo_server, llm_client, endpoint_tools) -> dispatch dict

Call this after creating the FastMCP instance and before calling mcp.http_app() so that
all query tools are registered and share the same auth provider.

Returns a dict of {tool_name: fn} suitable for merging into _TOOL_DISPATCH in mongo_mcp.py.

Generic tools (upsert_document, aggregate_query, get_unique_values, get_collection_info)
are registered directly with @mcp.tool() using closures over mongo_server.

Collection-pinned tools (vector_search, text_search, geospatial_search) are registered
under their config-driven names from endpoint_tools using Tool.from_function + mcp.add_tool(),
so multiple config entries can share the same underlying handler with different
collection/index values.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Annotated

from pydantic import Field
from pymongo.errors import PyMongoError
from fastmcp.tools import Tool
from fastmcp.server.dependencies import get_access_token, AccessToken

from .handlers import build_query_handler_fns
from .multi_step import build_multi_step_handler

logger = logging.getLogger(__name__)


def register_query_tools(mcp, mongo_server, llm_client, endpoint_tools: dict, middleware=None) -> dict:
    """
    Register all query MCP tools on the given FastMCP instance.

    Parameters
    ----------
    mcp           : FastMCP instance
    mongo_server  : MongoDBQueryServer — executes MongoDB queries
    llm_client    : ServerBedrockClient — used for generate_embedding
    endpoint_tools: dict from MongoDB config (middleware.endpoint_tools)

    Returns
    -------
    dict mapping tool_name -> fn for _TOOL_DISPATCH.
    """
    dispatch = {}
    handler_fns = build_query_handler_fns(mongo_server, llm_client)
    # Multi-step read-only functions (config-driven query chains) share the same
    # config-name -> handler registration path as the other pinned handlers.
    handler_fns["multi_step"] = build_multi_step_handler(mongo_server, llm_client)

    # ---- Generic tools (LLM supplies collection as a normal parameter) ----

    @mcp.tool()
    async def upsert_document(
        collection: Annotated[str, Field(description="Name of the MongoDB collection to upsert into.")],
        filter: Annotated[Any, Field(description="Filter to find the document to update.")],
        update: Annotated[Any, Field(description="Update data for the document.")],
        token: AccessToken = None,
    ) -> Dict[str, Any]:
        """Upsert a document in the specified MongoDB collection."""
        scopes = set()
        client_id = ""
        if token is None:
            token = get_access_token()
        if isinstance(token, dict):
            scopes = set(token.get("scope", []))
            client_id = token.get("agent_key", "")
        elif token is not None:
            scopes = set(token.scopes)
            client_id = token.client_id
        if "write" not in scopes:
            logger.error(f"Insufficient scope for upsert_document: write permission required for agent {client_id}")
            return {"error": "Insufficient scope: this agent does not have write permission."}
        try:
            # LLMs sometimes pass dicts as JSON strings — coerce before sending to MongoDB.
            if isinstance(filter, str):
                filter = json.loads(filter)
            if isinstance(update, str):
                update = json.loads(update)
            doc_id = await mongo_server.upsert_document(collection, filter, update)
            return {"message": f"Document {doc_id} upserted successfully in collection '{collection}'."}
        except (ValueError, PyMongoError) as e:
            logger.error(f"Upsert document failed: {e}")
            return {"error": f"Error executing upsert_document: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error in upsert_document: {e}")
            return {"error": f"Unexpected error executing upsert_document: {str(e)}"}

    dispatch["upsert_document"] = getattr(upsert_document, "fn", upsert_document)

    @mcp.tool()
    async def get_unique_values(
        collection: Annotated[str, Field(description="Name of the MongoDB collection.")],
        field: Annotated[str, Field(description="Field name to get unique values for.")],
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            pipeline = [
                {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
                {"$match": {"_id": {"$ne": None}}},
                {"$sort": {"count": -1}},
            ]
            results = await mongo_server.agg_pipeline(collection, pipeline)
            total_docs = await mongo_server.get_collection(collection).count_documents({})
            for result in results:
                result["percentage"] = round((result["count"] / total_docs) * 100, 2)
            return {
                "field": field,
                "unique_values": results,
                "total_unique_count": len(results),
                "total_documents": total_docs,
            }
        except Exception as e:
            logger.error(f"Get unique values failed: {e}")
            return {"error": f"Error executing get_unique_values: {str(e)}"}

    dispatch["get_unique_values"] = getattr(get_unique_values, "fn", get_unique_values)

    @mcp.tool()
    async def get_collection_info() -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            return await mongo_server.get_collection_info()
        except Exception as e:
            logger.error(f"Get collection info failed: {e}")
            return {"error": f"Error executing get_collection_info: {str(e)}"}

    dispatch["get_collection_info"] = getattr(get_collection_info, "fn", get_collection_info)

    @mcp.tool()
    async def aggregate_query(
        collection: Annotated[str, Field(description="Name of the MongoDB collection.")],
        pipeline: Annotated[List[Dict[str, Any]], Field(description="MongoDB aggregation pipeline as a list of stage objects.")],
        limit: Annotated[Optional[int], Field(default=None, description="Optional limit to apply to the results.", ge=1, le=1000)] = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not pipeline or not isinstance(pipeline, list):
                return {"error": "pipeline must be a non-empty list of aggregation stages"}
            for i, stage in enumerate(pipeline):
                if not isinstance(stage, dict):
                    return {"error": f"pipeline stage {i} must be a dictionary, got {type(stage)}"}
                if not stage:
                    return {"error": f"pipeline stage {i} cannot be empty"}
            final_pipeline = pipeline.copy()
            if limit is not None:
                if not any("$limit" in stage for stage in pipeline):
                    final_pipeline.append({"$limit": limit})
            _t0 = time.monotonic()
            results = await mongo_server.agg_pipeline(collection, final_pipeline)
            _elapsed_ms = int((time.monotonic() - _t0) * 1000)
            logger.info(f"Aggregation query returned {len(results)} results in {_elapsed_ms} ms")
            return {
                "results": results,
                "count": len(results),
                "query_info": {
                    "pipeline": final_pipeline,
                    "stages_count": len(final_pipeline),
                    "limit_applied": limit,
                },
            }
        except PyMongoError as e:
            logger.error(f"Aggregation query failed: {e}")
            return {"error": f"Error executing aggregation pipeline: {str(e)}"}
        except json.JSONDecodeError as e:
            logger.error(f"JSON serialization failed: {e}")
            return {"error": f"Error serializing results: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error in aggregate_query: {e}")
            return {"error": f"Unexpected error executing aggregate_query: {str(e)}"}

    dispatch["aggregate_query"] = getattr(aggregate_query, "fn", aggregate_query)

    # ---- Collection-pinned tools (registered under config-driven names) ----
    # Each endpoint_tools entry whose 'handler' (or name itself) matches a pinned handler
    # gets a Tool.from_function wrapper registered under the config-defined name.
    # This is what allows "search_listings" and "search_repairs" to both map to
    # vector_search with different collection values.
    for tool_name, tool_cfg in endpoint_tools.items():
        handler_name = tool_cfg.get("handler", tool_name)
        fn = handler_fns.get(handler_name)
        if fn is None:
            continue
        # output_schema=None disables FastMCP's inferred outputSchema. These handlers return
        # raw Mongo documents (BSON ObjectId/datetime in _id etc.), which FastMCP cannot
        # serialize into structuredContent — with an inferred schema that yields
        # "outputSchema defined but no structured output returned". Results are consumed as
        # text (content[0].text), so no structured schema is needed.
        tool_obj = Tool.from_function(fn, name=tool_name, output_schema=None)
        mcp.add_tool(tool_obj)
        dispatch[tool_name] = fn
        logger.info(f"Registered collection tool '{tool_name}' -> handler '{handler_name}'")

    def ensure_tool_registered(tool_name: str, tool_cfg: dict) -> bool:
        """Register a config-driven tool with FastMCP on demand (idempotent).

        Startup only registers tools that existed then. A function defined/redefined on
        ANY container mid-session (e.g. a dynamic fn_ query function) is written to Mongo
        and picked up by the middleware's fresh-read gate/injection — but FastMCP still has
        no handler for it, so dispatch fails with 'Unknown tool'. This registers it against
        the shared handler the moment it is first invoked, so define->invoke works with no
        container restart. Safe to call on every call: it no-ops once the tool is known.
        """
        if tool_name in dispatch:
            return True
        handler_name = (tool_cfg or {}).get("handler", tool_name)
        fn = handler_fns.get(handler_name)
        if fn is None:
            return False
        try:
            mcp.add_tool(Tool.from_function(fn, name=tool_name, output_schema=None))
            dispatch[tool_name] = fn
            logger.info(f"Dynamically registered collection tool '{tool_name}' -> handler '{handler_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to dynamically register tool '{tool_name}': {e}")
            return False

    # Expose on-demand registration to the middleware so on_call_tool can register a
    # freshly-defined function before dispatch (define->invoke without restart).
    if middleware is not None:
        middleware.ensure_tool_registered = ensure_tool_registered

    return dispatch
