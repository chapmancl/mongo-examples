from fastmcp.server.middleware.middleware import Middleware, MiddlewareContext, CallNext
from typing import List, Dict, Any, Optional
from bson import ObjectId
import mcp.types as mt
import jwt
import jwt.exceptions
import datetime
import logging
import os
import json
import traceback
from .mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)

_MEMORY_TOOLS = {
    "intake", "recall", "reflect", "query", "list_sessions",
    "schema_declare", "strategy_store", "strategy_recall", "get_instructions",
    # Agent tools — registered on agent_mcp, pass through middleware security gate.
    "run_prompt",
}

# Raw Python handler names that are never registered directly as @mcp.tool().
# register_collection_tools() wraps these under config-driven names instead.
_COLLECTION_PINNED_HANDLERS = frozenset({"vector_search", "text_search", "geospatial_search", "hybrid_search"})


SHOW_ONCE = 0
class MongoMCPMiddleware(Middleware):
    """
    FastMCP Middleware is the central point for connecting to MongoDB config database.
    handles intercept and print on_list_tools output
    middleware is always connected to the mongo MCP config collection.
    use this to make core config requests and send logging info to the central MCP config collections
    """
    def __init__(self, settings):
        super().__init__()
        self.endpoint_name = settings.TOOL_NAME
        self._is_local = settings.IS_LOCAL
        logger.info("MongoMCPMiddleware initialized")
        self.mongo_client = MongoDBClient(settings)
        self.ANNOTATIONS = None
        self.active_endpoints = [self.endpoint_name]
        self.endpoint_tools = {}
        self.load_annotations()

    def load_annotations(self):
        """Load tool annotations from the JSON out of mongo"""
        global SHOW_ONCE
        try:
            if self.mongo_client.sync_connect_to_mongodb():
                if SHOW_ONCE < 1:
                    logger.info(f"loading dynamic config for endpoint {self.endpoint_name}")
                # load the config for this specific tool, then we load it for everything so we can return all tools on the shared endpoint
                # make 2 calls because we need this config regardless of active state
                doc = self.mongo_client.get_collection().find_one({"Name": self.endpoint_name})
                self.ANNOTATIONS = doc
                self.endpoint_tools = self.ANNOTATIONS.get('tools', {})
                #### load all active endpoints to return configs
                if self._is_local:
                    if SHOW_ONCE < 1:
                        logger.info(f"Running in local mode, loading only the current endpoint config for {self.endpoint_name}")
                        SHOW_ONCE += 1
                else:
                    self.active_endpoints = list(self.mongo_client.get_collection().distinct("Name",{ "active": True}))
                    if SHOW_ONCE < 1:
                        logger.info(f"Running in dynamic mode, loading all available endpoint configs for endpoints: {self.active_endpoints}")
                        SHOW_ONCE += 1

                return doc
        except ConnectionError as ce:
            logger.error(f"MongoDB connection error while loading annotations for endpoint {self.endpoint_name}. check IP whitelist, networking etc.:\r\n {ce}")
            return None
        except Exception as e:
            logger.error(f"Failed to load annotations for endpoint {self.endpoint_name}:\r\n {e}")
            return None

    def refresh_active_endpoints(self) -> list:
        """Re-query active endpoints from MongoDB so newly activated entries are picked up."""
        if self._is_local:
            return self.active_endpoints
        try:
            if self.mongo_client.sync_connect_to_mongodb():
                self.active_endpoints = list(
                    self.mongo_client.get_collection().distinct("Name", {"active": True})
                )
        except Exception as e:
            logger.error(f"Failed to refresh active endpoints: {e}")
        return self.active_endpoints

    def check_authorization(self, token: str):
        """Check if the provided token is valid"""
        allowed = False
        agent_rec = None
        try:
            header = jwt.get_unverified_header(token)
            api_key = header.get("api_key")

            self.mongo_client.sync_connect_to_mongodb()
            agent_coll = self.mongo_client.get_collection("agent_identities")
            agent_rec = agent_coll.find_one({"agent_key": api_key})
            if agent_rec:
                # you should hash.... do as I say not as I do.
                # store the hash private key in secrets manager, then implement hash.
                # I think most will come in through a token service which makes this moot.
                # trying to keep the demo simple, so just verifying the token directly here.
                # in order to hash I would need a token generator service and I don't want to build that here.
                # https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/#oauth2passwordrequestform
                pvk = agent_rec.get("pvk")
                decoded_payload = jwt.decode(token,pvk, algorithms=["HS256"])
                agent_name = decoded_payload.get("agent_name")
                if decoded_payload.get("revoked", False):
                    logger.warning(f"Token for agent {agent_name}:{api_key} is revoked.")
                    return (False, None)
                agent_rec.pop("pvk")  # remove sensitive info

                if agent_name == agent_rec.get("agent_name"):
                    logger.debug(f"Authorization successful for agent: {agent_name}")
                    allowed = True

        except jwt.exceptions.InvalidTokenError  as je:
            logger.error(f"JWT decoding error: {je}")
            allowed = False
        except Exception as e:
            logger.error(f"Unexpected error during authorization check: {e}")
            allowed = False

        return (allowed, agent_rec)

    _PYTHON_TO_JSON_SCHEMA_TYPE = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "List": "array",
        "dict": "object",
        "Dict": "object",
    }

    def _python_type_to_json_schema(self, type_str: str) -> str:
        """Map a Python type annotation string (from MongoDB annotations) to a JSON Schema type."""
        if not type_str:
            return "string"
        t = type_str.strip()
        if t.startswith("Optional["):
            t = t[9:-1].strip()
        if t.startswith("List"):
            return "array"
        if t.startswith("Dict"):
            return "object"
        return self._PYTHON_TO_JSON_SCHEMA_TYPE.get(t, "string")

    def build_tools_from_annotations(self) -> List[Dict]:
        """Build Bedrock toolSpec JSON entirely from MongoDB annotations.

        Requires no FastMCP introspection — all tool names, descriptions, parameter
        types, defaults, and required lists come directly from the MongoDB config
        collection. Call this instead of get_llm_tools()/get_formatted_llm_tools()
        wherever you previously needed to first call mcp.get_tools().

        Returns:
            List of dicts in Bedrock toolSpec format, ready for LLM consumption.
        """
        self.load_annotations()
        tools_dict = []
        try:
            for tool_name, anot in self.endpoint_tools.items():
                description = anot.get("description", f"Tool: {tool_name}")
                returns = anot.get("returns")
                if returns:
                    description += f"\n\nReturns:\n\t{returns}"

                properties = {}
                for p_name, p_info in anot.get("parameters", {}).items():
                    json_type = self._python_type_to_json_schema(p_info.get("type", "str"))
                    prop = {
                        "type": json_type,
                        "description": p_info.get("description", ""),
                    }
                    if "default" in p_info and p_info["default"] is not None:
                        prop["default"] = p_info["default"]
                    properties[p_name] = prop

                tools_dict.append({
                    "toolSpec": {
                        "name": tool_name,
                        "description": description,
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": properties,
                                "required": anot.get("required", []),
                            }
                        }
                    }
                })
        except Exception as e:
            logger.error(f"Error building tools from annotations: {e}")
        return tools_dict

    def build_tools_from_all_endpoints(self) -> List[Dict]:
        """Build Bedrock toolSpec for ALL active endpoints from MongoDB, with endpoint-name prefix.

        Each tool name is prefixed as '<endpoint_name>_<tool_name>' so the HTTP dispatch
        in _make_mcp_call_fn can split on the first '_' to route to the correct MCP mount.
        Skips the current endpoint's tools — those are already covered by build_tools_from_annotations().
        In local mode, returns only the current endpoint's tools (same as build_tools_from_annotations).

        Returns:
            List of dicts in Bedrock toolSpec format with prefixed tool names.
        """
        if self._is_local:
            # Local mode: only one endpoint available; prefix current tools normally.
            tools = []
            for t in self.build_tools_from_annotations():
                t = t.copy()
                t["toolSpec"] = dict(t["toolSpec"])
                t["toolSpec"]["name"] = f"{self.endpoint_name}_{t['toolSpec']['name']}"
                tools.append(t)
            return tools

        all_tools = []
        try:
            if not self.mongo_client.sync_connect_to_mongodb():
                logger.warning("build_tools_from_all_endpoints: could not connect to MongoDB")
                return all_tools
            docs = list(self.mongo_client.get_collection().find({"active": True}))
            for doc in docs:
                endpoint_name = doc.get("Name", "")
                if not endpoint_name:
                    continue
                endpoint_tools = doc.get("tools", {})
                for tool_name, anot in endpoint_tools.items():
                    description = anot.get("description", f"Tool: {tool_name}")
                    returns = anot.get("returns")
                    if returns:
                        description += f"\n\nReturns:\n\t{returns}"
                    properties = {}
                    for p_name, p_info in anot.get("parameters", {}).items():
                        json_type = self._python_type_to_json_schema(p_info.get("type", "str"))
                        prop = {
                            "type": json_type,
                            "description": p_info.get("description", ""),
                        }
                        if "default" in p_info and p_info["default"] is not None:
                            prop["default"] = p_info["default"]
                        properties[p_name] = prop
                    all_tools.append({
                        "toolSpec": {
                            "name": f"{endpoint_name}_{tool_name}",
                            "description": description,
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": properties,
                                    "required": anot.get("required", []),
                                }
                            }
                        }
                    })
        except Exception as e:
            logger.error(f"build_tools_from_all_endpoints failed: {e}")
        return all_tools

    def inject_collection_args(self, toolname: str, kwargs: dict) -> dict:
        """Inject config-backed 'collection' and 'geo_field' into kwargs for collection-pinned handlers.

        Used by both the FastMCP on_call_tool middleware and the API-path tool_handler so that
        injection logic lives in exactly one place.

        Mutates and returns kwargs.
        """
        tool_cfg = self.endpoint_tools.get(toolname, {})
        if not tool_cfg:
            return kwargs
        handler_name = tool_cfg.get("handler", toolname)
        if handler_name not in _COLLECTION_PINNED_HANDLERS:
            return kwargs
        logger.debug("inject_collection_args: tool=%r handler=%r cfg=%r kwargs=%r", toolname, handler_name, tool_cfg, kwargs)
        current_collection = kwargs.get("collection")
        if current_collection is None or not str(current_collection).strip():
            collection = tool_cfg.get("collection")
            if not collection or not str(collection).strip():
                raise ValueError(
                    f"Tool '{toolname}' (handler: '{handler_name}') has no valid "
                    f"'collection' field in its config. Add 'collection' to the "
                    f"tool's config entry in MongoDB."
                )
            kwargs["collection"] = collection
            logger.debug("inject_collection_args: injected collection=%r for tool=%r", collection, toolname)
        if handler_name == "vector_search":
            # Inject index, vector_path, projection so the query provider never has to
            # fall back to tool_config['tools']['vector_search'] by raw handler name.
            for field in ("index", "vector_path", "projection"):
                if field not in kwargs and tool_cfg.get(field) is not None:
                    kwargs[field] = tool_cfg[field]
                    logger.debug("inject_collection_args: injected %s=%r for tool=%r", field, tool_cfg[field], toolname)
        if handler_name == "geospatial_search":
            current_geo_field = kwargs.get("geo_field")
            if current_geo_field is None or not str(current_geo_field).strip():
                location_field = tool_cfg.get("location_field")
                if location_field:
                    kwargs["geo_field"] = location_field
                    logger.debug("inject_collection_args: injected geo_field=%r for tool=%r", location_field, toolname)
        if handler_name == "hybrid_search":
            for field in ("vector_index", "text_index", "vector_path", "text_fields", "projection"):
                if field not in kwargs and tool_cfg.get(field) is not None:
                    kwargs[field] = tool_cfg[field]
                    logger.debug("inject_collection_args: injected %s=%r for tool=%r", field, tool_cfg[field], toolname)
        return kwargs

    def save_llm_conversation(
        self,
        conversation_data: Dict[str, Any],
        agent_id: str,
        tool_name: str,
        prompt_name: str,
        doc_id: Optional[str] = None,
    ) -> Optional[str]:
        """Insert or update an LLM conversation record in llm_history.

        If *doc_id* is None a new document is inserted and its string _id is
        returned.  If *doc_id* is provided the existing document is updated
        with ``$set`` and *doc_id* is returned unchanged.  Returns None on
        any failure.
        """
        try:
            if not self.mongo_client.sync_connect_to_mongodb():
                logger.error("MongoDB connection not established. Cannot save LLM conversation.")
                return None
            collection = self.mongo_client.get_collection("llm_history")
            if doc_id is not None:
                update_data = dict(conversation_data)
                update_data["updated_at"] = datetime.datetime.now().isoformat()
                collection.update_one(
                    {"_id": ObjectId(doc_id)},
                    {"$set": update_data},
                )
                logger.debug(f"LLM conversation updated for id: {doc_id}")
                return doc_id
            else:
                data = {
                    "agent_id": agent_id,
                    "tool_name": tool_name,
                    "prompt_name": prompt_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                data.update(conversation_data)
                result = collection.insert_one(data)
                inserted = str(result.inserted_id)
                logger.debug(f"LLM conversation saved with id: {inserted}")
                return inserted
        except Exception as e:
            logger.error(f"Failed to save LLM conversation: {e}")
            return None

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ):
        """Intercept tool calls to inject config-driven parameters from the tool annotations.

        For any tool that has a 'collection' field in its config entry, that value is
        injected into the call arguments so the LLM never needs to supply it.
        For geospatial_search, 'location_field' is also injected as 'geo_field' when present.
        Falls back to module_info.collection (the server default) when no per-tool override exists.
        """
        tool_name = context.message.name if hasattr(context.message, "name") else None
        if tool_name:
            logger.debug(
                "on_call_tool: received tool call name=%r args=%r endpoint=%r known_tools=%s",
                tool_name,
                getattr(context.message, "arguments", None),
                self.endpoint_name,
                sorted(self.endpoint_tools.keys()),
            )
            # Security gate: block calls to any tool not present in the config AND not a memory
            # tool. This prevents direct invocation of raw handler names (e.g. 'vector_search')
            # that are registered internally but intentionally not exposed in the config.
            if tool_name not in _MEMORY_TOOLS and tool_name not in self.endpoint_tools:
                logger.warning(f"on_call_tool: blocking call to unconfigured tool '{tool_name}'")
                raise PermissionError(f"Tool '{tool_name}' is not available.")

            args = dict(context.message.arguments or {})
            logger.debug("on_call_tool: pre-injection args for %r => %r", tool_name, args)
            args = self.inject_collection_args(tool_name, args)
            context.message.arguments = args
            logger.debug("on_call_tool: post-injection args for %r => %r", tool_name, args)
        return await call_next(context)

    async def on_list_prompts(self, context, call_next):
        return await super().on_list_prompts(context, call_next)

    async def on_list_tools(
        self,
        context: MiddlewareContext[mt.ListToolsRequest],
        call_next: CallNext[mt.ListToolsRequest, List[mt.Tool]]
    ) -> List[mt.Tool]:
        """Intercept list_tools and apply MongoDB annotation config: descriptions, parameter info, and tool filtering."""
        try:
            result = await call_next(context)

            if not result:
                logger.info("on_list_tools: no tools returned from handler")
                return result

            self.load_annotations()
            remove_tools = []
            for tool in result:
                # Memory layer tools: strip server-injected params the LLM must not supply.
                if tool.name in _MEMORY_TOOLS:
                    if tool.parameters and "properties" in tool.parameters:
                        tool.parameters["properties"].pop("agent_id", None)
                    continue
                anot = self.endpoint_tools.get(tool.name)
                if not anot:
                    logger.debug(f"No annotation found for tool '{tool.name}', removing from list")
                    remove_tools.append(tool)
                    continue

                description = anot.get("description", f"Tool: {tool.name}")
                returns = anot.get("returns")
                if returns:
                    description += f"\n\nReturns:\n    {returns}"
                tool.description = description

                if tool.parameters and "properties" in tool.parameters:
                    new_props = {}
                    for prop_name, prop_val in tool.parameters["properties"].items():
                        if prop_name in ("token", "agent_id"):
                            continue
                        # Strip params that are injected by on_call_tool from the tool config —
                        # the LLM must not see or supply these.
                        tool_cfg = self.endpoint_tools.get(tool.name, {})
                        if prop_name == "collection" and tool_cfg.get("collection"):
                            continue
                        if prop_name == "geo_field" and tool_cfg.get("location_field"):
                            continue
                        new_props[prop_name] = prop_val
                        param_info = anot.get("parameters", {}).get(prop_name, {})
                        if param_info.get("description"):
                            new_props[prop_name]["description"] = param_info["description"]
                    tool.parameters["properties"] = new_props

            for rt in remove_tools:
                result.remove(rt)

            return result

        except Exception as e:
            logger.error(f"ERROR in on_list_tools: {e}")
            traceback.print_exc()
            raise
