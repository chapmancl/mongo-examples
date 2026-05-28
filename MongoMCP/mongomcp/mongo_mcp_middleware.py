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
import time
import traceback
from .mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)

_MEMORY_TOOLS = {
    "intake", "recall", "reflect", "query", "list_sessions",
    "schema_declare", "strategy_store", "strategy_recall", "get_instructions",
    # Agent tools — registered on agent_mcp, pass through middleware security gate.
    "run_prompt",
    # Function-builder tools — registered on agent_mcp; not config-driven data-domain
    # tools, so they must be explicitly whitelisted to pass the on_call_tool gate.
    "define_query_function", "promote_query_function", "delete_query_function",
    # External-API connector tools — registered on agent_mcp; fixed-endpoint HTTP
    # proxies (e.g. MongoDB docs search, blog RSS, GitHub reader), not config-driven,
    # so whitelist them too.
    "search_mongodb_docs", "list_mongodb_doc_sources", "read_mongodb_blog", "read_github",
}

# Raw Python handler names that are never registered directly as @mcp.tool().
# register_collection_tools() wraps these under config-driven names instead.
_COLLECTION_PINNED_HANDLERS = frozenset({"vector_search", "rerank_search", "vector_rerank_search", "rerank_ids", "rerank_documents", "text_search", "geospatial_search", "hybrid_search", "custom_pipeline", "multi_step"})

# Config fields for custom_pipeline that are authoritative (config wins over any
# client-supplied value) and must be stripped from the LLM/MCP-visible schema.
_CUSTOM_PIPELINE_INJECTED = ("operation", "pipeline", "filter", "update", "document", "upsert", "multi")


def _coerce_param_value(value, type_str):
    """Coerce a declared parameter value to its declared type.

    Tool parameters carry a declared 'type' ('integer' | 'number' | 'boolean' | ...). A
    value can arrive as a string — from a JSON tool call or a stored default — which then
    breaks numeric aggregation stages (e.g. $limit: "100" -> 'Expected a number'). Coercing
    to the declared type here fixes that at the single point where params are folded.
    Best-effort: returns the value unchanged on any failure or unknown type.
    """
    if value is None or not isinstance(type_str, str):
        return value
    t = type_str.strip().lower()
    try:
        if t in ("int", "integer") and not isinstance(value, bool):
            return int(value)
        if t in ("float", "number", "double", "decimal"):
            return float(value)
        if t in ("bool", "boolean"):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
    except (ValueError, TypeError):
        return value
    return value


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
        # Per-tool config cache: tool_name -> (expires_at_monotonic, cfg). Serves the
        # read-your-writes lookup from memory on the hot path so it neither hits Mongo
        # nor blocks the async event loop every call. New/redefined functions on another
        # container still propagate within TOOL_CONFIG_CACHE_TTL seconds.
        self._tool_cfg_cache: dict = {}
        self._tool_cfg_ttl = float(os.getenv("TOOL_CONFIG_CACHE_TTL", "30"))
        # Set by register_query_tools: registers a config-driven tool with FastMCP on
        # demand, so a function defined mid-session is dispatchable without a restart.
        self.ensure_tool_registered = None
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

    @staticmethod
    def _stage_visible(anot: dict, requested_stage: str) -> bool:
        """Return True if a tool config entry is visible for the requested stage.

        Tools carry an optional 'stage' field: 'disabled' | 'dev' | 'prod'. Entries
        without a 'stage' (all legacy/hand-authored tools) are treated as 'prod'.
          - requested 'prod' -> only 'prod' entries.
          - requested 'dev'  -> 'dev' AND 'prod' entries (test the whole set).
          - 'disabled' is never visible.
        """
        stage = (anot.get("stage") or "prod") if isinstance(anot, dict) else "prod"
        if stage == "disabled":
            return False
        if requested_stage == "dev":
            return stage in ("dev", "prod")
        return stage == "prod"

    def build_tools_from_annotations(self, stage: str = "prod") -> List[Dict]:
        """Build Bedrock toolSpec JSON entirely from MongoDB annotations.

        Requires no FastMCP introspection — all tool names, descriptions, parameter
        types, defaults, and required lists come directly from the MongoDB config
        collection. Call this instead of get_llm_tools()/get_formatted_llm_tools()
        wherever you previously needed to first call mcp.get_tools().

        ``stage`` selects which dynamic functions are visible ('prod' default, or
        'dev' to also surface in-development functions).

        Returns:
            List of dicts in Bedrock toolSpec format, ready for LLM consumption.
        """
        self.load_annotations()
        tools_dict = []
        try:
            for tool_name, anot in self.endpoint_tools.items():
                if not self._stage_visible(anot, stage):
                    continue
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

    def build_tools_from_all_endpoints(self, stage: str = "prod") -> List[Dict]:
        """Build Bedrock toolSpec for ALL active endpoints from MongoDB, with endpoint-name prefix.

        Each tool name is prefixed as '<endpoint_name>_<tool_name>' so the HTTP dispatch
        in _make_mcp_call_fn can split on the first '_' to route to the correct MCP mount.
        Skips the current endpoint's tools — those are already covered by build_tools_from_annotations().
        In local mode, returns only the current endpoint's tools (same as build_tools_from_annotations).

        ``stage`` selects which dynamic functions are visible ('prod' default, 'dev' to
        also surface in-development functions).

        Returns:
            List of dicts in Bedrock toolSpec format with prefixed tool names.
        """
        if self._is_local:
            # Local mode: only one endpoint available; prefix current tools normally.
            tools = []
            for t in self.build_tools_from_annotations(stage=stage):
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
                    if not self._stage_visible(anot, stage):
                        continue
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
        if handler_name == "multi_step":
            # Multi-step functions carry a per-step 'collection'; the top-level one is an
            # optional default. 'steps' and 'output' are authoritative from config (a
            # caller can never substitute its own step chain). Declared params fold into
            # a single 'params' dict just like custom_pipeline.
            kwargs["steps"] = tool_cfg.get("steps") or []
            if tool_cfg.get("output") is not None:
                kwargs["output"] = tool_cfg.get("output")
            if tool_cfg.get("collection") is not None and not kwargs.get("collection"):
                kwargs["collection"] = tool_cfg.get("collection")
            declared = tool_cfg.get("parameters", {}) or {}
            params = dict(kwargs.get("params") or {})
            for pname, pinfo in declared.items():
                if pname in kwargs and pname not in ("params",):
                    params[pname] = kwargs.pop(pname)
                elif pname not in params and isinstance(pinfo, dict) and pinfo.get("default") is not None:
                    params[pname] = pinfo.get("default")
                # Coerce to the declared type so e.g. an 'integer' param reaches $limit as
                # a real number rather than a string.
                if pname in params and isinstance(pinfo, dict):
                    params[pname] = _coerce_param_value(params[pname], pinfo.get("type"))
            kwargs["params"] = params
            logger.debug("inject_collection_args: multi_step steps=%d params=%r for tool=%r", len(kwargs["steps"]), params, toolname)
            return kwargs
        current_collection = kwargs.get("collection")
        if current_collection is None or not str(current_collection).strip():
            collection = tool_cfg.get("collection")
            if collection and str(collection).strip():
                kwargs["collection"] = collection
                logger.debug("inject_collection_args: injected collection=%r for tool=%r", collection, toolname)
            elif handler_name != "rerank_documents":
                # rerank_documents reranks a caller-supplied 'documents' array — it never
                # touches a collection, so collection is optional for it alone.
                raise ValueError(
                    f"Tool '{toolname}' (handler: '{handler_name}') has no valid "
                    f"'collection' field in its config. Add 'collection' to the "
                    f"tool's config entry in MongoDB."
                )
        if handler_name == "vector_search":
            # Inject index, vector_path, projection so the query provider never has to
            # fall back to tool_config['tools']['vector_search'] by raw handler name.
            for field in ("index", "vector_path", "projection"):
                if field not in kwargs and tool_cfg.get(field) is not None:
                    kwargs[field] = tool_cfg[field]
                    logger.debug("inject_collection_args: injected %s=%r for tool=%r", field, tool_cfg[field], toolname)
        if handler_name in ("rerank_search", "vector_rerank_search"):
            # Self-contained vector-retrieve -> rerank. Piggybacks vector_search retrieval
            # config, plus rerank_field naming the document text the reranker scores.
            # rerank_field can't be inferred, so it must be present in the tool config.
            for field in ("index", "vector_path", "projection", "rerank_field"):
                if field not in kwargs and tool_cfg.get(field) is not None:
                    kwargs[field] = tool_cfg[field]
                    logger.debug("inject_collection_args: injected %s=%r for tool=%r", field, tool_cfg[field], toolname)
            if not kwargs.get("rerank_field") or not str(kwargs.get("rerank_field")).strip():
                raise ValueError(
                    f"Tool '{toolname}' (handler: '{handler_name}') has no valid "
                    f"'rerank_field' in its config. Add 'rerank_field' (the document "
                    f"text field to rerank on) to the tool's config entry in MongoDB."
                )
        if handler_name in ("rerank_ids", "rerank_documents"):
            # Decoupled rerank (Design B): no retrieval/embedding — reranks a candidate set
            # (fetched by _id for rerank_ids, or supplied directly for rerank_documents).
            # Only rerank_field is config-driven and required.
            if "rerank_field" not in kwargs and tool_cfg.get("rerank_field") is not None:
                kwargs["rerank_field"] = tool_cfg["rerank_field"]
                logger.debug("inject_collection_args: injected rerank_field=%r for tool=%r", tool_cfg.get("rerank_field"), toolname)
            if not kwargs.get("rerank_field") or not str(kwargs.get("rerank_field")).strip():
                raise ValueError(
                    f"Tool '{toolname}' (handler: '{handler_name}') has no valid "
                    f"'rerank_field' in its config. Add 'rerank_field' (the document "
                    f"text field to rerank on) to the tool's config entry in MongoDB."
                )
        if handler_name == "geospatial_search":
            current_geo_field = kwargs.get("geo_field")
            if current_geo_field is None or not str(current_geo_field).strip():
                location_field = tool_cfg.get("location_field")
                if location_field:
                    kwargs["geo_field"] = location_field
                    logger.debug("inject_collection_args: injected geo_field=%r for tool=%r", location_field, toolname)
            if "projection" not in kwargs and tool_cfg.get("projection") is not None:
                kwargs["projection"] = tool_cfg["projection"]
                logger.debug("inject_collection_args: injected projection=%r for tool=%r", tool_cfg["projection"], toolname)
        if handler_name == "text_search":
            if "projection" not in kwargs and tool_cfg.get("projection") is not None:
                kwargs["projection"] = tool_cfg["projection"]
                logger.debug("inject_collection_args: injected projection=%r for tool=%r", tool_cfg["projection"], toolname)
        if handler_name == "hybrid_search":
            for field in ("vector_index", "text_index", "vector_path", "text_fields", "projection"):
                if field not in kwargs and tool_cfg.get(field) is not None:
                    kwargs[field] = tool_cfg[field]
                    logger.debug("inject_collection_args: injected %s=%r for tool=%r", field, tool_cfg[field], toolname)
        if handler_name == "custom_pipeline":
            # Config is authoritative for the operation and every template block —
            # force-override so a client can never substitute its own pipeline/filter/update.
            kwargs["operation"] = tool_cfg.get("operation", "aggregate")
            for field in _CUSTOM_PIPELINE_INJECTED[1:]:  # skip 'operation' (set above)
                if tool_cfg.get(field) is not None:
                    kwargs[field] = tool_cfg[field]
            # Fold declared placeholder params (and their defaults) into a single
            # 'params' dict. On the invoke_llm path these arrive as top-level kwargs;
            # on the MCP path they arrive pre-nested as 'params'.
            declared = tool_cfg.get("parameters", {}) or {}
            params = dict(kwargs.get("params") or {})
            for pname, pinfo in declared.items():
                if pname in kwargs and pname not in ("params", "limit"):
                    params[pname] = kwargs.pop(pname)
                elif pname not in params and isinstance(pinfo, dict) and pinfo.get("default") is not None:
                    params[pname] = pinfo.get("default")
                # Coerce to the declared type so numeric params reach the pipeline as numbers.
                if pname in params and isinstance(pinfo, dict):
                    params[pname] = _coerce_param_value(params[pname], pinfo.get("type"))
            kwargs["params"] = params
            logger.debug("inject_collection_args: custom_pipeline op=%r params=%r for tool=%r", kwargs["operation"], params, toolname)
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

    def _refresh_tool_config(self, tool_name: str) -> Optional[dict]:
        """Return ONE tool's config, served from a short-TTL in-memory cache and refreshed
        from Mongo (projected) on a miss. Read-your-writes is preserved within the TTL:
        a function defined/redefined on ANY container is picked up after at most
        TOOL_CONFIG_CACHE_TTL seconds (default 30s). The cache also keeps the projected
        Mongo read (a blocking sync pymongo call) off the hot async path on every call.
        Best-effort: on any failure the existing cached entry is kept.
        """
        # Fast path: serve from the TTL cache while fresh.
        now = time.monotonic()
        cached = self._tool_cfg_cache.get(tool_name)
        if cached is not None and cached[0] > now:
            return cached[1]
        try:
            if not self.mongo_client.sync_connect_to_mongodb():
                return self.endpoint_tools.get(tool_name)
            doc = self.mongo_client.get_collection().find_one(
                {"Name": self.endpoint_name},
                {f"tools.{tool_name}": 1},
            )
            cfg = ((doc or {}).get("tools") or {}).get(tool_name) if doc else None
            if cfg is not None:
                self.endpoint_tools[tool_name] = cfg
            elif tool_name in self.endpoint_tools:
                # Removed from config since last load — drop it so the gate blocks it.
                self.endpoint_tools.pop(tool_name, None)
            self._tool_cfg_cache[tool_name] = (now + self._tool_cfg_ttl, cfg)
            return cfg
        except Exception as e:
            logger.warning("on_call_tool: fresh config read for %r failed (using cached): %s", tool_name, e)
            return self.endpoint_tools.get(tool_name)

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
        _t0 = time.monotonic()
        _refresh_ms = 0
        if tool_name:
            # Read-your-writes: refresh THIS tool's config fresh from Mongo before the gate
            # and injection, so a function defined/redefined on any other container is
            # honored immediately — no per-replica cache staleness (see _refresh_tool_config).
            # Memory/agent tools are not config-driven, so skip the read for them.
            if tool_name not in _MEMORY_TOOLS:
                _tr = time.monotonic()
                cfg = self._refresh_tool_config(tool_name)
                _refresh_ms = int((time.monotonic() - _tr) * 1000)
                # Register with FastMCP on demand if this container has never seen the tool
                # (defined on another container after startup) so dispatch doesn't 404 with
                # 'Unknown tool'. No-ops once the tool is registered.
                if cfg and callable(self.ensure_tool_registered):
                    self.ensure_tool_registered(tool_name, cfg)
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
        _t_call = time.monotonic()
        try:
            return await call_next(context)
        finally:
            logger.info(
                "on_call_tool timing: tool=%r endpoint=%r server_total=%d ms "
                "(config_refresh=%d ms, pre_call=%d ms, handler=%d ms)",
                tool_name, self.endpoint_name,
                int((time.monotonic() - _t0) * 1000),
                _refresh_ms,
                int((_t_call - _t0) * 1000),
                int((time.monotonic() - _t_call) * 1000),
            )

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
                    tool_cfg = self.endpoint_tools.get(tool.name, {})
                    is_custom_pipeline = tool_cfg.get("handler") == "custom_pipeline"
                    is_multi_step = tool_cfg.get("handler") == "multi_step"
                    for prop_name, prop_val in tool.parameters["properties"].items():
                        if prop_name in ("token", "agent_id"):
                            continue
                        # Strip params that are injected by on_call_tool from the tool config —
                        # the LLM must not see or supply these.
                        if prop_name == "collection" and tool_cfg.get("collection"):
                            continue
                        if prop_name == "geo_field" and tool_cfg.get("location_field"):
                            continue
                        # custom_pipeline: hide the injected template blocks; only 'params'
                        # and 'limit' plus the declared placeholder params are caller-facing.
                        if is_custom_pipeline and prop_name in _CUSTOM_PIPELINE_INJECTED:
                            continue
                        # multi_step: hide the authoritative step chain; only the declared
                        # placeholder params (folded into 'params') are caller-facing.
                        if is_multi_step and prop_name in ("steps", "output"):
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
