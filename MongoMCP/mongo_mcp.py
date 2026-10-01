import asyncio
import copy
import datetime
import inspect
import json
from bson import ObjectId
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Annotated
import logging
import importlib
import fastmcp
from fastmcp import FastMCP
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastmcp.server.dependencies import AccessToken
from starlette.responses import JSONResponse
from mongomcp import MongoDBQueryServer, MongoMCPMiddleware, ServerBedrockClient, MongoTokenVerifier, register_memory_tools, register_query_tools, get_memory_bedrock_toolspecs, register_agent_tools, get_agent_bedrock_toolspecs, register_function_builder_tools, register_external_api_tools, __version__ as MCP_VERSION
from mongomcp.mongodb_client import query_capture_cv as _mongo_capture_cv, _query_capture_registry as _mongo_capture_registry, set_query_capture_enabled as _set_query_capture_enabled, _CAPTURE_LISTENER as _mongo_capture_listener
from mongomcp.agent.prompt_agent import PromptAgent
from mongomcp.agent.tool_router import ToolRouter
from starlette.types import ASGIApp, Receive, Scope, Send
import mcp.types as mt
import traceback
import os
import sys
import time
import uuid

_MODULE_NAME = os.getenv("SETTINGS_MODULE", "local_settings")
settings = importlib.import_module(_MODULE_NAME).settings

from mongomcp import MongoDBQueryServer, MongoMCPMiddleware, ServerBedrockClient, MongoTokenVerifier, register_memory_tools, register_query_tools, get_memory_bedrock_toolspecs, register_agent_tools, get_agent_bedrock_toolspecs, register_function_builder_tools, register_external_api_tools, __version__ as MCP_VERSION
from mongomcp.mongodb_client import query_capture_cv as _mongo_capture_cv, _query_capture_registry as _mongo_capture_registry, set_query_capture_enabled as _set_query_capture_enabled, _CAPTURE_LISTENER as _mongo_capture_listener
from mongomcp.agent.prompt_agent import PromptAgent
from mongomcp.agent.tool_router import ToolRouter

logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("fastmcp.server.mixins.mcp_operations").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.streamable_http").setLevel(logging.WARNING)
logging.getLogger("mcp.server.streamable_http_manager").setLevel(logging.CRITICAL)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
logging.getLogger("mongomcp.mongo_mcp_middleware").setLevel(logging.DEBUG)
logging.getLogger("mongomcp.mongodb_client").setLevel(logging.WARNING)
logging.getLogger("mongomcp.memory").setLevel(logging.DEBUG)
logging.getLogger("mongomcp.memory.tools").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
main component flow:
1. MongoMCPMiddleware: Connects to MongoDB config database, loads tool configurations, and prep token authorization.
2. MongoDBQueryServer: Implements core MongoDB query functionalities for the specific tool_name from env
3. BedrockClient: Handles AWS Bedrock LLM interactions and tool integrations.
4. instantiate FastMCP servera with MongoMCPMiddleware, MongoDBQueryServer, MongoTokenVerifier
5. instantiate FastAPI app, mounts FastMCP app
6. define additional endpoints for health checks, tool configuration retrieval, settings reset, LLM invocation, and text vectorization.

"""

TOOL_NAME = settings.TOOL_NAME
mongo_middleware: MongoMCPMiddleware
mongo_server: MongoDBQueryServer
auth_provider = None

def setup_from_mongo():
    """
     setup the list tools middleware to load the tool configuration from mongo
     this will also verify we can connect to mongo before starting the server
     the middleware will be added to the MCP server instance below to intercept tool calls
    """
    global mongo_middleware
    global mongo_server
    global auth_provider
    mongo_middleware = None
    mongo_server = None
    auth_provider = None
    max_attempts = 5
    error = None

    # load or reload the mongo middleware and server config
    # we do this to get fresh settings from mongo if reset_settings is called
    for attempt in range(1, max_attempts + 1):
        failed = False
        error = None
        try:
            mongo_middleware = MongoMCPMiddleware(settings)
            if mongo_middleware.ANNOTATIONS:
                mongo_server = MongoDBQueryServer(settings)
                mongo_server.set_config(mongo_middleware.ANNOTATIONS)
                auth_provider = MongoTokenVerifier(mongo_middleware)
                return
            failed = True
            error = "Configuration annotations were empty"
        except ConnectionError as e:
            failed = True
            error = e

        if failed and attempt < max_attempts:
            wait_seconds = attempt * 5
            logger.error(
                f"Attempt {attempt}/{max_attempts}: failed to get configuration from MongoDB. "
                f"Retrying in {wait_seconds}s. Error: {error}"
            )
            time.sleep(wait_seconds)

    logger.error(
        f"Failed to get configuration from MongoDB after {max_attempts} attempts. Last error: {error}"
    )
    sys.exit(1)

setup_from_mongo()
_set_query_capture_enabled(os.environ.get("QUERY_LOGGING", "").lower() in ("1", "true", "yes"))

# MCP_AUTH_ENABLED=true (default): validate tokens against agent_identities.
# Set false only when an upstream gateway has already validated the bearer token.
_mcp_auth_enabled = os.environ.get("MCP_AUTH_ENABLED", "true").lower() in ("1", "true", "yes")
if auth_provider is not None:
    auth_provider.strict = _mcp_auth_enabled
if not _mcp_auth_enabled:
    logger.info("FastMCP auth in non-strict mode (MCP_AUTH_ENABLED=false) — identity parsed from token, validation skipped")

# Create FastMCP server instance with bearer token authentication
# this is the mongo tools from config load.
llm_client = ServerBedrockClient(settings)
mcp = FastMCP("mongodb-vector-server", auth=auth_provider)
mcp.add_middleware(mongo_middleware)
_query_dispatch = register_query_tools(mcp, mongo_server, llm_client, mongo_middleware.endpoint_tools, middleware=mongo_middleware)


# Separate FastMCP instance for the memory layer — keeps memory tools off the main tool catalog.
_agent_instructions = getattr(settings, "agent_instructions", None)
memory_mcp = FastMCP("memory-server", auth=auth_provider, instructions=_agent_instructions or None)
memory_mcp.add_middleware(mongo_middleware)
_memory_dispatch = register_memory_tools(memory_mcp, mongo_server, llm_client, settings)

agent_mcp = FastMCP("agent-server", auth=auth_provider)
agent_mcp.add_middleware(mongo_middleware)

def _get_agent_tool_catalog():
    memory_tools = []
    for tool in get_memory_bedrock_toolspecs():
        tool = copy.deepcopy(tool)
        tool["toolSpec"]["name"] = f"memory_{tool['toolSpec']['name']}"
        memory_tools.append(tool)

    endpoint_tools = []
    try:
        endpoint_tools = [copy.deepcopy(tool) for tool in mongo_middleware.build_tools_from_all_endpoints()]
    except Exception as exc:
        logger.error("_get_agent_tool_catalog: failed to load endpoint tools: %s", exc)

    agent_tools = []
    for tool in get_agent_bedrock_toolspecs():
        name = tool.get("toolSpec", {}).get("name", "")
        if name == "run_prompt":
            continue
        tool = copy.deepcopy(tool)
        tool["toolSpec"]["name"] = f"agent_{name}"
        agent_tools.append(tool)
    return endpoint_tools + memory_tools + agent_tools

async def _local_tool_call(token, toolname, tool_input):
    if not toolname or "_" not in toolname:
        return (False, None)
    endpoint_name, bare_name = toolname.split("_", 1)
    if endpoint_name not in (settings.TOOL_NAME, "memory", "agent"):
        return (False, None)
    if bare_name not in _TOOL_DISPATCH:
        return (False, None)
    return (True, await tool_handler(token, bare_name, tool_input))


register_agent_tools(
    agent_mcp, settings, _get_agent_tool_catalog,
    mongo_middleware.save_llm_conversation, local_call_fn=_local_tool_call,
)
_function_builder_dispatch = register_function_builder_tools(agent_mcp, settings, mongo_middleware, llm_client)
_external_api_dispatch = register_external_api_tools(agent_mcp, settings)

_CAPTURE_HEADER = "x-capture-doc-id"
_CAPTURE_TOOL_HEADER = "x-capture-tool"


async def _push_query_log(doc_id: str, tool_name: str, captured: list) -> None:
    entries = []
    for query in captured:
        entry = {
            "tool": tool_name,
            "command": query.get("command"),
            "database": query.get("database"),
            "collection": query.get("collection"),
            "ts": datetime.datetime.now().isoformat(),
        }
        if "pipeline" in query:
            entry["pipeline"] = query["pipeline"]
        else:
            for field in ("filter", "projection", "sort", "limit"):
                if field in query:
                    entry[field] = query[field]
        entries.append(entry)
    try:
        await mongo_middleware.mongo_client.ensure_connection()
        collection = mongo_middleware.mongo_client.db["llm_history"]
        result = collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$push": {"queries_used": {"$each": entries}}},
        )
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            await result
    except Exception as exc:
        logger.warning("query_log push failed (doc_id=%s tool=%s): %s", doc_id, tool_name, exc)


class _QueryCaptureMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and _mongo_capture_listener.enabled:
            headers = dict(scope.get("headers", []))
            doc_id = headers.get(_CAPTURE_HEADER.encode(), b"").decode()
            if doc_id:
                tool_name = headers.get(_CAPTURE_TOOL_HEADER.encode(), b"").decode() or "unknown"
                request_key = f"{doc_id}:{id(scope)}"
                _mongo_capture_registry[request_key] = []
                context_token = _mongo_capture_cv.set(request_key)
                try:
                    await self.app(scope, receive, send)
                finally:
                    _mongo_capture_cv.reset(context_token)
                    captured = _mongo_capture_registry.pop(request_key, [])
                    if captured:
                        asyncio.create_task(_push_query_log(doc_id, tool_name, captured))
                return
        await self.app(scope, receive, send)


def _make_mcp_call_fn(base_url: str, jwt: str, capture_doc_id: Optional[str] = None, local_token=None):
    async def mcp_call_fn(toolname: str, tool_input: dict) -> Any:
        if toolname in ("run_prompt", "agent_run_prompt"):
            return {"error": f"Agent tool recursion prevented: {toolname}"}
        if local_token is not None:
            handled, result = await _local_tool_call(local_token, toolname, tool_input)
            if handled:
                return result
        if "_" not in toolname:
            return {"error": f"Cannot resolve endpoint for unprefixed tool '{toolname}'"}
        endpoint_name, endpoint_tool_name = toolname.split("_", 1)
        headers = {"Authorization": f"Bearer {jwt}"}
        if capture_doc_id:
            headers[_CAPTURE_HEADER] = capture_doc_id
            headers[_CAPTURE_TOOL_HEADER] = toolname
        mcp_root = os.environ.get("MONGO_MCP_ROOT", "").rstrip("/")
        url = f"{mcp_root}/{endpoint_name}/mcp" if mcp_root else f"{base_url}{endpoint_name}/mcp"
        cfg = {"url": url, "transport": "http", "headers": headers}
        client = fastmcp.Client({"mcpServers": {endpoint_name: cfg}}, timeout=60)
        last_exc = None
        for attempt in range(2):
            try:
                async with client:
                    raw = await client.session.send_request(
                        mt.ClientRequest(mt.CallToolRequest(params=mt.CallToolRequestParams(
                            name=endpoint_tool_name, arguments=tool_input,
                        ))),
                        mt.CallToolResult,
                    )
                if raw.content and hasattr(raw.content[0], "text"):
                    return raw.content[0].text
                if raw.structuredContent is not None:
                    return json.dumps(raw.structuredContent)
                return str(raw)
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and any(code in str(exc) for code in ("502", "503", "504")):
                    logger.warning("HTTP dispatch transient error for %s: %s", toolname, exc)
                    await asyncio.sleep(1)
                    continue
                break
        logger.error("HTTP dispatch failed for %s: %s", toolname, last_exc)
        return {"error": f"Tool call failed: {last_exc}"}
    return mcp_call_fn


# ----------------------- BEGIN FASTAPI SECTION ----------------------- #

# We have our tools, mount the mcp to fastapi and setup our fastapi authentication
# everything after this should be FastAPI endpoints.
# both of these get registered here, but memory will get its own route below
mcp_app = mcp.http_app(path=f"/mcp", stateless_http=True, host_origin_protection=False)
memory_app = memory_mcp.http_app(path="/mcp", stateless_http=True, host_origin_protection=False)
agent_app = agent_mcp.http_app(path="/mcp", stateless_http=True, host_origin_protection=False)


@asynccontextmanager
async def _combined_lifespan(app):
    async with mcp_app.lifespan(app):
        async with memory_app.lifespan(app):
            async with agent_app.lifespan(app):
                try:
                    if mongo_server is not None:
                        await mongo_server.ensure_connection()
                except Exception as exc:
                    logger.warning("Data-domain connection pre-warm failed: %s", exc)
                yield


app = FastAPI(title=settings.TOOL_NAME, lifespan=_combined_lifespan)
app.add_middleware(_QueryCaptureMiddleware)
security_token = HTTPBearer()
optional_token = HTTPBearer(auto_error=False)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    raw_headers = dict(request.scope.get("headers", []))
    existing_id = raw_headers.get(b"x-request-id", b"").decode()
    request.state.request_id = existing_id or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response

def get_request_id(request: Request) -> str:
    """Read request ID from X-Request-ID header; generate a UUID if absent."""
    request_id = request.headers.get("x-request-id") or request.headers.get("mcp-session-id")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id

def verify_token(credentials: HTTPAuthorizationCredentials) -> Any:
    #print("mongomcp: verify_token")
    (allowed, agent_rec) = mongo_middleware.check_authorization(credentials.credentials)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    agent_rec["token"] = credentials.credentials
    return agent_rec

async def get_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security_token)]
):
    return verify_token(credentials)

def verify_optional_token(credentials: Optional[HTTPAuthorizationCredentials]) -> Any:
    if not credentials:
        return None
    return verify_token(credentials)

async def get_optional_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(optional_token)]
):
    return verify_optional_token(credentials)

# Dispatch table: tool_name → callable for the HTTP-path tool_handler.
# All query tools come from _query_dispatch; memory tools from _memory_dispatch.
_TOOL_DISPATCH = {
    **_query_dispatch,
    **_memory_dispatch,
    **_function_builder_dispatch,
    **_external_api_dispatch,
}

# Frozen set of memory tool names — handled with token passthrough so wrappers
# can derive agent_id internally.
_MEMORY_TOOL_NAMES = frozenset(_memory_dispatch.keys())

def _fn_accepts_kw(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())

async def tool_handler(token: AccessToken, toolname: str, tool_input: dict) -> dict:
    """Map toolname to the appropriate MCP tool function and execute it.
        This is only for an API call which invoked an LLM and needs to call an mcp tool
        We don't need it to go out to the webAPI, just call it within the local process
    """
    fn = _TOOL_DISPATCH.get(toolname)
    if fn is None:
        return {"error": f"Unknown tool: {toolname}"}
    try:
        kwargs = dict(tool_input)
        if toolname == "upsert_document":
            kwargs["token"] = token
        elif mongo_middleware.endpoint_tools.get(toolname, {}).get("handler") == "custom_pipeline":
            kwargs["token"] = token
        # Inject config-backed collection/geo_field via shared middleware method.
        kwargs = mongo_middleware.inject_collection_args(toolname, kwargs)
        # Memory wrappers derive agent_id from token internally.
        if toolname in _MEMORY_TOOL_NAMES:
            # Never pass agent_id directly; not all memory wrapper signatures expose it.
            kwargs.pop("agent_id", None)
            if _fn_accepts_kw(fn, "token"):
                kwargs.setdefault("token", token)
        if toolname == "get_instructions":
            logger.info("[PIPELINE] tool_handler: calling get_instructions, fn type=%s, fn=%r, qualname=%s",
                        type(fn).__name__, fn, getattr(fn, "__qualname__", "?"))
        result = await fn(**kwargs)
        if toolname == "get_instructions":
            logger.info("[PIPELINE] tool_handler: get_instructions result=%r", result)
        return result
    except Exception as e:
        logger.error(f"Tool handler error for {toolname}: {e}")
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
        return {"error": f"Error executing {toolname}: {str(e)}"}

# Root route
@app.get("/")
async def root_endpoint(token: Annotated[str | None, Depends(get_optional_token)]) -> Dict[str, Any]:
    """Root endpoint"""
    if token:
        active_tools = mongo_middleware.active_endpoints
        if "memory" not in active_tools:
            active_tools = [*active_tools, "memory"]
        if "agent" not in active_tools:
            active_tools = [*active_tools, "agent"]
        return {
            "message": "MongoDB Vector Server MCP",
            "status": "running",
            "version": MCP_VERSION,
            "available_tools": active_tools,
            "available_endpoints": [
                f"GET  /{settings.TOOL_NAME}/health",
                f"GET  /{settings.TOOL_NAME}/collection_info",
                f"GET  /{settings.TOOL_NAME}/llm_tools",
                f"POST /{settings.TOOL_NAME}/route",
                f"GET  /{settings.TOOL_NAME}/reset",
                f"POST /{settings.TOOL_NAME}/prompt/{{prompt_name}}",
                "GET  /memory/mcp  (memory layer — always available)",
                "GET  /tools_config",
                "POST /vectorize",
            ]
        }
    else:
        return {
            "message": "OK",
            "status": "running"
        }

# this is for the AWS load balancer health check
@app.get(f"/{settings.TOOL_NAME}/health")
@app.get("/health")
async def http_health_check(token: Annotated[str | None, Depends(get_optional_token)]) -> Dict[str, Any]:
    """Regular HTTP GET endpoint for health checks"""
    # always return something or else the load balancer will mark it unhealthy and continue to reload the container
    failed, server_info = await mongo_server.get_mongo_info(False)
    output = server_info.copy()
    output["version"] = MCP_VERSION
    output["llm_provider"] = getattr(settings, "LLM_PROVIDER", "bedrock")
    output["llm_model_id"] = settings.LLM_MODEL_ID
    if not token:
        # no token, remove sensitive info
        output.pop("mongodb")
        output.pop("description")
        output["connected"] = server_info["mongodb"].get("connected", False)
        output["timestamp"] = server_info["mongodb"].get("timestamp", "")

    status_code = 200
    #if failed:
    #    status_code = 500
    return output

@app.get("/tools_config")
async def http_get_tools_config(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Regular HTTP GET endpoint for tools config"""
    active = mongo_middleware.refresh_active_endpoints()
    if "memory" not in active:
        active = [*active, "memory"]
    if "agent" not in active:
        active = [*active, "agent"]
    return {"available_tools": active, "tool_name": settings.TOOL_NAME}

@app.get(f"/{settings.TOOL_NAME}/collection_info")
async def http_get_collection_info(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Regular HTTP GET endpoint for collection info"""        
    results = await _TOOL_DISPATCH["get_collection_info"]()
    return {"collection_info": results}


@app.get(f"/{settings.TOOL_NAME}/llm_tools")
async def http_get_llm_tools(token: Annotated[str, Depends(get_token)], stage: str = "prod") -> Dict[str, Any]:
    """Returns preformatted Bedrock toolSpec JSON for the active tool endpoint (MongoDB annotations)."""
    stage = stage if stage in ("prod", "dev") else "prod"
    tools = mongo_middleware.build_tools_from_annotations(stage=stage)
    module_info = (mongo_middleware.ANNOTATIONS or {}).get("module_info", {})
    return {
        "tools": tools,
        "count": len(tools),
        "description": module_info.get("description", ""),
        "title": module_info.get("title", ""),
    }


@app.get("/memory/llm_tools")
async def http_get_memory_llm_tools(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Returns preformatted Bedrock toolSpec JSON for all memory layer tools."""
    tools = get_memory_bedrock_toolspecs()
    return {"tools": tools, "count": len(tools)}


@app.get("/memory/instructions")
@app.get("/memory/get_instructions")
async def http_get_memory_instructions(token: Annotated[str | None, Depends(get_optional_token)]) -> Dict[str, Any]:
    result = await _TOOL_DISPATCH["get_instructions"]()
    if isinstance(result, dict) and "error" in result:
        return JSONResponse(result, 500)
    return result


@app.get("/agent/llm_tools")
async def http_get_agent_llm_tools(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Returns preformatted Bedrock toolSpec JSON for the agent layer (run_prompt)."""
    tools = get_agent_bedrock_toolspecs()
    return {"tools": tools, "count": len(tools)}


@app.get("/memory/collection_info")
async def http_get_memory_collection_info(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Returns module-level description for the memory layer (no collection stats)."""
    module_info = (mongo_middleware.ANNOTATIONS or {}).get("module_info", {})
    return {
        "tool_name": "memory",
        "title": "Mongo Memory Layer",
        "description": "Self-curating persistent memory system with semantic search, graph linking, and shard scan. Always available on every container.",
        "database": getattr(settings, "memory_db", "mcp_config"),
        "collections": ["memory_episodic", "memory_semantic"],
        "tools": [t["toolSpec"]["name"] for t in get_memory_bedrock_toolspecs()],
        "version": MCP_VERSION,
    }


@app.post(f"/{settings.TOOL_NAME}/route")
async def route_tools(body: Dict[str, Any], token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Select a subset of tools relevant to a question or explicit tool list.

    Body options (mutually exclusive):
      {"question": "..."}         — LLM routing: asks the model which tools are needed
      {"tools": ["ep.tool", ...]} — Static routing: deterministic filter by name

    The routing prompt is read from mongo config at prompts.tool_router if it exists.
    """
    all_tools = mongo_middleware.build_tools_from_annotations()
    question = body.get("question")
    explicit_tools = body.get("tools")

    if explicit_tools and isinstance(explicit_tools, list):
        # Static routing — no LLM call
        router = ToolRouter(tool_catalog=all_tools)
        filtered = router.select_tools(explicit_tools)
        return {"tools": filtered, "count": len(filtered), "routing": "static"}

    if question:
        # LLM routing
        routing_prompt = (
            mongo_server.tool_config.get("prompts", {}).get("tool_router")
            if hasattr(mongo_server, "tool_config") else None
        )
        router = ToolRouter(tool_catalog=all_tools, llm_client=llm_client)
        filtered = await router.route_for_question(question, routing_prompt)
        return {"tools": filtered, "count": len(filtered), "routing": "llm"}

    return JSONResponse({"error": "Request body must contain 'question' (string) or 'tools' (list)"}, 400)


@app.get(f"/{settings.TOOL_NAME}/reset")
async def reset_settings(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Reload config from MongoDB and reconfigure the LLM client with the latest tool annotations."""
    logger.info(f"Begin settings reset for {settings.TOOL_NAME}")
    output = {"action": "reset settings"}
    try:
        setup_from_mongo()
        new_client = ServerBedrockClient(settings)
        new_client.configure_tools(mongo_middleware.build_tools_from_annotations())
        global llm_client
        llm_client = new_client
        output["result"] = "success"
        logger.info(f"Finished settings reset for {settings.TOOL_NAME}: Success")
    except Exception as e:
        logger.error(f"reset_settings failed: {e}")
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
        output["error"] = f"Error executing reset_settings: {str(e)}"
        output["result"] = "failed"
        logger.info(f"Finished settings reset for {settings.TOOL_NAME}: Failed")
        return JSONResponse(output, 500)

    return output

@app.post(f"/{settings.TOOL_NAME}/prompt_sync/{{prompt_name}}")
async def invoke_llm_old(prompt_name: str, body: Dict[str, Any], 
                     token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:    
    """
    Invoke LLM with specified prompt and incoming context.
    The prompt is looked from and must exist in the MongoDB tool configuration prompts section.
    
    """     
    if not "llm:invoke" in token.get("scope", []):
        logger.error(f"Insufficient scope for invoke_llm: llm:invoke permission required for agent {token["agent_key"]}")
        raise HTTPException(status_code=403, detail="Insufficient scope")

    context = body.get("context")
    output = {
        "prompt_name": prompt_name,
        "input_context": context
    }

    def _save_output_snapshot() -> None:
        """Best-effort save of invoke output for observability, including failures/warnings."""
        try:
            mongo_middleware.save_llm_conversation(
                output,
                token.get("agent_key", "unknown"),
                settings.TOOL_NAME,
                prompt_name,
            )
        except Exception as save_err:
            logger.warning("Failed to save invoke_llm conversation snapshot: %s", save_err)

    def _mark_context_limit_warning(err_msg: str) -> None:
        warning = (
            "Prompt/context exceeded the model token limit. "
            "Summarize context and save key facts to memory before retrying."
        )
        output["status"] = "Context Limit Warning"
        output["message"] = warning
        output["warning"] = warning
        output["error"] = err_msg

    try:
        if not context:
            raise ValueError("context must be a non-empty json object in the request body")
        
        # don't load llm client with tools unless there are prompts available.         
        # if the prompt changes (on the mongo side), then reset_settings must be called to reload the tool annotations.
        # this will finish the setup next time an invoke is called.
        global llm_client
        tools_config = [
            *mongo_middleware.build_tools_from_annotations(),
            *get_memory_bedrock_toolspecs(),
        ]
        llm_client.configure_tools(tools_config)
                        
        # Lookup prompt from mongo_server.tool_config["prompts"] if it exists        
        if ("prompts" in mongo_server.tool_config and 
            prompt_name in mongo_server.tool_config["prompts"]):
            #We have a prompt!
            prompt = mongo_server.tool_config["prompts"][prompt_name]
            output["prompt"] = prompt

            async def scoped_mcp_call(toolname: str, tool_input: dict) -> dict:
                # Keep token handling in this top-level request scope.
                return await tool_handler(token, toolname, tool_input)

            # Bind request-scoped callback on the client instance; BedrockClient no longer
            # accepts a per-call tool callback parameter.
            llm_client.mcp_call = scoped_mcp_call

            resp_obj = await llm_client.invoke_bedrock_with_tools(
                prompt=prompt,
                context=json.dumps(context),
            )
            output.update(resp_obj)  # merge the response object into output
            
            # lots of potential errors and exceptions here, so catch them all. 
            # tried to pass most through the return, but some may still raise
            # I could not handle all the exceptions by name either, some would raise a runtime exception
            # instead of passing the exception directly
            return_json = {}
            if resp_obj.get("error"):
                return_json = JSONResponse(output, 500)                    
            else:
                logger.info(f"invoke successful for prompt {prompt_name}")
                return_json = JSONResponse(output, 201)
            
            # We want to save the full conversation including LLM output regardless of success or failure
            # Try to handle the exceptions and bubble them up to the output so we don't hit the catches below.
            _save_output_snapshot()
            return return_json

        else:
            output["error"] = f"Prompt '{prompt_name}' not found in configuration."
            _save_output_snapshot()
            return JSONResponse(output, 404)        
        
    except HTTPException as he:
        logger.error(f"Authorization failed: {he.detail}")
        output["error"] = he.detail
        _save_output_snapshot()
        return JSONResponse(output,he.status_code)
    except Exception as e:
        logger.error(f"invoke_llm failed: {e}")
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))        
        output["error"] = f"Error executing invoke_llm: {str(e)}"
        _save_output_snapshot()
        return JSONResponse(output, 500)


@app.post(f"/{settings.TOOL_NAME}/prompt/{{prompt_name}}")
async def invoke_llm(prompt_name: str, body: Dict[str, Any],
                     request: Request,
                     token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Spawn a PromptAgent sub-agent for the given prompt and return immediately.

    The agent runs in the background; save_llm_conversation is called when it
    finishes (or fails).  Returns HTTP 202 Accepted.
    """
    if "llm:invoke" not in token.get("scope", []):
        raise HTTPException(status_code=403, detail="Insufficient scope")

    session_id = get_request_id(request)
    context = body.get("context")
    if not context:
        return JSONResponse({"error": "context must be a non-empty json object in the request body"}, 400)

    if not ("prompts" in mongo_server.tool_config and prompt_name in mongo_server.tool_config["prompts"]):
        return JSONResponse({"error": f"Prompt '{prompt_name}' not found in configuration."}, 404)

    prompt = mongo_server.tool_config["prompts"][prompt_name]
    # Optional tool filter from the request body — same semantics as run_prompt:
    # the full catalog is built, filtered to this list, then memory tools are
    # always added back regardless.  Pass None to use the full catalog.
    tool_names: Optional[List[str]] = body.get("tool_names") or None
    jwt = token.get("token")
    base_url = str(request.base_url)
    agent_id = token.get("agent_key", "unknown")

    async def _run_agent() -> dict:
        output = {"prompt_name": prompt_name, "input_context": context, "prompt": prompt}

        # Pre-save: open the history record before the agent runs so we have an
        # _id to update when it finishes (or fails).
        doc_id: Optional[str] = None
        try:
            doc_id = mongo_middleware.save_llm_conversation(
                {"status": "running", "prompt": prompt, "input_context": context, "session_id": session_id},
                agent_id, settings.TOOL_NAME, prompt_name,
            )
        except Exception as pre_save_err:
            logger.warning("invoke_llm failed to save initial snapshot: %s", pre_save_err)

        # --- Query logging: capture actual MongoDB queries via CommandListener ---
        # Enabled by setting query_logging: true in the MongoDB tool config (reloaded on /reset).
        # The CommandListener in mongodb_client.py reads query_capture_cv which is set by
        # _QueryCaptureMiddleware on each incoming MCP sub-request that carries _CAPTURE_HEADER.
        _query_logging_enabled = _mongo_capture_listener.enabled

        # Each receiving pod writes its own captures directly to llm_history via
        # _push_query_log (fired from _QueryCaptureMiddleware after each response).
        # No in-process registry needed here — just forward the headers and let the
        # load balancer route freely.
        _agent_mcp_call_fn = _make_mcp_call_fn(
            base_url, jwt,
            capture_doc_id=doc_id if (_query_logging_enabled and doc_id) else None,
            local_token=token,
        )

        try:
            agent = PromptAgent(
                settings=settings,
                mcp_call_fn=_agent_mcp_call_fn,
                tool_catalog=_get_agent_tool_catalog(),
            )
            result = await agent.run(
                prompt=prompt,
                context=json.dumps(context) if isinstance(context, dict) else context,
                tool_names=tool_names,
                session_id=session_id,
                token=token,
            )
            output.update(result)
        except Exception as e:
            logger.error("invoke_llm agent failed for prompt '%s': %s", prompt_name, e)
            logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
            output["error"] = str(e)
        finally:
            try:
                output["status"] = "error" if output.get("error") else "complete"
                mongo_middleware.save_llm_conversation(
                    output, agent_id, settings.TOOL_NAME, prompt_name, doc_id=doc_id
                )
            except Exception as save_err:
                logger.warning("invoke_llm failed to save agent snapshot: %s", save_err)
        return output

    # Launch the agent as an asyncio Task so it can outlive this request if needed.
    task = asyncio.create_task(_run_agent())
    try:
        # asyncio.shield() prevents wait_for from cancelling the underlying task
        # when the timeout fires — the agent keeps running to completion in the
        # background and will still call save_llm_conversation when it finishes.
        # If the agent completes within 10s we return the full result synchronously.
        output = await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
        status_code = 500 if output.get("error") else 200
        return JSONResponse(output, status_code)
    except asyncio.TimeoutError:
        # Agent is still running via the shielded task — return 202 immediately
        # so the caller isn't blocked. The result will be saved to MongoDB by
        # save_llm_conversation once the agent finishes.
        return JSONResponse({
            "status": "spawned",
            "message": f"Sub-agent for prompt '{prompt_name}' has been spawned and is running in the background.",
            "prompt_name": f"{settings.TOOL_NAME}_{prompt_name}",
            "session_id": session_id,
        }, 202)


@app.post("/vectorize")
async def vectorize_text(body: Dict[str, Any],
                     token: Annotated[str, Depends(get_token)]
                     )  -> Dict[str, Any]:
    """
    API endpoint to vectorize input text using the LLM embedding model.
    this is not an MCP tool
    """
    try:
        if not "llm:invoke" in token.get("scope", []):
            raise HTTPException(status_code=403, detail="Insufficient scope")

        # Extract textChunk from the request body
        text_chunk = body.get("textChunk")

        if not text_chunk or not isinstance(text_chunk, str):
            raise Exception("textChunk must be a non-empty string in the request body")

        vector_info = await llm_client.generate_embedding(text_chunk)
        logger.info(f"Vectorization successful for input text of length {len(text_chunk)}")
        return {
            "input_text": text_chunk,
            "embedding_model": vector_info["embedding_model"],
            "vector": vector_info["vector"]
        }

    except HTTPException as he:
        logger.error(f"Authorization failed: {he.detail}")
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
        input = json.dumps(body)
        return {
            "error": f"Error executing vectorize_text: {str(e)}",
            "body" : input
        }


@app.post("/llm_history/save")
async def save_llm_history(
    body: Dict[str, Any],
    token: Annotated[str, Depends(get_token)]
) -> Dict[str, Any]:
    """
    Save LLM conversation history to MongoDB llm_history collection.

    Body parameters:
        username (str): Username of the user
        prompt (str): The prompt/question sent to the LLM
        response (str): The LLM's response
        tool_name (str, optional): Name of the tool/service (default: settings.TOOL_NAME)
        prompt_name (str, optional): Name of the specific prompt (default: "user_query")
        metadata (dict, optional): Additional metadata to store

    Returns:
        dict: {"status": "success", "id": "<document_id>"} or {"status": "error", "error": "<message>"}
    """
    try:
        # Extract required fields
        username = body.get("username")
        prompt = body.get("prompt")
        response = body.get("response")

        if not username:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "username is required"}
            )

        if not prompt:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "prompt is required"}
            )

        if not response:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "response is required"}
            )

        # Extract optional fields
        tool_name = body.get("tool_name", settings.TOOL_NAME)
        prompt_name = body.get("prompt_name", "user_query")
        metadata = body.get("metadata", {})

        # Build conversation data
        conversation_data = {
            "username": username,
            "prompt": prompt,
            "response": response,
            "metadata": metadata
        }

        # Use agent_id from token if available, otherwise use username
        agent_id = token.get("agent_name", username)

        # Save to MongoDB using middleware
        doc_id = mongo_middleware.save_llm_conversation(
            conversation_data=conversation_data,
            agent_id=agent_id,
            tool_name=tool_name,
            prompt_name=prompt_name
        )

        if doc_id:
            logger.info(f"LLM history saved for user: {username}, id: {doc_id}")
            return {
                "status": "success",
                "id": doc_id,
                "username": username,
                "tool_name": tool_name,
                "prompt_name": prompt_name
            }
        else:
            logger.error("Failed to save LLM history to MongoDB")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "error": "Failed to save to database"}
            )

    except HTTPException as he:
        logger.error(f"Authorization failed: {he.detail}")
        return JSONResponse(
            status_code=he.status_code,
            content={"status": "error", "error": he.detail}
        )
    except Exception as e:
        logger.error(f"Error saving LLM history: {e}")
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.post("/metrics")
async def record_metrics(body: Dict[str, Any], token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Upsert per-user and per-session token usage."""
    try:
        browser_id = str(body.get("browserId", "") or "")
        username = str(body.get("username", "") or "")
        ip = str(body.get("ip", "") or "")
        session_id = str(body.get("sessionId", "") or "")
        input_tokens = int(body.get("inputTokens", 0) or 0)
        output_tokens = int(body.get("outputTokens", 0) or 0)
        data_domains = body.get("dataDomains") or []
        if isinstance(data_domains, str):
            data_domains = [data_domains]
        data_domains = [str(domain) for domain in data_domains if str(domain).strip()]

        now = datetime.datetime.now(datetime.timezone.utc)
        await mongo_middleware.mongo_client.ensure_connection()

        async def resolve(result):
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                return await result
            return result

        collection = mongo_middleware.mongo_client.db["user_metrics"]
        await resolve(collection.update_one(
            {"browserId": browser_id, "username": username, "ip": ip},
            {
                "$inc": {"turns": 1, "inputTokens": input_tokens, "outputTokens": output_tokens},
                "$addToSet": {"sessionIds": session_id},
                "$setOnInsert": {"firstSeen": now},
                "$set": {"lastSeen": now},
            },
            upsert=True,
        ))

        if session_id:
            session_update = {
                "$inc": {"turns": 1, "inputTokens": input_tokens, "outputTokens": output_tokens},
                "$setOnInsert": {
                    "browserId": browser_id,
                    "username": username,
                    "ip": ip,
                    "startedAt": now,
                },
                "$set": {"lastActivityAt": now},
            }
            if data_domains:
                session_update["$push"] = {"domainUsage": {
                    "ts": now,
                    "dataDomains": data_domains,
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                }}
            await resolve(mongo_middleware.mongo_client.db["tracking_sessions"].update_one(
                {"sessionId": session_id}, session_update, upsert=True,
            ))
        else:
            logger.warning("[METRICS] no sessionId in body — skipping tracking_sessions rollup")
        return {"status": "ok"}
    except Exception as exc:
        logger.warning("[METRICS] FAILED: %s", exc)
        logger.warning("[METRICS] %s", "".join(traceback.format_exception(None, exc, exc.__traceback__)))
        return JSONResponse(status_code=500, content={"error": str(exc)})


# now that all the other API endpoints are established, lets add our mcp routes to the fastapi app.
app.mount(f"/{settings.TOOL_NAME}", mcp_app)
app.mount("/memory", memory_app)
app.mount("/agent", agent_app)


# These are not really used, left them in just in case.
def main():
    """
    Main entry point for the FastMCP server
    python mongo_mcp.py
    for the container call fastapi directly
    fastapi run mongo_mcp.py
    fastmcp mongo_mcp.py --transport sse --port 8001

    """
    #mcp.run(transport="sse", host="0.0.0.0", port=8001)
    #mcp.run(transport="sse",  port=8001) # this is for local IDE/Cline integration
    mcp.run(transport=settings.transport, host=settings.host, port=settings.port) # this is for AWS containers  


if __name__ == "__main__":
    main()
