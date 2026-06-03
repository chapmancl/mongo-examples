import asyncio
import copy
import datetime
import json
import os
from bson import ObjectId
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Annotated
import logging
from pydantic import Field
import fastmcp
import mcp.types as mt
from fastmcp import FastMCP
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastmcp.server.dependencies import AccessToken
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from AWS_settings import settings
#from local_settings import settings # change this to use AWS_settings
from mongomcp import MongoDBQueryServer, MongoMCPMiddleware, ServerBedrockClient, MongoTokenVerifier, register_memory_tools, register_query_tools, get_memory_bedrock_toolspecs, register_agent_tools, get_agent_bedrock_toolspecs, __version__ as MCP_VERSION
from mongomcp.mongodb_client import query_capture_cv as _mongo_capture_cv, _query_capture_registry as _mongo_capture_registry, set_query_capture_enabled as _set_query_capture_enabled, _CAPTURE_LISTENER as _mongo_capture_listener
from mongomcp.agent.prompt_agent import PromptAgent
from mongomcp.agent.tool_router import ToolRouter
import traceback
import os
import sys
import time
import uuid

# logs were getting very bloated, lets reduce that a bit.
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
logging.getLogger("mongomcp.mongo_mcp_middleware").setLevel(logging.INFO)
logging.getLogger("mongomcp.mongodb_client").setLevel(logging.WARNING)
logging.getLogger("mongomcp.memory").setLevel(logging.WARNING)
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

# MCP_AUTH_ENABLED=false (default): FastMCP accepts any well-formed JWT and extracts
# identity for logging, but skips MongoDB signature validation. Use when the container
# sits behind API Gateway which already validated the Bearer token.
# MCP_AUTH_ENABLED=true: enforces MongoDB agent_identities lookup (strict mode).
_mcp_auth_enabled = os.environ.get("MCP_AUTH_ENABLED", "false").lower() in ("1", "true", "yes")
if auth_provider is not None:
    auth_provider.strict = _mcp_auth_enabled
if not _mcp_auth_enabled:
    logger.info("FastMCP auth in non-strict mode (MCP_AUTH_ENABLED=false) — identity parsed from token, validation skipped")

# Create FastMCP server instance with bearer token authentication
# this is the mongo tools from config load.
llm_client = ServerBedrockClient(settings)
mcp = FastMCP("mongodb-vector-server", auth=auth_provider)
mcp.add_middleware(mongo_middleware)
_query_dispatch = register_query_tools(mcp, mongo_server, llm_client, mongo_middleware.endpoint_tools)


# Separate FastMCP instance for the memory layer — keeps memory tools off the main tool catalog.
_agent_instructions = getattr(settings, "agent_instructions", None)
memory_mcp = FastMCP("memory-server", auth=auth_provider, instructions=_agent_instructions or None)
memory_mcp.add_middleware(mongo_middleware)
_memory_dispatch = register_memory_tools(memory_mcp, mongo_server, llm_client, settings)

# Agent layer — run_prompt tool, always available like memory.
agent_mcp = FastMCP("agent-server", auth=auth_provider)
agent_mcp.add_middleware(mongo_middleware)

def _get_agent_tool_catalog():
    """Full tool catalog for sub-agents with endpoint-prefixed names.

    Queries ALL active endpoints from MongoDB so the sub-agent can call tools
    across every container, not just the current one.  Tool names are prefixed
    as '<endpoint_name>_<tool_name>' so _make_mcp_call_fn can split on the
    first '_' to route each call to the correct MCP mount path.
    Memory tools are always appended with the 'memory_' prefix.
    Agent tools are excluded to prevent recursion.
    """
    # Memory tools are always included — build them first so they survive any
    # endpoint-loading failure below.
    memory_tools = []
    for t in get_memory_bedrock_toolspecs():
        t = copy.deepcopy(t)
        t["toolSpec"]["name"] = f"memory_{t['toolSpec']['name']}"
        memory_tools.append(t)

    endpoint_tools = []
    try:
        # Load tools from ALL active endpoints (each prefixed with its endpoint name).
        for t in mongo_middleware.build_tools_from_all_endpoints():
            endpoint_tools.append(copy.deepcopy(t))
    except Exception as e:
        logger.error("_get_agent_tool_catalog: failed to load endpoint tools: %s", e)

    return endpoint_tools + memory_tools

register_agent_tools(agent_mcp, settings, _get_agent_tool_catalog, mongo_middleware.save_llm_conversation)


# Custom headers used to propagate query-capture context across container boundaries.
# doc_id ties all captures to the originating llm_history document;
# tool_name is carried so the receiving pod knows which tool triggered each query.
_CAPTURE_HEADER = "x-capture-doc-id"
_CAPTURE_TOOL_HEADER = "x-capture-tool"


async def _push_query_log(doc_id: str, tool_name: str, captured: list) -> None:
    """Fire-and-forget: push captured MongoDB pipelines to llm_history.queries_used.

    Called from _QueryCaptureMiddleware after each tool-call response so that
    any pod — not just the originating one — can write its captures.
    Uses Motor (async) via ensure_connection() and bypasses the collection cache
    to avoid getting a stale sync PyMongo collection.
    """
    entries = []
    for q in captured:
        entry = {
            "tool": tool_name,
            "command": q.get("command"),
            "database": q.get("database"),
            "collection": q.get("collection"),
            "ts": datetime.datetime.now().isoformat(),
        }
        if "pipeline" in q:
            entry["pipeline"] = q["pipeline"]
        else:
            for field in ("filter", "projection", "sort", "limit"):
                if field in q:
                    entry[field] = q[field]
        entries.append(entry)
    try:
        # ensure_connection() will recreate the Motor client if sync_connect replaced it.
        await mongo_middleware.mongo_client.ensure_connection()
        # Bypass the collection cache — it may hold a stale sync PyMongo collection.
        coll = mongo_middleware.mongo_client.db["llm_history"]
        result = coll.update_one(
            {"_id": ObjectId(doc_id)},
            {"$push": {"queries_used": {"$each": entries}}},
        )
        # coll may be Motor (coroutine) or sync PyMongo (UpdateResult already done).
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            await result
    except Exception as e:
        logger.warning("query_log push failed (doc_id=%s tool=%s): %s", doc_id, tool_name, e)


class _QueryCaptureMiddleware:
    """Pure-ASGI middleware that captures MongoDB queries for any request carrying
    _CAPTURE_HEADER.  Works across container boundaries: each pod captures its own
    Motor queries and writes them directly to llm_history via _push_query_log, so
    the load balancer can route tool calls to any replica.

    A per-request registry key (doc_id + scope id) isolates concurrent captures.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and _mongo_capture_listener.enabled:
            headers = dict(scope.get("headers", []))
            doc_id = headers.get(_CAPTURE_HEADER.encode(), b"").decode()
            if doc_id:
                tool_name = headers.get(_CAPTURE_TOOL_HEADER.encode(), b"").decode() or "unknown"
                # Per-request key so concurrent calls to the same doc_id don't collide.
                req_key = f"{doc_id}:{id(scope)}"
                _mongo_capture_registry[req_key] = []
                cv_token = _mongo_capture_cv.set(req_key)
                try:
                    await self.app(scope, receive, send)
                finally:
                    _mongo_capture_cv.reset(cv_token)
                    captured = _mongo_capture_registry.pop(req_key, [])
                    if captured:
                        asyncio.create_task(_push_query_log(doc_id, tool_name, captured))
                return
        await self.app(scope, receive, send)


def _make_mcp_call_fn(base_url: str, jwt: str, capture_doc_id: Optional[str] = None):
    """Return an async mcp_call_fn that dispatches endpoint-prefixed tool calls via HTTP.

    Tool names must be prefixed with their endpoint (e.g. 'memory_recall',
    'claimsSearch_vector_search').  The prefix is split on the first '_' to
    derive the MCP endpoint path.  Agent tools are blocked to prevent recursion.
    When capture_doc_id is provided the header is forwarded so the receiving
    task can set query_capture_cv for CommandListener correlation.
    """
    async def mcp_call_fn(toolname: str, tool_input: dict) -> Any:
        if toolname == "run_prompt" or toolname.startswith("agent_"):
            return {"error": f"Agent tool recursion prevented: {toolname}"}
        if "_" not in toolname:
            return {"error": f"Cannot resolve endpoint for unprefixed tool '{toolname}'"}
        endpoint_name, endpoint_tool_name = toolname.split("_", 1)
        _headers = {"Authorization": f"Bearer {jwt}"}
        if capture_doc_id:
            _headers[_CAPTURE_HEADER] = capture_doc_id
            _headers[_CAPTURE_TOOL_HEADER] = toolname
        cfg = {
            "url": f"{base_url}{endpoint_name}/mcp",
            "transport": "http",
            "headers": _headers,
        }
        client = fastmcp.Client({"mcpServers": {endpoint_name: cfg}}, timeout=60)
        last_exc = None
        for attempt in range(2):  # 1 retry for transient 502/503/504
            try:
                async with client:
                    raw = await client.session.send_request(
                        mt.ClientRequest(
                            mt.CallToolRequest(
                                params=mt.CallToolRequestParams(
                                    name=endpoint_tool_name,
                                    arguments=tool_input,
                                )
                            )
                        ),
                        mt.CallToolResult,
                    )
                if raw.content and hasattr(raw.content[0], "text"):
                    return raw.content[0].text
                if raw.structuredContent is not None:
                    return json.dumps(raw.structuredContent)
                return str(raw)
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if attempt == 0 and any(code in err_str for code in ("502", "503", "504")):
                    logger.warning("HTTP dispatch transient error for %s (attempt %d): %s — retrying", toolname, attempt + 1, exc)
                    await asyncio.sleep(1)
                    continue
                break
        logger.error("HTTP dispatch failed for %s: %s", toolname, last_exc)
        return {"error": f"Tool call failed: {str(last_exc)}"}
    return mcp_call_fn


#***********  BEGIN FASTAPI SECTION  ***************

# We have our tools, mount the mcp to fastapi and setup our fastapi authentication
# everything after this should be FastAPI endpoints.
# both of these get registered here, but memory will get its own route below
mcp_app = mcp.http_app(path=f"/mcp")
memory_app = memory_mcp.http_app(path="/mcp")
agent_app = agent_mcp.http_app(path="/mcp")


@asynccontextmanager
async def _combined_lifespan(app):
    async with mcp_app.lifespan(app):
        async with memory_app.lifespan(app):
            async with agent_app.lifespan(app):
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


def _resolve_tool_callable(tool_obj):
    """Return the underlying callable for either FastMCP tool wrappers or plain functions."""
    return getattr(tool_obj, "fn", tool_obj)

# Dispatch table: tool_name → callable for the HTTP-path tool_handler.
# All query tools come from _query_dispatch; memory tools from _memory_dispatch.
_TOOL_DISPATCH = {
    **_query_dispatch,
    **_memory_dispatch,
}

# Frozen set of memory tool names — handled with token passthrough so wrappers
# can derive agent_id internally.
_MEMORY_TOOL_NAMES = frozenset(_memory_dispatch.keys())

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
        # Inject config-backed collection/geo_field via shared middleware method.
        kwargs = mongo_middleware.inject_collection_args(toolname, kwargs)
        # Memory wrappers derive agent_id from token internally.
        if toolname in _MEMORY_TOOL_NAMES:
            # Never pass agent_id directly; not all memory wrapper signatures expose it.
            kwargs.pop("agent_id", None)
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
async def http_get_llm_tools(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Returns preformatted Bedrock toolSpec JSON for the active tool endpoint (MongoDB annotations)."""
    tools = mongo_middleware.build_tools_from_annotations()
    return {"tools": tools, "count": len(tools)}


@app.get("/memory/llm_tools")
async def http_get_memory_llm_tools(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Returns preformatted Bedrock toolSpec JSON for all memory layer tools."""
    tools = get_memory_bedrock_toolspecs()
    return {"tools": tools, "count": len(tools)}


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
    if "llm:invoke" not in token.get("scope", []):
        logger.error(
            "Insufficient scope for invoke_llm: llm:invoke permission required for agent %s",
            token.get("agent_key"),
        )
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

        # Include both endpoint tools and memory tools so invoke_llm prompts can use
        # memory intake/recall/reflect flows in the same request path.
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

            # Keep the same warning semantics as query path when token limit is exceeded.
            err_msg = str(resp_obj.get("error", ""))
            if err_msg and "prompt is too long" in err_msg.lower():
                _mark_context_limit_warning(err_msg)

            # lots of potential errors and exceptions here, so catch them all.
            # tried to pass most through the return, but some may still raise
            # I could not handle all the exceptions by name either, some would raise a runtime exception
            # instead of passing the exception directly
            return_json = {}
            if resp_obj.get("error"):
                status_code = 413 if "prompt is too long" in err_msg.lower() else 500
                return_json = JSONResponse(output, status_code)
            else:
                logger.info(f"invoke successful for prompt {prompt_name}")
                return_json = JSONResponse(output, 201)

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
        err_msg = f"Error executing invoke_llm: {str(e)}"
        if "prompt is too long" in str(e).lower():
            _mark_context_limit_warning(err_msg)
            _save_output_snapshot()
            return JSONResponse(output, 413)
        output["error"] = err_msg
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
    mcp.run(transport="http", host="0.0.0.0", port=8000, log_level="warning") # this is for AWS containers


if __name__ == "__main__":
    main()
