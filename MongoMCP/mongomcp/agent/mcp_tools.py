"""
mongomcp.agent.mcp_tools
========================
Registers the agent orchestration tool (run_prompt) as a real FastMCP tool,
mirroring how register_memory_tools works for the memory layer.

Mount the returned app at /agent/mcp and expose /agent/llm_tools so the
webui discovers it exactly like any other endpoint — no synthetic toolspecs needed.
"""
import json
import asyncio
import anyio
import datetime
import logging
from typing import Any, Callable, Dict, List, Optional, Annotated

import fastmcp
import mcp.types as mt
from pydantic import Field
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastmcp.server.dependencies import get_access_token, AccessToken, get_http_request
from fastmcp import Context
from fastmcp.dependencies import CurrentContext
from starlette.requests import Request
from .prompt_agent import PromptAgent

logger = logging.getLogger(__name__)

# Strong refs to fire-and-forget sub-agent tasks (wait=False). Without this the event
# loop may garbage-collect an in-flight task; the done-callback discards + logs.
_BACKGROUND_AGENT_TASKS: set = set()

# How often a background (wait=False) sub-agent pushes a liveness heartbeat timestamp
# into its agent:run status doc. Each push re-embeds the status content, so keep it coarse.
_HEARTBEAT_INTERVAL_S = 30

_bearer = HTTPBearer(auto_error=False)

async def _get_raw_jwt(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)] = None,
) -> str:
    """Extract the raw JWT string from the Authorization header."""
    return credentials.credentials if credentials else ""

_AGENT_TOOL_DESCRIPTION = (
    "Run a focused sub-agent that executes a full Bedrock invoke loop with a specific "
    "prompt and an optionally filtered tool set.\n\n"
    "**REQUIRED WORKFLOW — always follow these steps before calling this tool:**\n"
    "1. Call memory_strategy_recall with the task description as the query to search for "
    "a matching strategy in memory.\n"
    "2. If a strategy is found: extract its _id as memory_id and its payload.tools list "
    "as tool_names, then pass both to this tool.\n"
    "3. If no strategy is found: call this tool without memory_id — the sub-agent will "
    "use all available tools.\n\n"
    "By default the sub-agent runs SYNCHRONOUSLY (wait=true): the call blocks until it "
    "finishes and returns its response_text. For FAN-OUT/parallel work, pass wait=false "
    "— the call returns IMMEDIATELY with {status:'started', status_memory_id, session_id} "
    "while the sub-agent runs in the background. POLL that record with "
    "memory_query(ids=[status_memory_id]) to read payload.status (started|completed|failed) "
    "plus response_preview/error; the sub-agent's own memory writes also persist. Re-fire "
    "only the gaps. Its tool calls route through the same MCP layer as all other tools."
)


def get_agent_bedrock_toolspecs() -> List[Dict[str, Any]]:
    """Return Bedrock-format toolSpec dicts for agent tools (bare names, no prefix).

    Called by /agent/llm_tools so the webui can discover and prefix the tools
    the same way it handles memory tools (memory_intake, memory_recall, etc.).
    """
    return [
        {
            "toolSpec": {
                "name": "run_prompt",
                "description": _AGENT_TOOL_DESCRIPTION,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The instruction or task for the sub-agent to execute.",
                            },
                            "agent_name": {
                                "type": "string",
                                "description": (
                                    "Short descriptive name for this sub-agent run. "
                                    "Prefixed on all progress messages in the UI "
                                    "(e.g. 'search_agent', 'batch1', 'summariser'). Required."
                                ),
                            },
                            "session_id": {
                                "type": "string",
                                "description": (
                                    "Session identifier. Use format "
                                    "'{username}:{session_id}:{YYYY-MM-DDTHH:MM}' so memory "
                                    "entries are grouped by user, session, and time. Always provide this."
                                ),
                            },
                            "tool_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Tool names to give the sub-agent. Populate from strategy_recall "
                                    "payload.tools. Accepts bare, dot-notation, or prefixed names. "
                                    "Memory tools always included. Pass empty list if no strategy found."
                                ),
                            },
                            "context": {
                                "type": ["object", "string"],
                                "description": "Optional structured or text context passed alongside the prompt.",
                            },
                            "memory_id": {
                                "type": "string",
                                "description": (
                                    "ObjectId hex of the strategy document returned by "
                                    "memory_strategy_recall. Sub-agent is instructed to load it "
                                    "for context and playbook."
                                ),
                            },
                            "system_instructions": {
                                "type": "string",
                                "description": "Platform-injected memory operating instructions. Do not set — leave empty.",
                            },
                            "wait": {
                                "type": "boolean",
                                "description": (
                                    "Whether to block until the sub-agent finishes (default true, returns "
                                    "response_text). Pass false to fire-and-forget for parallel fan-out: "
                                    "returns immediately with {status:'started', status_memory_id}; poll "
                                    "memory_query(ids=[status_memory_id]) for payload.status and results."
                                ),
                            },
                        },
                        "required": ["prompt", "agent_name", "session_id", "tool_names"],
                    }
                },
            }
        },
        *_function_builder_specs(),
        *_external_api_specs(),
    ]


def _function_builder_specs() -> List[Dict[str, Any]]:
    """Lazily pull in the define/promote query-function toolspecs (avoids import cycles)."""
    try:
        from .function_builder import get_function_builder_toolspecs
        return get_function_builder_toolspecs()
    except Exception as e:  # pragma: no cover - defensive
        logger.error("Failed to load function_builder toolspecs: %s", e)
        return []


def _external_api_specs() -> List[Dict[str, Any]]:
    """Lazily pull in the external-API connector toolspecs (avoids import cycles)."""
    try:
        from .external_api import get_external_api_toolspecs
        return get_external_api_toolspecs()
    except Exception as e:  # pragma: no cover - defensive
        logger.error("Failed to load external_api toolspecs: %s", e)
        return []


def build_mcp_http_call(jwt: str, base_url: Any) -> Callable[[str, dict], Any]:
    """Build an async router that dispatches an endpoint-prefixed tool call to the
    container that OWNS that endpoint, over the data-domain HTTP route.

    CROSS-CONTAINER ROUTING FOR DATA DOMAINS
    ----------------------------------------
    Every data domain (e.g. 'shipwreckSearch', 'AirbnbSearch') runs in its OWN container,
    connected to its OWN MongoDB database with its OWN credentials. A tool registered on
    the universal /agent container therefore must NEVER touch another domain's database
    directly — it has neither the right connection nor the right credentials, and doing so
    would breach domain isolation.

    Instead, tool names are ENDPOINT-PREFIXED as '<endpoint>_<tool>' (e.g.
    'shipwreckSearch_get_collection_info', 'memory_recall'). This router splits on the
    FIRST '_' to derive the endpoint, then sends the bare tool name to that endpoint's own
    MCP mount so the OWNING container executes it against its own DB connection:

      URL resolution (priority):
        1. MEMORY_RUNTIME_URL env  — when endpoint == 'memory' (AgentCore runtime split)
        2. MONGO_MCP_ROOT env      — '<root>/<endpoint>/mcp' for every other endpoint
        3. request.base_url        — legacy k8s mount-path fallback

    The caller's raw JWT is forwarded as the Bearer token so scopes/identity propagate
    unchanged. This is the single routing mechanism shared by the sub-agent (run_prompt)
    and by define_query_function's get_collection_info lookup — endpoint names are assumed
    to contain no underscores (true for all current domains).
    """
    async def _http_call(toolname: str, tool_input: dict) -> Any:
        import os as _os
        if "_" not in toolname:
            return {"error": f"Cannot resolve endpoint for unprefixed tool '{toolname}'"}
        endpoint_name, endpoint_tool_name = toolname.split("_", 1)
        if not jwt:
            return {"error": "No bearer token available for cross-container dispatch"}

        _mongo_mcp_root = _os.environ.get("MONGO_MCP_ROOT", "").rstrip("/")
        _memory_runtime_url = _os.environ.get("MEMORY_RUNTIME_URL", "").rstrip("/")

        if endpoint_name == "memory" and _memory_runtime_url:
            mcp_url = _memory_runtime_url
        elif _mongo_mcp_root:
            mcp_url = f"{_mongo_mcp_root}/{endpoint_name}/mcp"
        else:
            if not base_url:
                return {"error": "Cannot determine MCP root URL from request or settings"}
            mcp_url = f"{base_url}{endpoint_name}/mcp"

        cfg = {
            "url": mcp_url,
            "transport": "http",
            "headers": {"Authorization": f"Bearer {jwt}"},
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
    return _http_call


def register_agent_tools(
    mcp,
    settings: Any,
    get_tools_fn: Callable[[], List[Dict[str, Any]]],
    save_fn: Optional[Callable] = None,
    local_call_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Register agent orchestration tools on the given FastMCP instance.

    Parameters
    ----------
    mcp          : FastMCP instance (already configured with auth)
    settings     : Application settings — must expose ``mongo_mcp_root`` (MONGO_MCP_ROOT env)
    get_tools_fn : Callable ``() -> List[dict]`` — returns endpoint-prefixed Bedrock toolSpec
                   dicts for the sub-agent's tool catalog (no agent_ tools to prevent recursion)
    local_call_fn: Optional async ``(token, prefixed_toolname, tool_input) -> (handled, result)``.
                   When the target endpoint is served by THIS process (its own data domain +
                   memory), the sub-agent's tool call is executed IN-PROCESS (handled=True)
                   instead of an HTTP self-loopback — this avoids the connection saturation
                   (ConnectTimeout / 60s waits) seen under fan-out. handled=False → HTTP.

    Returns
    -------
    dict mapping tool_name -> fn for inclusion in _TOOL_DISPATCH

    Dispatch strategy
    -----------------
    Tool calls from the sub-agent are routed via HTTP back through the load balancer using
    ``settings.mongo_mcp_root``.  Tool names in the catalog are endpoint-prefixed
    (e.g. ``memory_recall``, ``AirbnbSearch_vector_search``); the prefix is split on the
    first ``_`` to derive the MCP endpoint path.  The caller's ``AccessToken.token`` (raw
    JWT string) is forwarded as the Bearer token so permissions propagate unchanged.
    """
    

    @mcp.tool()
    async def run_prompt(
        prompt: Annotated[str, Field(description="The instruction or task for the sub-agent to execute.")],
        agent_name: Annotated[str, Field(description="Short descriptive name for this sub-agent run. Prefixed on all progress messages (e.g. 'search_agent', 'batch1'). Required.")],
        session_id: Annotated[str, Field(description="Session identifier. Use format '{username}:{session_id}:{YYYY-MM-DDTHH:MM}' so memory entries are grouped by user, session, and time.")],
        tool_names: Annotated[List[str], Field(description="Tool names for the sub-agent. Populate from strategy_recall payload.tools. Memory tools always included. Pass empty list if no strategy found.")],
        context: Annotated[Optional[str], Field(default=None, description="Optional structured or text context passed alongside the prompt.")] = None,
        memory_id: Annotated[Optional[str], Field(default=None, description="ObjectId hex of a strategy document. Sub-agent is instructed to load it for context and playbook.")] = None,
        system_instructions: Annotated[Optional[str], Field(default=None, description="Platform-injected memory operating instructions. Do not set.")] = None,
        wait: Annotated[bool, Field(default=True, description="Block until the sub-agent finishes (default). Pass false to fire-and-forget for parallel fan-out and poll memory by session_id for results.")] = True,
        token: Annotated[AccessToken, Depends(get_access_token)] = None,
        ctx: Context = CurrentContext(),
    ):
        """Run a focused sub-agent Bedrock invoke loop with filtered tools and streaming progress.

        REQUIRED WORKFLOW: call memory_strategy_recall first to find a matching strategy,
        then pass its _id as memory_id and payload.tools as tool_names.
        """
        # Resolve the raw JWT once — token.token if injected, otherwise the raw_jwt Depends value.
        jwt = None
        if token is None:
            token = get_access_token()
        if isinstance(token, dict):
            jwt = token.get("token")
        elif token is not None:
            jwt = token.token
        request: Request = get_http_request()
        base_url =  request.base_url

        # Cross-container routing for data domains: endpoint-prefixed tool names are
        # dispatched to the OWNING domain's container over the data-domain route,
        # forwarding this JWT. See build_mcp_http_call for the full routing rationale.
        _http_call = build_mcp_http_call(jwt, base_url)

        async def mcp_call_fn(toolname: str, tool_input: dict) -> Any:
            """Route tool calls; a sub-agent may use any tool EXCEPT spawning another
            sub-agent. Prefer in-process dispatch for tools this process hosts (its own
            data domain + memory + agent), else HTTP to the owning container."""
            # Block ONLY sub-agent recursion — every other agent-domain tool is allowed.
            if toolname in ("run_prompt", "agent_run_prompt"):
                return {"error": f"Sub-agents cannot spawn sub-agents: '{toolname}' is blocked."}
            if local_call_fn is not None:
                try:
                    handled, result = await local_call_fn(token, toolname, tool_input)
                    if handled:
                        return result
                except Exception as _local_err:
                    logger.warning(
                        "[run_prompt] in-process dispatch failed for %s; falling back to HTTP: %s",
                        toolname, _local_err,
                    )
            return await _http_call(toolname, tool_input)

        tool_catalog = get_tools_fn()
        agent = PromptAgent(
            settings=settings,
            mcp_call_fn=mcp_call_fn,
            tool_catalog=tool_catalog,
            save_fn=save_fn,
        )

        # --- Fire-and-forget (wait=False): spawn the sub-agent as a detached task and
        # return immediately. A lifecycle STATUS doc is written to episodic memory FIRST
        # (memory_type="agent:run") and its _id is returned as `status_memory_id` so the
        # parent can poll definitive status. The background task flips that doc to
        # completed/failed in its finally. The task runs to completion on the event loop
        # after this request returns; it uses only captured values (jwt/base_url via
        # mcp_call_fn, token) and touches no request-scoped context. ---
        if not wait:
            _username = session_id.split(":")[0] if session_id and ":" in session_id else None
            _started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _status_content = f"Sub-agent run '{agent_name}' (session {session_id}) status tracker."
            _base_payload = {
                "agent_name": agent_name,
                "session_id": session_id,
                "tool_names": tool_names or [],
                "started_at": _started_at,
            }

            status_id = None
            try:
                _raw = await mcp_call_fn("memory_intake", {
                    "content": _status_content,
                    "memory_type": "agent:run",
                    "session_id": session_id,
                    "scope": 30,
                    "importance": 0.5,
                    "decay_rate": 0.02,
                    "tags": ["agent:run", agent_name],
                    "entities": [agent_name],
                    "username": _username,
                    "payload": {**_base_payload, "status": "started", "heartbeats": []},
                })
                _parsed = json.loads(_raw) if isinstance(_raw, str) else _raw
                if isinstance(_parsed, dict):
                    status_id = _parsed.get("id")
            except Exception as _st_err:
                logger.warning("[run_prompt] failed to write status doc for agent=%s: %s", agent_name, _st_err)

            async def _heartbeat_loop(stop_evt: "asyncio.Event", beats: List[str]):
                """Append a liveness timestamp to payload.heartbeats every interval until stopped."""
                while not stop_evt.is_set():
                    try:
                        await asyncio.wait_for(stop_evt.wait(), timeout=_HEARTBEAT_INTERVAL_S)
                        break  # stop signalled
                    except asyncio.TimeoutError:
                        pass  # interval elapsed → emit a beat
                    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    beats.append(ts)
                    if status_id:
                        try:
                            await mcp_call_fn("memory_intake", {
                                "_id": status_id,
                                "memory_type": "agent:run",
                                "scope": 30,
                                "importance": 0.5,
                                "decay_rate": 0.02,
                                "payload_push": {"heartbeats": ts},
                            })
                        except Exception as _hb_err:  # pragma: no cover - best-effort
                            logger.debug("[run_prompt] heartbeat push failed for %s: %s", status_id, _hb_err)

            async def _run_detached():
                _final: Dict[str, Any] = dict(_base_payload)
                _beats: List[str] = []
                _stop = asyncio.Event()
                _hb_task = asyncio.create_task(_heartbeat_loop(_stop, _beats)) if status_id else None
                try:
                    _res = await agent.run(
                        prompt=prompt,
                        context=context,
                        memory_id=memory_id,
                        tool_names=tool_names or None,
                        session_id=session_id,
                        system_instructions=system_instructions,
                        token=token,
                    )
                    if isinstance(_res, dict) and _res.get("error"):
                        _final["status"] = "failed"
                        _final["error"] = str(_res.get("error"))
                    else:
                        _final["status"] = "completed"
                        _final["response_preview"] = str((_res or {}).get("response_text") or "")[:500]
                except Exception as bg_err:  # pragma: no cover - background best-effort
                    logger.error("[run_prompt] background agent=%s failed: %s", agent_name, bg_err)
                    _final["status"] = "failed"
                    _final["error"] = str(bg_err)
                finally:
                    # Stop heartbeats FIRST so no push races the final payload $set.
                    _stop.set()
                    if _hb_task is not None:
                        try:
                            await _hb_task
                        except Exception:
                            pass
                    _final["heartbeats"] = _beats
                    _final["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    if status_id:
                        try:
                            await mcp_call_fn("memory_intake", {
                                "_id": status_id,
                                "content": f"Sub-agent run '{agent_name}' (session {session_id}) status: {_final.get('status')}.",
                                "memory_type": "agent:run",
                                "scope": 30,
                                "importance": 0.5,
                                "decay_rate": 0.02,
                                "tags": ["agent:run", agent_name],
                                "entities": [agent_name],
                                "payload": _final,
                            })
                        except Exception as _up_err:
                            logger.warning("[run_prompt] failed to update status doc %s: %s", status_id, _up_err)

            bg_task = asyncio.create_task(_run_detached())
            _BACKGROUND_AGENT_TASKS.add(bg_task)
            bg_task.add_done_callback(_BACKGROUND_AGENT_TASKS.discard)
            logger.info(
                "[run_prompt] agent=%s started in background (wait=false) session=%s status_id=%s",
                agent_name, session_id, status_id,
            )
            return {
                "status": "started",
                "agent_name": agent_name,
                "session_id": session_id,
                "status_memory_id": status_id,
                "note": (
                    "Sub-agent running in the background; this call did not wait. Poll "
                    "memory_query(ids=['<status_memory_id>']) to read payload.status "
                    "(started | completed | failed) plus response_preview/error. Do NOT block "
                    "on this call."
                ),
            }

        try:
            result = await agent.run(
                prompt=prompt,
                context=context,
                memory_id=memory_id,
                tool_names=tool_names or None,
                session_id=session_id,
                system_instructions=system_instructions,
                token=token,
            )
        except anyio.ClosedResourceError:
            logger.warning(
                "[run_prompt] agent=%s session orphaned (client disconnected) — "
                "agent completed in background, result saved to llm_history",
                agent_name,
            )
            return {"status": "orphaned", "message": "MCP session closed by client; result saved to llm_history."}

        # Return only what the parent LLM needs — omit prompt/history/usage/stats.
        output: Dict[str, str] = {"response_text": result.get("response_text", "")}
        if result.get("error"):
            output["error"] = result["error"]
        return output

    return {"run_prompt": run_prompt}
