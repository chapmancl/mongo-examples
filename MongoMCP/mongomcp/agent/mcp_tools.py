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
    "The sub-agent runs synchronously and streams progress notifications back to the UI. "
    "Its tool calls route through the same MCP layer as all other tools."
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
                        },
                        "required": ["prompt", "agent_name", "session_id", "tool_names"],
                    }
                },
            }
        }
    ]


def register_agent_tools(
    mcp,
    settings: Any,
    get_tools_fn: Callable[[], List[Dict[str, Any]]],
    save_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Register agent orchestration tools on the given FastMCP instance.

    Parameters
    ----------
    mcp          : FastMCP instance (already configured with auth)
    settings     : Application settings — must expose ``mongo_mcp_root`` (MONGO_MCP_ROOT env)
    get_tools_fn : Callable ``() -> List[dict]`` — returns endpoint-prefixed Bedrock toolSpec
                   dicts for the sub-agent's tool catalog (no agent_ tools to prevent recursion)

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

        async def _http_call(toolname: str, tool_input: dict) -> Any:
            """Route a tool call via HTTP to the correct MCP endpoint.

            Tool names must be endpoint-prefixed (e.g. 'memory_recall',
            'AirbnbSearch_vector_search'). The endpoint name is the portion before
            the first '_'; the remainder is the bare tool name sent to that endpoint.
            Assumes endpoint names contain no underscores (true for all current endpoints).

            Routing priority (allows AgentCore agent runtime to override base_url):
              1. MEMORY_RUNTIME_URL env var — used when endpoint_name == "memory"
              2. MONGO_MCP_ROOT env var    — used for all other endpoints
              3. request.base_url fallback — legacy k8s mount-path routing
            """
            import os as _os
            if "_" not in toolname:
                return {"error": f"Cannot resolve endpoint for unprefixed tool '{toolname}'"}
            endpoint_name, endpoint_tool_name = toolname.split("_", 1)
            if not jwt:
                return {"error": "No bearer token available for sub-agent dispatch"}

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

        async def mcp_call_fn(toolname: str, tool_input: dict) -> Any:
            """Route tool calls via HTTP; block agent tool recursion."""
            if toolname == "run_prompt" or toolname.startswith("agent_"):
                return {"error": f"Agent tool recursion prevented: {toolname}"}
            return await _http_call(toolname, tool_input)

        tool_catalog = get_tools_fn()
        agent = PromptAgent(
            settings=settings,
            mcp_call_fn=mcp_call_fn,
            tool_catalog=tool_catalog,
            save_fn=save_fn,
        )

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
