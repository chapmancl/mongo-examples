import asyncio
import json
import queue
import time
import traceback
from typing import Any, List, Optional

import fastmcp
import mcp.types as mt
import requests
from pydantic import BaseModel

from aws_settings import settings
from mongomcp.agent.cache_utils import create_cache_key as _create_cache_key
from mongomcp.mongo_cache import MongoSessionCache
from mongomcp.agent.tool_router import ToolRouter
from mongomcp.agent.webui_bedrock_client import WebUiBedrockClient

import logging
logger = logging.getLogger(__name__)


class QueryResponse(BaseModel):
    content: Optional[dict] = None
    error: Optional[str] = None
    status: Optional[str] = None
    history: Optional[List[Any]] = None
    message: Optional[str] = None
    clear_history: Optional[bool] = None

    def json(self):
        if self.message is not None:
            self.message = self.message.replace("\n", " ").replace("\r", "")
        return self.model_dump_json()


class QueryRequest(BaseModel):
    input: str
    history: Optional[List[Any]] = None
    user_id: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None


def _annotate_cached(result: Any) -> Any:
    """Inject _cached=true into a JSON tool result string so it's visible in the UI tree viewer."""
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                parsed["_cached"] = True
                return json.dumps(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(result, dict):
        result["_cached"] = True
    return result


class APIQueryProcessor:
    _NO_TOOLS_ERROR = "No MCP tools configured. Tool discovery may have failed."
    TOOLS_CACHE_TTL_SECONDS = 300  # re-discover tools every 5 minutes

    # Tool suffixes that return stable schema/index data — cache for 1 hour.
    _CACHED_TOOL_SUFFIXES = frozenset({
        "list_indexes",
        "list_search_indexes",
    })
    _CACHED_TOOL_TTL = 3600  # seconds

    def __init__(self):
        self._init_error: Optional[Exception] = None
        self._message_queue: queue.Queue = queue.Queue()

        logger.info(f"Initializing processor with endpoint: {settings.mongo_mcp_root}")
        try:
            self.llm_client = WebUiBedrockClient(settings)
            self.mcp_endpoints: List[str] = []
            self.mcp_endpoint_configs: dict = {}
            self.endpoint_clients: dict = {}
            self.mongo_collection_info: dict = {}
            self.mcp_tools_config: Optional[List[dict]] = None
            self._session_instructions: Optional[str] = None
            self._tools_fetched_at: Optional[float] = None
            self._base_system_prompt: List[dict] = []
            _cache_ns = getattr(settings, "CACHE_NAMESPACE", "global")
            self._tool_response_cache = MongoSessionCache(
                settings, username="webui", session_id=_cache_ns,
                cache_object_name="tool_response",
                per_document=True,  # Each entry is its own document to avoid 16MB limit
            )
            # Feature flags from settings
            self.enable_mcp_tool_caching = getattr(settings, "ENABLE_MCP_TOOL_CACHING", False)
            self.enable_response_caching = getattr(settings, "ENABLE_RESPONSE_CACHING", False)
            # Tool discovery cache — same MongoDB collection, different cache_object_name.
            if self.enable_mcp_tool_caching:
                self._tool_discovery_cache = MongoSessionCache(
                    settings, username="webui", session_id=_cache_ns,
                    cache_object_name="tool_discovery",
                )
            else:
                self._tool_discovery_cache = None
            self._discover_tools()
        except Exception as e:
            self._init_error = e
            logger.error(f"Processor initialization failed: {e}")

    @property
    def _headers(self) -> dict:
        """Build auth headers fresh on every call so Cognito tokens are never stale."""
        return {
            "Authorization": f"Bearer {settings.get_auth_token()}",
            "Content-Type": "application/json",
        }

    @property
    def init_error(self) -> Optional[Exception]:
        return self._init_error

    def _emit(self, message, status="Processing"):
        if isinstance(message, Exception):
            resp = QueryResponse(status="Error", error=str(message), message=str(message))
        else:
            resp = QueryResponse(status=status, message=str(message))
        self._message_queue.put(resp.json())

    def pop_queued_messages(self) -> List[str]:
        msgs = []
        try:
            while True:
                msgs.append(self._message_queue.get_nowait())
        except queue.Empty:
            pass
        return msgs

    def read_message_stream(self, timeout=0.1):
        try:
            while True:
                try:
                    yield self._message_queue.get(timeout=timeout)
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error reading message stream: {e}")

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def _discover_tools(self, emit=None):
        asyncio.run(self._discover_tools_async(emit=emit))

    def _looks_like_no_tools_error(self, result: dict) -> bool:
        err = (result or {}).get("error")
        if not isinstance(err, str):
            return False
        return self._NO_TOOLS_ERROR in err

    def _rediscover_tools(self, emit=None) -> bool:
        """Refresh local tool discovery, retrying up to 3 times with back-off."""
        import time
        emit_fn = emit or self._emit
        for attempt in range(1, 4):
            try:
                self._discover_tools(emit=emit_fn)
                if self.mcp_tools_config:
                    return True
            except Exception as e:
                logger.error(f"Tool rediscovery attempt {attempt} failed: {e}")
                emit_fn(f"Tool rediscovery attempt {attempt} failed: {e}", status="Error")
            if attempt < 3:
                emit_fn(f"Retrying tool discovery (attempt {attempt + 1}/3)...", status="Recovering")
                time.sleep(attempt * 2)
        emit_fn("Tool rediscovery exhausted all attempts.", status="Error")
        return False

    async def _discover_tools_async(self, emit=None):
        emit_fn = emit or self._emit

        # Check MongoDB discovery cache first (persists across restarts).
        _DISCOVERY_CACHE_KEY = "mcp_tools_discovery_v1"
        if self._tool_discovery_cache is not None:
            try:
                self._tool_discovery_cache.reset_connection()
                cached = await self._tool_discovery_cache.get(_DISCOVERY_CACHE_KEY)
                if cached is not None:
                    self.mcp_endpoints = cached.get("endpoints", [])
                    self.mcp_endpoint_configs = cached.get("endpoint_configs", {})
                    self.mongo_collection_info = cached.get("collection_info", {})
                    bedrock_tools = cached.get("tools", [])
                    system_prompt = cached.get("system_prompt", [])
                    self._base_system_prompt = system_prompt
                    self.llm_client.system = system_prompt
                    self.mcp_tools_config = bedrock_tools
                    self._tools_fetched_at = time.monotonic()
                    self.llm_client.configure_tools(bedrock_tools, self._call_mcp_tool)
                    emit_fn(
                        f"Ready (cached): {len(bedrock_tools)} tools from {len(self.mcp_endpoints)} endpoint(s)",
                        status="Tools Ready",
                    )
                    return
            except Exception as exc:
                logger.warning("Tool discovery cache read failed: %s", exc)

        try:
            # Agentcore gateway serves discovery at GET /  (returns {"services": [...]}).
            # Legacy deployments serve it at /tools_config (returns {"available_tools": [...]}).
            # Try the root first; fall back to /tools_config on 404.
            for discovery_path in ("/", "/tools_config"):
                resp = requests.get(
                    f"{settings.mongo_mcp_root}{discovery_path}",
                    headers=self._headers, timeout=15,
                )
                logger.info("Discovery %s -> HTTP %s", discovery_path, resp.status_code)
                if resp.status_code == 404 and discovery_path == "/":
                    continue
                resp.raise_for_status()
                data = resp.json()
                logger.info("Discovery response keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))
                self.mcp_endpoints = data.get("available_tools") or data.get("services", [])
                break
            logger.info("Discovered endpoints: %s", self.mcp_endpoints)
            emit_fn(f"Discovered endpoints: {self.mcp_endpoints}", status="Discovering Tools")
        except Exception as e:
            logger.error("Discovery failed: %s", e, exc_info=True)
            emit_fn(f"Error fetching endpoint list: {e}", status="Error")
            self.mcp_endpoints = []

        root_fmt = f"{settings.mongo_mcp_root}/{{}}/mcp"
        results = await asyncio.gather(*[
            self._fetch_endpoint(name, root_fmt, emit=emit_fn) for name in self.mcp_endpoints
        ])

        bedrock_tools, agent_prompts = [], {}
        for name, config, tools, collection_info, agent_prompt in results:
            self.mcp_endpoint_configs[name] = config
            bedrock_tools.extend(tools)
            if collection_info:
                self.mongo_collection_info[name] = collection_info
            if agent_prompt and agent_prompt.strip():
                agent_prompts[name] = agent_prompt.strip()

        # Fetch DB-driven instructions (master_instructions + webui-specific overrides).
        # Only attempt if the memory endpoint actually loaded tools — if memory/mcp is
        # unreachable (e.g. no ingress route on K8s), skip to avoid blocking startup.
        # Also cap each call at 15s so a slow endpoint never kills the gunicorn worker.
        memory_has_tools = any(
            t.get("toolSpec", {}).get("name", "").startswith("memory_")
            for t in bedrock_tools
        )
        db_prompts = []
        if not memory_has_tools:
            logger.info("Skipping strategy recall — memory endpoint has no tools (unreachable or not deployed)")
        else:
            for strategy_name in ("master_instructions", "dynamicmcp_webui_instructions"):
                try:
                    raw = await asyncio.wait_for(
                        self._call_mcp_tool("memory_strategy_recall", {"name": strategy_name}),
                        timeout=15,
                    )
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    strategies = (data or {}).get("strategies") or (data or {}).get("results", [])
                    if strategies:
                        content = strategies[0].get("content", "")
                        if content:
                            db_prompts.append(content)
                except Exception as exc:
                    logger.warning("Failed to load strategy %s: %s", strategy_name, exc)

        if db_prompts:
            system_prompt = [{"text": t} for t in db_prompts]
        else:
            system_prompt = [{"text": t} for t in getattr(settings, "BEDROCK_SYSTEM_PROMPT_TEXTS", [])]
        for ep, pt in agent_prompts.items():
            system_prompt.append({"text": f"***IMPORTANT {ep}: {pt}"})
        non_empty = {k: v for k, v in self.mongo_collection_info.items() if v}
        if non_empty:
            print(f"Emitting collection info for {len(non_empty)} endpoints to system prompt")
            system_prompt.append({"text": json.dumps(non_empty)})

        self._base_system_prompt = system_prompt
        self.llm_client.system = system_prompt
        self.mcp_tools_config = bedrock_tools
        self._tools_fetched_at = time.monotonic()
        self.llm_client.configure_tools(bedrock_tools, self._call_mcp_tool)
        emit_fn(
            f"Ready: {len(bedrock_tools)} tools from {len(self.mcp_endpoints)} endpoint(s)",
            status="Tools Ready",
        )

        # Persist discovery to MongoDB for future restarts.
        if self._tool_discovery_cache is not None and bedrock_tools:
            try:
                await self._tool_discovery_cache.set(_DISCOVERY_CACHE_KEY, {
                    "endpoints": self.mcp_endpoints,
                    "endpoint_configs": self.mcp_endpoint_configs,
                    "collection_info": self.mongo_collection_info,
                    "tools": bedrock_tools,
                    "system_prompt": system_prompt,
                }, ttl=getattr(settings, "CACHE_TTL", 300))
                logger.info("Tool discovery saved to MongoDB cache")
            except Exception as exc:
                logger.warning("Tool discovery cache write failed: %s", exc)

    async def _fetch_endpoint(self, name, root_fmt, emit=None):
        emit_fn = emit or self._emit
        config = {
            "url": root_fmt.format(name),
            "transport": "http",
            "headers": {"Authorization": f"Bearer {settings.get_auth_token()}"},
        }
        tools, agent_prompt, collection_info = [], "", {}
        try:
            resp = await asyncio.to_thread(
                requests.get, f"{settings.mongo_mcp_root}/{name}/llm_tools",
                headers=self._headers, timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
            agent_prompt = payload.get("agent_prompt", "") if isinstance(payload, dict) else ""
            tools = payload.get("tools", []) if isinstance(payload, dict) else []
            for tool in tools:
                if "toolSpec" in tool and "name" in tool["toolSpec"]:
                    tool["toolSpec"]["name"] = f"{name}_{tool['toolSpec']['name']}"
            emit_fn(f"Fetched {len(tools)} tools from {name}", status="Tools Discovered")
        except Exception as e:
            emit_fn(f"Error fetching tools for {name}: {e}", status="Error")
        try:
            if name in {"memory", "agent"}:
                collection_info = None
            else:
                resp = await asyncio.to_thread(
                    requests.get, f"{settings.mongo_mcp_root}/{name}/collection_info",
                    headers=self._headers, timeout=15,
                )
                resp.raise_for_status()
                payload = resp.json()
                collection_info = payload.get("collection_info", {}) if isinstance(payload, dict) else payload
        except Exception as e:
            emit_fn(f"Error fetching collection info for {name}: {e}", status="Error")
        return name, config, tools, collection_info, agent_prompt

    # ------------------------------------------------------------------
    # MCP tool dispatch
    # ------------------------------------------------------------------

    async def _call_mcp_tool(self, toolname, tool_input):
        endpoint_name, endpoint_tool_name = self._resolve_endpoint(toolname)
        cfg = self.mcp_endpoint_configs.get(endpoint_name)
        if cfg is None:
            raise RuntimeError(f"No config for endpoint '{endpoint_name}'")

        is_agent_tool = endpoint_name == "agent" or endpoint_tool_name in ("run_prompt",)
        is_memory_tool = ToolRouter._is_memory_tool(toolname)

        # Determine whether this call should be cached.
        # Memory tools are always fresh — they write/read user state and must never be stale.
        # Agent tools run sub-agent loops and are not idempotent.
        if is_agent_tool or is_memory_tool:
            should_cache = False
            cache_ttl = None
        elif endpoint_tool_name in self._CACHED_TOOL_SUFFIXES:
            should_cache = True
            cache_ttl = self._CACHED_TOOL_TTL
        elif self.enable_response_caching:
            should_cache = True
            cache_ttl = getattr(settings, "CACHE_TTL", 300)
        else:
            should_cache = False
            cache_ttl = None

        # Check cache before opening an MCP connection.
        cache_key = None
        if should_cache:
            cache_key = _create_cache_key(toolname, tool_input)
            self._tool_response_cache.reset_connection()
            cached = await self._tool_response_cache.get(cache_key)
            if cached is not None:
                self._emit(f"Cached: {toolname}", status="Tool Cache")
                return _annotate_cached(cached)

        # Agent tools run a full sub-agent loop; give them much more time.
        mcp_timeout = 600 if is_agent_tool else 60
        outer_timeout = 620 if is_agent_tool else 90

        # Always create a fresh client per call — reusing a single client across
        # concurrent async-with entries corrupts the connection state.
        client = fastmcp.Client({"mcpServers": {endpoint_name: cfg}}, timeout=mcp_timeout)

        async def _run():
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

        last_exc: Optional[Exception] = None
        # Agent tools are not retried — a second run_prompt call would spawn a
        # duplicate sub-agent.  All other tools retry up to 4 times.
        max_attempts = 1 if is_agent_tool else 4
        for attempt in range(max_attempts):
            try:
                result = await asyncio.wait_for(_run(), timeout=outer_timeout)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    wait = 1.0 * (attempt + 1)
                    logger.warning("MCP call to %r failed (attempt %d/%d): %s — retrying in %.1fs", toolname, attempt + 1, max_attempts, exc, wait)
                    await asyncio.sleep(wait)
        else:
            raise last_exc  # type: ignore[misc]

        # Store result in cache.
        if cache_key is not None:
            self._tool_response_cache.reset_connection()
            await self._tool_response_cache.set(cache_key, result, ttl=cache_ttl)

        return result

    def _resolve_endpoint(self, toolname):
        for candidate in sorted(self.mcp_endpoints, key=len, reverse=True):
            if toolname.startswith(f"{candidate}_"):
                return candidate, toolname[len(candidate) + 1:]
        if len(self.mcp_endpoints) == 1:
            return self.mcp_endpoints[0], toolname
        raise RuntimeError(
            f"Cannot resolve endpoint for '{toolname}'. Known: {self.mcp_endpoints}"
        )

    # ------------------------------------------------------------------
    # Query / history
    # ------------------------------------------------------------------

    MAX_CONTEXT_TOKENS = 200_000
    WARN_RATIO = 0.70          # emit warning above this fraction
    RESERVED_TOKENS = 20_000   # headroom for next response
    MAX_HISTORY_MSGS = 20      # hard cap on message count

    async def _prefetch_session_context(self, username: str) -> str:
        """Pre-fetch recent sessions and user preferences before the first LLM turn.

        Runs two parallel MCP tool calls and returns a formatted Markdown block
        to inject into the system prompt. Falls back gracefully on any error.
        """
        from datetime import datetime, timezone
        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sessions_text = ""
        prefs_text = ""
        try:
            sessions_res, prefs_res = await asyncio.gather(
                self._call_mcp_tool("memory_list_sessions", {"filter": {"username": username}, "limit": 5}),
                self._call_mcp_tool("memory_recall", {
                    "query": "preferences working style tools interests",
                    "username": username,
                    "memory_types": ["user_preference"],
                    "limit": 5,
                }),
                return_exceptions=True,
            )
            if not isinstance(sessions_res, Exception):
                try:
                    data = json.loads(sessions_res) if isinstance(sessions_res, str) else sessions_res
                    sessions = (data or {}).get("sessions", (data or {}).get("results", []))
                    if sessions:
                        lines = []
                        for s in sessions[:5]:
                            sid = s.get("session_id") or s.get("_id", "?")
                            summary = s.get("summary") or s.get("content") or ""
                            lines.append(f"- {sid}: {summary[:120]}")
                        sessions_text = "### Recent Sessions\n" + "\n".join(lines)
                except Exception:
                    pass
            if not isinstance(prefs_res, Exception):
                try:
                    data = json.loads(prefs_res) if isinstance(prefs_res, str) else prefs_res
                    prefs = (data or {}).get("results", [])
                    if prefs:
                        lines = [f"- {p.get('content', '')[:150]}" for p in prefs[:5]]
                        prefs_text = "### User Preferences\n" + "\n".join(lines)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Session pre-fetch error: %s", exc)
        header = (
            f"## Pre-loaded Session Context\n"
            f"**context_loaded:** true  \n"
            f"**username:** {username}  \n"
            f"**fetched_at:** {fetched_at}"
        )
        sections = [header]
        if sessions_text:
            sections.append(sessions_text)
        if prefs_text:
            sections.append(prefs_text)
        if len(sections) == 1:
            sections.append("*(no prior session data found)*")
        return "\n\n".join(sections)

    def query_with_mcp_tools(self, request: "QueryRequest", emit=None) -> "QueryResponse":
        emit_fn = emit or self._emit
        # Store session_id so sub-agents can inherit it without the LLM needing to pass it.
        self._current_session_id = request.session_id
        # History is owned entirely by the frontend — use only what was sent in the request.
        history = self._trim_history(list(request.history or []))

        is_new_session = not history

        msgs = list(history)
        user_content = [{"text": request.input}]
        self._inject_token_warning(msgs, user_content)
        msgs.append({"role": "user", "content": user_content})
        self.llm_client.message_handler = emit_fn

        # Re-discover tools if the cache is stale.
        age = (time.monotonic() - self._tools_fetched_at) if self._tools_fetched_at else float("inf")
        if age >= self.TOOLS_CACHE_TTL_SECONDS:
            logger.info("Tools cache is stale (%.0fs old) — refreshing before query.", age)
            self._discover_tools(emit=emit_fn)

        _ctx_block: Optional[str] = None
        _ctx_prefetched = False

        async def _invoke():
            nonlocal _ctx_block, _ctx_prefetched
            # Once per conversation: pre-fetch user context on the first message.
            prefetch_name = request.username or request.user_id
            if is_new_session and prefetch_name and not _ctx_prefetched:
                try:
                    _ctx_block = await self._prefetch_session_context(prefetch_name)
                    logger.info("Session context pre-fetched for user=%s (%d chars)", prefetch_name, len(_ctx_block))
                except Exception as exc:
                    logger.warning("Session pre-fetch failed: %s", exc)
                    _ctx_block = "## Pre-loaded Session Context\n**context_loaded:** false"
                _ctx_prefetched = True

            effective_system = list(self._base_system_prompt)
            if _ctx_block:
                effective_system.append({"text": _ctx_block})
            self.llm_client.system = effective_system
            self.llm_client.configure_tools(self.mcp_tools_config, self._call_mcp_tool)
            return await self.llm_client.invoke_bedrock_with_tools_text(messages=msgs)

        result = asyncio.run(_invoke())
        if self._looks_like_no_tools_error(result):
            emit_fn(
                "MCP tools were unavailable. Reloading backend tool configuration and retrying...",
                status="Recovering",
            )
            recovered = self._rediscover_tools(emit=emit_fn)
            if recovered:
                result = asyncio.run(_invoke())

        if self._looks_like_no_tools_error(result):
            # Keep this backend recovery detail out of the user-visible response.
            result = {
                **result,
                "error": "Tool configuration is reloading. Please retry your request in a few seconds.",
            }

        updated_history = result.get("history", msgs)
        answer = result.get("response_text", "")
        jsondata = result.get("jsondata")
        corrupt = bool(result.get("clear_history"))

        # Emit token usage summary after every response
        usage = result.get("usage") or {}
        in_tok = int(usage.get("inputTokens", 0) or 0)
        out_tok = int(usage.get("outputTokens", 0) or 0)
        total = in_tok + out_tok
        pct = round(total / self.MAX_CONTEXT_TOKENS * 100, 1)
        emit_fn(
            f"Tokens — in: {in_tok:,}  out: {out_tok:,}  total: {total:,}  ({pct}% of {self.MAX_CONTEXT_TOKENS // 1000}k)",
            status="Token Usage",
        )

        content = {"text": answer, "jsondata": jsondata}
        if corrupt:
            emit_fn("Corrupt tool state detected — cleaning history and retrying...", status="Recovering")
            sanitized = self._sanitize_history(list(updated_history or []))
            # Rebuild msgs with sanitized history + original user input and retry once.
            msgs = sanitized
            msgs.append({"role": "user", "content": user_content})
            result = asyncio.run(_invoke())
            updated_history = result.get("history", msgs)
            answer = result.get("response_text", "")
            jsondata = result.get("jsondata")
            content = {"text": answer, "jsondata": jsondata}
            if result.get("error"):
                return QueryResponse(
                    status="Error",
                    message=result["error"],
                    error=result["error"],
                    history=updated_history,
                )

        return QueryResponse(
            status="Query Completed", message="Completed",
            content=content,
            history=updated_history,
        )

    # ------------------------------------------------------------------
    # History trimming / sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_history(history: list) -> list:
        """Strip incomplete tool-use turns, keeping only clean text-only turns."""
        clean = []
        for msg in history:
            content = msg.get("content", [])
            if not isinstance(content, list):
                clean.append(msg)
                continue
            text_blocks = [
                b for b in content
                if isinstance(b, dict) and "text" in b
                and "toolUse" not in b and "toolResult" not in b
            ]
            if text_blocks:
                clean.append({"role": msg["role"], "content": text_blocks})
        return clean

    @staticmethod
    def _estimate_tokens(obj) -> int:
        if not obj:
            return 0
        if isinstance(obj, str):
            return max(1, len(obj) // 4)
        return max(1, len(json.dumps(obj, default=str)) // 4)

    def _estimate_history_tokens(self, history: list) -> int:
        total = 0
        for msg in history:
            total += 6
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    total += self._estimate_tokens(block)
            else:
                total += self._estimate_tokens(content)
        return total

    def _trim_history(self, history: list) -> list:
        """Drop oldest whole turns until under token budget and message cap."""
        if not history:
            return history
        token_limit = self.MAX_CONTEXT_TOKENS - self.RESERVED_TOKENS

        turns, current = [], []
        for msg in history:
            is_new_turn = (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
                and msg["content"]
                and not all(isinstance(b, dict) and "toolResult" in b for b in msg["content"])
            )
            if is_new_turn and current:
                turns.append(current)
                current = [msg]
            else:
                current.append(msg)
        if current:
            turns.append(current)

        def flatten(t):
            return [m for turn in t for m in turn]

        while len(flatten(turns)) > self.MAX_HISTORY_MSGS and len(turns) > 1:
            turns = turns[1:]

        while len(turns) > 1 and self._estimate_history_tokens(flatten(turns)) > token_limit:
            turns = turns[1:]

        result = flatten(turns)
        # Find the first clean user message (not a toolResult-only message)
        # to avoid starting history mid tool-use cycle which Bedrock rejects.
        for i, msg in enumerate(result):
            if msg.get("role") == "user":
                result = result[i:]
                break
        else:
            return []
        # Remove any orphaned toolResult messages whose toolUseId has no
        # matching toolUse block in the trimmed history.
        return self._repair_tool_cycle(result)

    @staticmethod
    def _repair_tool_cycle(history: list) -> list:
        """Remove orphaned toolResult messages that have no matching toolUse.

        When _trim_history drops the front of the conversation it can cut the
        assistant toolUse message while leaving the user toolResult response,
        producing a history[0] = user/toolResult that Bedrock rejects with
        'Expected toolResult blocks ... for the following Ids: <id>'.

        This pass collects all toolUseIds mentioned in assistant toolUse blocks,
        then strips any user message whose content is entirely toolResult blocks
        with no matching toolUseId in scope.
        """
        # Collect all toolUseIds present in assistant toolUse blocks.
        declared_ids: set = set()
        for msg in history:
            if msg.get("role") != "assistant":
                continue
            for block in msg.get("content") or []:
                if isinstance(block, dict) and "toolUse" in block:
                    tid = block["toolUse"].get("toolUseId")
                    if tid:
                        declared_ids.add(tid)

        repaired = []
        for msg in history:
            content = msg.get("content")
            if not isinstance(content, list):
                repaired.append(msg)
                continue
            # Check if this is a toolResult-only user message.
            tool_result_blocks = [
                b for b in content
                if isinstance(b, dict) and "toolResult" in b
            ]
            non_tool_blocks = [
                b for b in content
                if not (isinstance(b, dict) and "toolResult" in b)
            ]
            if tool_result_blocks and not non_tool_blocks:
                # Pure toolResult message — drop it if any ID is undeclared.
                result_ids = {
                    b["toolResult"].get("toolUseId")
                    for b in tool_result_blocks
                    if isinstance(b.get("toolResult"), dict)
                }
                if result_ids - declared_ids:
                    # At least one orphaned ID — drop the whole message.
                    continue
            repaired.append(msg)
        return repaired

    def _inject_token_warning(self, history: list, user_content: list) -> None:
        """Append a context-pressure warning when nearing token limit."""
        estimated = self._estimate_history_tokens(history)
        if estimated < self.MAX_CONTEXT_TOKENS * self.WARN_RATIO:
            return
        pct = int(estimated / self.MAX_CONTEXT_TOKENS * 100)
        user_content.append({"text": (
            f"\n\n[Context Warning] Conversation history is at ~{pct}% of the "
            f"{self.MAX_CONTEXT_TOKENS // 1000}k token limit. "
            "Please review the conversation so far and use the memory intake tool "
            "to save a concise summary of key facts, decisions, and any important data "
            "before it is lost to trimming. Keep your response concise. "
            "Older turns will be dropped on the next turn if the limit is not reduced."
        )})

    def reset(self) -> "QueryResponse":
        """Full reset: drain queue, drop cached MCP clients, re-discover tools."""
        try:
            while True:
                self._message_queue.get_nowait()
        except queue.Empty:
            pass
        self.endpoint_clients = {}
        # Clear MongoDB tool response cache.
        try:
            self._tool_response_cache.reset_connection()
            asyncio.run(self._tool_response_cache.clear())
        except Exception as exc:
            logger.warning("Failed to clear tool response cache: %s", exc)
        # Clear MongoDB discovery cache so next startup does a fresh discovery.
        if self._tool_discovery_cache is not None:
            try:
                self._tool_discovery_cache.reset_connection()
                asyncio.run(self._tool_discovery_cache.delete("mcp_tools_discovery_v1"))
            except Exception as exc:
                logger.warning("Failed to clear tool discovery cache: %s", exc)
        self._discover_tools()
        return QueryResponse(status="Reset", message="Application reset — tools reloaded", history=[], clear_history=True)

    def get_mcp_config(self) -> dict:
        return {
            "endpoints": self.mcp_endpoints or [],
            "collection_info": self.mongo_collection_info or {},
        }

    def save_pattern(self, user_id: str, session_id: str, history: Optional[List[Any]] = None) -> "QueryResponse":
        message = (
            f"The user marked this conversation as a useful pattern worth saving. "
            f"Please store the key question, approach, and answer from this session "
            f"(session_id: `{session_id}`, user_id: `{user_id}`) to memory as a "
            f"'pattern:query' memory_type entry in the semantic collection with "
            f"high importance (0.9). Include the user's original question and the "
            f"approach used to answer it so it can be reused in future sessions."
        )
        req = QueryRequest(input=message, history=history or [], user_id=user_id, session_id=session_id)
        return self.query_with_mcp_tools(req)

    def record_feedback(self, user_id: str, session_id: str, feedback: str, history: Optional[List[Any]] = None) -> "QueryResponse":
        if feedback == "positive":
            message = (
                f"\U0001f44d The user confirmed this approach **worked** for session `{session_id}` "
                f"(user: {user_id}). "
                f"Please review the conversation history and save the validated pattern to memory. "
                f"Use the memory intake tool twice:\n"
                f"1. A `pattern:query` entry (importance 0.9, semantic scope) containing: "
                f"the user's original question, the approach/strategy used to answer it, "
                f"and the key result — so it can be recalled and reused in future sessions.\n"
                f"2. A `feedback:positive` entry (importance 0.6) recording that session "
                f"`{session_id}` received positive feedback from user `{user_id}`."
            )
        else:
            message = (
                f"\U0001f44e The user indicated this approach **did not work** for session `{session_id}` "
                f"(user: {user_id}). "
                f"Please store a `feedback:negative` memory entry (importance 0.7) that records "
                f"what was tried, why it may have failed, and what to avoid in similar situations. "
                f"Include session_id `{session_id}` and user `{user_id}` in the entry."
            )
        req = QueryRequest(input=message, history=history or [], user_id=user_id, session_id=session_id)
        return self.query_with_mcp_tools(req)

    @staticmethod
    def _trim_history_for_ui(history, max_text_len=2000):
        if not history:
            return history
        trimmed = []
        for msg in history:
            content = msg.get("content")
            if not isinstance(content, list):
                trimmed.append(msg)
                continue
            needs_trim = any(
                isinstance(b, dict) and "toolResult" in b
                and any(
                    isinstance(c, dict) and len(c.get("text", "")) > max_text_len
                    for c in b["toolResult"].get("content", [])
                )
                for b in content
            )
            if not needs_trim:
                trimmed.append(msg)
                continue
            new_content = []
            for b in content:
                if isinstance(b, dict) and "toolResult" in b:
                    tr = b["toolResult"]
                    new_parts = [
                        {"text": c["text"][:max_text_len] + f"... [truncated {len(c['text'])} chars]"}
                        if isinstance(c, dict) and "text" in c and len(c["text"]) > max_text_len
                        else c
                        for c in tr.get("content", [])
                    ]
                    new_content.append({"toolResult": {**tr, "content": new_parts}})
                else:
                    new_content.append(b)
            trimmed.append({**msg, "content": new_content})
        return trimmed
