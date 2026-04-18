import copy
import json
import traceback
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError
import requests
import asyncio
import mcp.types as mt
from .webui_bedrock_client import WebUiBedrockClient
from .tool_router import ToolRouter

from ..mongo_cache import MongoSessionCache
from ..mongodb_client import MongoDBClient
from ..memory import build_memory_dispatch, get_memory_bedrock_toolspecs
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING) # don't log every API call
logger = logging.getLogger(__name__)

import fastmcp

class CachedQueryProcessor:
    """Enhanced QueryProcessor with comprehensive caching support
    
    Implements caching at multiple levels:
    1. Bedrock message caching with cache points
    2. MCP tool discovery caching
    3. MCP tool response caching
    4. Conversation history caching
    """

    _CACHE_VERSION = "v5"  # Bump when tool discovery cache schema changes to auto-invalidate stale entries

    def __init__(self, settings, message_handler: Optional[callable] = None):
        """Initializes the CachedQueryProcessor with caching configuration"""
        # Conversation history (starts empty)
        self.settings = settings
        self.history = None
        self.message_handler = message_handler or self._handle_message
        # Bedrock client used by both MCP server and web UI paths.
        self.llm_client = WebUiBedrockClient(settings)
        
        self.mcp_client = None
        self.endpoint_clients: Dict[str, fastmcp.Client] = {}
        self.mcp_endpoint_configs: Dict[str, Dict[str, Any]] = {}
        self.mcp_tools_config = None
        self.mcp_endpoints = None
        self.mongo_collection_info = {}
        self._agent_prompts: Dict[str, str] = {}
        self.tool_router: Optional[ToolRouter] = None
        self._system_prompt = []
        self._headers = {
            'Authorization': f'Bearer {settings.AUTH_TOKEN}',
            'Content-Type': 'application/json' 
        }    
            
        self._tool_discovery_cache = MongoSessionCache(
            username="default_user",
            session_id="default_session",
            cache_object_name="tool_discovery",
            settings=settings
        )
        self._tool_response_cache = MongoSessionCache(
            username="default_user",
            session_id="default_session",
            cache_object_name="tool_response",
            settings=settings
        )
        
        # Cache control flags
        self.enable_mcp_tool_caching = getattr(settings, "ENABLE_MCP_TOOL_CACHING", False)
        self.enable_response_caching = getattr(settings, "ENABLE_RESPONSE_CACHING", True)
        self.enable_ai_tool_routing = getattr(settings, "AI_TOOL_ROUTING", False)
        self.enable_tool_routing = getattr(settings, "TOOL_ROUTING", True)

        # Token budget controls for history management.
        self.max_context_tokens = int(getattr(settings, "LLM_MAX_CONTEXT_TOKENS", 200000))
        self.token_warning_ratio = float(getattr(settings, "LLM_TOKEN_WARNING_RATIO", 0.80))
        self.reserved_tokens = int(getattr(settings, "LLM_RESERVED_TOKENS", 20000))
        self.history_token_limit = max(1000, self.max_context_tokens - self.reserved_tokens)
        self.token_warning_threshold = int(self.max_context_tokens * self.token_warning_ratio)
        self._last_usage: Dict[str, int] = {}

        # Memory layer — always available regardless of MCP endpoint configuration.
        # Build a MongoDBClient and dispatch table for direct (non-HTTP) memory calls.
        self._memory_db_client = MongoDBClient(settings=settings)
        # llm_client is used for embedding; assigned after WebUiBedrockClient construction above.
        self._memory_fns: Dict[str, Any] = {}  # populated in generate_toolconfig after llm_client ready
        self._memory_toolspecs = get_memory_bedrock_toolspecs()

        self.generate_toolconfig()

    @staticmethod
    def _normalize_agent_prompt(value: Any) -> str:
        """Return cleaned agent prompt text, or empty string when not meaningful."""
        if not isinstance(value, str):
            return ""
        cleaned = value.strip()
        if cleaned.lower() in {"", "null", "none", "{}", "[]"}:
            return ""
        return cleaned

    def set_show_response_progress(self, show: bool):
        """Control whether to show LLM response progress updates"""
        self.llm_client.show_response_progress = show

    async def async_message_handler(self, message, status="Processing") -> None:
        return await asyncio.to_thread(self.message_handler, message, status)
    
    def _handle_message(self, message, status="Processing") -> None:
        """Handle incoming messages from the server."""
        if isinstance(message, Exception):
            print(f"Error in message handler: {message}")
            return    
        #print(message)
    
    def clear_all_caches(self):
        """Clear all caches - useful for testing or when data changes"""
        asyncio.run(self._tool_discovery_cache.clear())
        asyncio.run(self._tool_response_cache.clear())
        self.endpoint_clients = {}
        self.mcp_endpoint_configs = {}
        self.mcp_client = None
        self.mcp_tools_config = None
        self._agent_prompts = {}
        self.generate_toolconfig()
        self.message_handler("All caches cleared", status="Cache Cleared")

    async def _execute_mcp_tool_cached_async(self, tool_name: str, tool_input: dict) -> str:
        """Async MCP tool execution with cache support for BedrockClient tool callbacks."""

        # Memory tools are dispatched directly — no HTTP, no cache, always fresh.
        if ToolRouter._is_memory_tool(tool_name) and tool_name in self._memory_fns:
            try:
                result = await self._memory_fns[tool_name](**tool_input)
                return json.dumps(result, default=str)
            except Exception as exc:
                logger.error("Memory tool %s failed: %s", tool_name, exc)
                return json.dumps({"error": str(exc)})
        
        # Fast path: intercept and serve collection info from in-memory cache instead of calling MCP.
        # this cache was built from API calls during setup so we're avoiding heavy common calls 
        # to the MCP for collection info which is unlikely to change often and can be large.
        # Handles both prefixed ({endpoint}_get_collection_info) and unprefixed single-endpoint calls.
        suffix = "_get_collection_info"
        if tool_name.endswith(suffix):
            endpoint_key = tool_name[: -len(suffix)]
        elif tool_name == "get_collection_info" and self.mcp_endpoints and len(self.mcp_endpoints) == 1:
            endpoint_key = self.mcp_endpoints[0]
        else:
            endpoint_key = None

        if self.enable_mcp_tool_caching and endpoint_key is not None and endpoint_key in self.mongo_collection_info:
            self.message_handler(f"Using cached collection info for {endpoint_key}", status="Tool Cache")
            return json.dumps(self.mongo_collection_info.get(endpoint_key, {}), default=str)

        # back to the mcp calling layer for other tools and cacheable collection info calls that aren't in the in-memory cache.
        if not self.enable_response_caching:
            return await self._call_mcp_tool(tool_name, tool_input)
        
        # we have caching enabled, route through the cache layer with get_or_compute_async which handles cache hits, misses, and async compute.
        cache_key = MongoSessionCache.create_cache_key(tool_name, tool_input)
        result = await self._tool_response_cache.get_or_compute(
            cache_key,
            compute=lambda: self._call_mcp_tool(tool_name, tool_input),
            on_cache_hit=lambda: self.message_handler(f"Using cached response for {tool_name}", status="Tool Cache"),
        )
        self.message_handler(f"Response for {tool_name}", status="LLM Reasoning...")
        return result

    def generate_toolconfig(self) -> list:
        """Discover and configure Bedrock tools from the MCP server HTTP APIs. Idempotent — no-op if already configured."""
        if not self.mcp_tools_config:
            asyncio.run(self._setup_tools_async())
        return self.mcp_tools_config

    async def _setup_tools_async(self) -> None:
        """Fetch endpoints, Bedrock-formatted tools, and collection info from the MCP HTTP APIs.

        Calls /tools_config to list endpoints, then /{endpoint}/llm_tools and
        /{endpoint}/collection_info for each. No MCP session is needed for discovery —
        the server-side annotation pipeline handles all formatting.
        Results are written to cache; on a cache hit, _apply_cached_state() restores everything.
        """
        self._tool_discovery_cache.reset_connection()
        cache_key = "mcp_tools_discovery"

        if self.enable_mcp_tool_caching:
            cached = await self._tool_discovery_cache.get(cache_key)
            if cached is not None:
                if cached.get("cache_version") == self._CACHE_VERSION:
                    self.message_handler("Using cached MCP tools discovery")
                    self._apply_cached_state(cached)
                    return
                self.message_handler("Stale tool discovery cache (version mismatch), re-fetching", status="Discovering Tools")
                await self._tool_discovery_cache.clear()

        # Resolve available endpoint names from the server
        try:
            response = requests.get(f"{self.settings.mongo_mcp_root}/tools_config", headers=self._headers)
            response.raise_for_status()
            self.mcp_endpoints = response.json().get("available_tools", [])
            self.message_handler(f"Discovered endpoints: {self.mcp_endpoints}", status="Discovering Tools")
        except requests.RequestException as e:
            self.message_handler(f"Error fetching endpoint list: {e}", status="Error")
            self.mcp_endpoints = []

        bedrock_tools = []
        root_frmt = f"{self.settings.mongo_mcp_root}/{{}}/mcp"

        results = await asyncio.gather(*[
            self._fetch_endpoint_data(name, root_frmt) for name in self.mcp_endpoints
        ])
        for name, config, tools, collection_info, agent_prompt in results:
            self.mcp_endpoint_configs[name] = config
            bedrock_tools.extend(tools)
            self.mongo_collection_info[name] = collection_info
            if agent_prompt:
                self._agent_prompts[name] = self._normalize_agent_prompt(agent_prompt)

        if self.mcp_endpoint_configs:
            self.mcp_client = fastmcp.Client({"mcpServers": self.mcp_endpoint_configs})

        self._configure_llm_client(bedrock_tools)
        self.message_handler(f"Using {len(bedrock_tools)} tools from {len(self.mcp_endpoints)} endpoint(s)", status="Tools Ready")

        if self.enable_mcp_tool_caching:
            await self._tool_discovery_cache.set(cache_key, {
                "cache_version": self._CACHE_VERSION,
                "endpoints": self.mcp_endpoints,
                "endpoint_configs": self.mcp_endpoint_configs,
                "collection_info": self.mongo_collection_info,
                "agent_prompts": self._agent_prompts,
                "tools": bedrock_tools,
            })

    async def _fetch_endpoint_data(self, name: str, root_frmt: str) -> tuple:
        """Fetch llm_tools and collection_info for a single endpoint. Runs concurrently via asyncio.gather."""
        config = {
            "url": root_frmt.format(name),
            "transport": "http",
            "headers": {"Authorization": f"Bearer {self.settings.AUTH_TOKEN}"}
        }

        tools = []
        agent_prompt = ""
        try:
            # spin up a thread for each endpoint call since requests is blocking and we want concurrency here, 
            # especially if there are many endpoints or slow responses. FastMCP sessions require async context and was slow 
            # so I pulled it out to an API call with requests instead of using the session tool discovery
            resp = await asyncio.to_thread(
                requests.get, f"{self.settings.mongo_mcp_root}/{name}/llm_tools", headers=self._headers
            )
            resp.raise_for_status()
            payload = resp.json()
            agent_prompt = payload.get("agent_prompt", "") if isinstance(payload, dict) else ""
            tools = payload.get("tools", []) if isinstance(payload, dict) else []
            for tool in tools:
                if "toolSpec" in tool and "name" in tool["toolSpec"]:
                    tool["toolSpec"]["name"] = f"{name}_{tool['toolSpec']['name']}"
            self.message_handler(f"Fetched {len(tools)} tools from {name}", status="Tools Discovered")
        except Exception as e:
            self.message_handler(f"Error fetching tools for {name}: {e}", status="Error")

        collection_info = {}
        try:
            resp = await asyncio.to_thread(
                requests.get, f"{self.settings.mongo_mcp_root}/{name}/collection_info", headers=self._headers
            )
            resp.raise_for_status()
            payload = resp.json()
            # Support both API shapes:
            # 1) {"collection_info": ...}
            # 2) direct list/dict payload
            collection_info = payload.get("collection_info", {}) if isinstance(payload, dict) else payload
        except Exception as e:
            self.message_handler(f"Error fetching collection info for {name}: {e}", status="Error")

        return name, config, tools, collection_info, agent_prompt

    def _apply_cached_state(self, cached: dict) -> None:
        """Restore full instance state — endpoints, tools, and LLM client — from a cached discovery payload."""
        self.mcp_endpoints = cached.get("endpoints", [])
        self.mcp_endpoint_configs = cached.get("endpoint_configs", {})
        self.mongo_collection_info = cached.get("collection_info", {})
        self._agent_prompts = cached.get("agent_prompts", {})
        self.mcp_client = fastmcp.Client({"mcpServers": self.mcp_endpoint_configs}) if self.mcp_endpoint_configs else None
        self._configure_llm_client(cached.get("tools", []))

    def _configure_llm_client(self, bedrock_tools: list) -> None:
        """Set the system prompt, register tools on the LLM client, and initialize the ToolRouter."""
        # Build memory dispatch table now that llm_client is ready (needs embedding capability).
        if not self._memory_fns:
            try:
                self._memory_fns = build_memory_dispatch(
                    db_client=self._memory_db_client,
                    llm_client=self.llm_client,
                    settings=self.settings,
                )
                logger.info("Memory dispatch table built (%d tools)", len(self._memory_fns))
            except Exception as exc:
                logger.warning("Failed to build memory dispatch table: %s", exc)

        # Always include memory toolspecs — deduplicate by name so we don't double-add
        # if the memory endpoint also appears in the HTTP-discovered catalog.
        existing_names = {t["toolSpec"]["name"] for t in bedrock_tools if "toolSpec" in t}
        memory_additions = [
            spec for spec in self._memory_toolspecs
            if spec["toolSpec"]["name"] not in existing_names
        ]
        all_tools = bedrock_tools + memory_additions
        if memory_additions:
            logger.info("Injected %d memory toolspecs into tool catalog", len(memory_additions))

        self._system_prompt = [
            {"text": t}
            for t in getattr(self.settings, "BEDROCK_SYSTEM_PROMPT_TEXTS", [])
        ]
        print("System prompt:", self._system_prompt)
        for endpoint_name, agent_prompt in self._agent_prompts.items():
            cleaned = self._normalize_agent_prompt(agent_prompt)
            if cleaned:
                self._system_prompt.append({"text": f"***IMPORTANT {endpoint_name}:{cleaned}"})

        # Avoid injecting empty per-endpoint collection info blocks like {"endpoint": {}}.
        non_empty_collection_info = {}
        for endpoint_name, info in (self.mongo_collection_info or {}).items():
            if info is None:
                continue
            if isinstance(info, (dict, list, str, tuple, set)) and len(info) == 0:
                continue
            non_empty_collection_info[endpoint_name] = info

        if non_empty_collection_info:
            self._system_prompt.append({"text": json.dumps(non_empty_collection_info)})
        self.llm_client.system = self._system_prompt
        self.mcp_tools_config = all_tools
        self.llm_client.configure_tools(self.mcp_tools_config, self._execute_mcp_tool_cached_async)
        self.tool_router = ToolRouter(
            tool_catalog=all_tools,
            llm_client=self.llm_client,
            message_handler=self.message_handler,
            settings=self.settings,
            memory_fns=self._memory_fns,
        )


    async def _call_mcp_tool(self, toolname: str, tool_input: dict) -> str:
        """Initialize a stateless session for tool calls."""
        await self.async_message_handler(
            f"Calling MCP tool {toolname} with input: {tool_input}",
            status="Tool Execution"
        )
        try:
            endpoint_name = None
            endpoint_tool_name = toolname
            endpoints = self.mcp_endpoints or []
            # we need to split the toolname to get the endpoint server to call 
            # the toolname is expected to be in the format {endpoint}_{tool} to allow for multiple endpoints 
            # with overlapping tool names, but we have to use a signle session object instead of the class
            # level client otherwise it sends the tool calls to every endpoint. this may be a bug with fastmcp.
            # Match the longest endpoint prefix first for deterministic routing.
            for candidate in sorted(endpoints, key=len, reverse=True):
                prefix = f"{candidate}_"
                if toolname.startswith(prefix):
                    endpoint_name = candidate
                    endpoint_tool_name = toolname[len(prefix):]
                    break

            # Compatibility path for single-endpoint mode with unprefixed tool names.
            if endpoint_name is None and self.mcp_endpoints and len(self.mcp_endpoints) == 1:
                endpoint_name = self.mcp_endpoints[0]
                endpoint_tool_name = toolname

            if endpoint_name is None:
                raise RuntimeError(
                    f"Unable to resolve endpoint for tool '{toolname}'. Known endpoints: {self.mcp_endpoints}"
                )
            if not endpoint_tool_name:
                raise RuntimeError(
                    f"Resolved empty tool name for endpoint '{endpoint_name}' from '{toolname}'"
                )

            endpoint_client = self.endpoint_clients.get(endpoint_name)
            if endpoint_client is None:
                endpoint_config = self.mcp_endpoint_configs.get(endpoint_name)
                if endpoint_config is None:
                    raise RuntimeError(f"No endpoint config found for '{endpoint_name}'")
                endpoint_client = fastmcp.Client({"mcpServers": {endpoint_name: endpoint_config}})
                self.endpoint_clients[endpoint_name] = endpoint_client

            # Wrap the entire session open → request → session close in a timeout.
            # The session close can hang indefinitely (fastmcp bug), so we need to
            # time out the whole block, not just the request.
            async def _run_in_session():
                async with endpoint_client:
                    return await endpoint_client.session.send_request(
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

            try:
                result = await asyncio.wait_for(_run_in_session(), timeout=90)
            except asyncio.TimeoutError:
                await self.async_message_handler(
                    f"MCP tool {toolname} timed out (session open/close included)", status="Error"
                )
                raise RuntimeError(f"MCP tool call {toolname} timed out after 90s")

            if result.content and hasattr(result.content[0], "text"):
                return result.content[0].text
            if result.structuredContent is not None:
                return json.dumps(result.structuredContent)
            return str(result)
        except Exception as e:
            await self.async_message_handler(f"Failed MCP {toolname} call: {e}", status="Error")
            traceback.print_exc()
            raise
 
    def _trim_history(self, history: list) -> list:
        """Trim history by message count and token budget.

        Keeps conversation start aligned to a user message so Bedrock
        ordering rules are satisfied.
        """
        max_msgs = getattr(self.settings, 'LLM_MAX_HISTORY', 20)
        turns = self._history_to_turns(history)

        if not turns:
            return []

        # First respect max message count, but only by removing whole turns.
        while len(self._flatten_turns(turns)) > max_msgs and len(turns) > 1:
            turns = turns[1:]

        trimmed = self._flatten_turns(turns)
        trimmed = self._align_history_to_user(trimmed)

        # Token-aware trimming: remove oldest whole turns until under budget.
        while len(turns) > 1 and self._estimate_history_tokens(trimmed) > self.history_token_limit:
            turns = turns[1:]
            trimmed = self._align_history_to_user(self._flatten_turns(turns))

        # Final sanity pass: if the suffix is still structurally invalid, drop oldest
        # turns until all toolUse/toolResult pairs are balanced.
        while len(turns) > 1 and not self._history_has_balanced_tool_calls(trimmed):
            turns = turns[1:]
            trimmed = self._align_history_to_user(self._flatten_turns(turns))

        if not self._history_has_balanced_tool_calls(trimmed):
            logger.warning("History remained structurally invalid after trimming; clearing retained history.")
            return []

        return trimmed

    @staticmethod
    def _is_tool_result_message(msg: dict) -> bool:
        content = msg.get("content", [])
        return bool(
            msg.get("role") == "user"
            and isinstance(content, list)
            and content
            and all(isinstance(block, dict) and "toolResult" in block for block in content)
        )

    def _history_to_turns(self, history: list) -> list:
        """Group history into user-initiated turns so trimming preserves tool cycles."""
        turns = []
        current_turn = []

        for msg in history:
            starts_new_turn = (
                msg.get("role") == "user"
                and not self._is_tool_result_message(msg)
            )
            if starts_new_turn and current_turn:
                turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)

        if current_turn:
            turns.append(current_turn)

        return turns

    @staticmethod
    def _flatten_turns(turns: list) -> list:
        flattened = []
        for turn in turns:
            flattened.extend(turn)
        return flattened

    @staticmethod
    def _align_history_to_user(history: list) -> list:
        for i, msg in enumerate(history):
            if msg.get("role") == "user":
                return history[i:]
        return history

    @staticmethod
    def _estimate_text_tokens(text: Any) -> int:
        if not text:
            return 0
        if not isinstance(text, str):
            text = str(text)
        # Rough heuristic for Bedrock token accounting.
        return max(1, len(text) // 4)

    def _estimate_history_tokens(self, history: list) -> int:
        total = 0
        for msg in history:
            total += 6  # per-message structural overhead
            content = msg.get("content", [])
            if not isinstance(content, list):
                total += self._estimate_text_tokens(content)
                continue
            for block in content:
                if not isinstance(block, dict):
                    total += self._estimate_text_tokens(block)
                    continue
                if "text" in block:
                    total += self._estimate_text_tokens(block.get("text", ""))
                elif "toolUse" in block:
                    total += self._estimate_text_tokens(json.dumps(block.get("toolUse", {}), default=str))
                elif "toolResult" in block:
                    total += self._estimate_text_tokens(json.dumps(block.get("toolResult", {}), default=str))
                else:
                    total += self._estimate_text_tokens(json.dumps(block, default=str))
        return total

    def _history_has_balanced_tool_calls(self, history: list) -> bool:
        """Return True when every retained toolUse has a matching retained toolResult."""
        pending_tool_use_ids = []

        for msg in history:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                tool_use = block.get("toolUse")
                if isinstance(tool_use, dict):
                    tool_use_id = tool_use.get("toolUseId")
                    if tool_use_id:
                        pending_tool_use_ids.append(tool_use_id)
                tool_result = block.get("toolResult")
                if isinstance(tool_result, dict):
                    tool_use_id = tool_result.get("toolUseId")
                    if tool_use_id in pending_tool_use_ids:
                        pending_tool_use_ids.remove(tool_use_id)

        return not pending_tool_use_ids

    def _maybe_inject_token_warning(self, user_block: list, question: str) -> None:
        """Inject a user-context warning when history is nearing token limits."""
        history_tokens = self._estimate_history_tokens(self.history or [])
        latest_input_tokens = int((self._last_usage or {}).get("inputTokens", 0) or 0)
        pressure_tokens = max(history_tokens + self._estimate_text_tokens(question), latest_input_tokens)
        if pressure_tokens < self.token_warning_threshold:
            return

        ratio = int((pressure_tokens / max(1, self.max_context_tokens)) * 100)
        warning_text = (
            "[Context Warning] Conversation context is nearing token capacity "
            f"(~{ratio}% of {self.max_context_tokens}). "
            "Before continuing, briefly summarize durable memory and reflect on memeory management strategies."
            "because older history may be trimmed soon."
        )
        user_block.append({"text": f"\n\n{warning_text}"})

    def query_with_mcp_tools(self, question: str, history: Optional[list] = None, user_id: Optional[str] = None, session_id: Optional[str] = None, use_llm_routing: bool = False) -> tuple:
        """
        Query LLM with MCP tool support using Bedrock's Converse API with caching.
        This flow is very complex because there are a lot of json formatting paths
        and we want to preserve the ability to cache at multiple levels (tool discovery, tool responses) 
        without accidentally caching errors or stale data. 
        There are a number of competing concerns to balance:
        - Providing polymorphic support for json formats in tool inputs and outputs. We have 2 now, unknown future
        - Caching tool discovery results to avoid redundant API calls, but ensuring cache invalidation on schema changes
        - Caching tool responses to speed up repeated calls, but ensuring errors aren't cached and that the cache is bypassed when disabled
        - Preserving conversation history across calls while allowing it to be cleared when needed

        Flow:
          1. Prepare history and append question
          2. Discover/cache MCP tools (generate_toolconfig)
          3. Invoke Bedrock via Converse API, keeping MCP session open for tool callbacks
             → _invoke (inline coroutine, resets Motor connections, opens MCP session if present)
               → llm_client.invoke_bedrock_with_tools_text  (WebUiBedrockClient)
                 → WebUiBedrockClient.invoke_bedrock_with_tools  (formats request)
                   → BedrockClient.invoke_bedrock_with_tools     (base class, actual API call)
               → normalize_bedrock_response (WebUiBedrockClient, splits text / jsondata)
          4. Return (answer, jsondata, history)

        Returns:
            tuple: (answer: str, jsondata: dict|None, history: list, clear_history: bool)
                - answer: LLM response text
                - jsondata: structured JSON data extracted from the response, or None
                - history: updated conversation history
                - clear_history: True if the conversation history was corrupt and has been cleared;
                  callers should discard any client-side history and prompt the user to retry
        """
        if history:
            self.history = history
        if self.history is None:
            self.history = []

        # Scope the tool-response cache to this user so each browser user gets
        # independent cached results.  Fallback to "default_user" when no user_id
        # is provided (e.g. non-browser callers).
        if user_id:
            self._tool_response_cache.username = user_id
            self._tool_response_cache.session_id = "tool_response"

        self.history = self._trim_history(self.history)
        self.generate_toolconfig()

        messages = self.history
        messages.append({"role": "user", "content": [{"text": question}]})
        self._maybe_inject_token_warning(messages[-1]["content"], question)
        self.llm_client.message_handler = self.message_handler

        async def _invoke(msgs):
            tools_for_question, hint_text = [], None
            was_cache_hit = False
            llm_routing_ran = False  # tracks whether LLM routing made an explicit decision

            if not self.enable_tool_routing:
                tools_for_question = self.mcp_tools_config or []
                self.llm_client.configure_tools(tools_for_question, self._execute_mcp_tool_cached_async)
                logger.info(f"[TOOL SELECTION] method=full_catalog_tool_routing_disabled tools={[t['toolSpec']['name'] for t in tools_for_question]}")
            else:
                # Phase 1: pattern cache lookup — skipped when caller requests direct LLM routing
                if self.tool_router is not None and not use_llm_routing:
                    tools_for_question, hint_text = await self.tool_router.try_pattern_match(question)
                    was_cache_hit = bool(tools_for_question)
                    if was_cache_hit:
                        logger.info(f"[TOOL SELECTION] method=pattern_cache tools={[t['toolSpec']['name'] for t in tools_for_question]}")
                    else:
                        logger.info("[TOOL SELECTION] method=pattern_cache result=MISS — no matching pattern found")
                # Phase 2: LLM routing — on cache miss, or when server flag / caller forces it
                if not tools_for_question and self.tool_router is not None and (self.enable_ai_tool_routing or use_llm_routing):
                    try:
                        tools_for_question, hint_text = await self.tool_router.route_via_llm(question)
                        llm_routing_ran = True
                        if tools_for_question:
                            method = "llm_routing_forced" if use_llm_routing else "llm_routing_cache_miss"
                            logger.info(f"[TOOL SELECTION] method={method} tools={[t['toolSpec']['name'] for t in tools_for_question]}")
                        else:
                            logger.info("[TOOL SELECTION] method=llm_routing result=no tools — LLM will answer directly without tools")
                    except Exception as e:
                        traceback.print_exc()
                        logger.warning(f"Tool routing failed, falling back to full tool set: {e}")
                if tools_for_question:
                    self.llm_client.configure_tools(tools_for_question, self._execute_mcp_tool_cached_async)
                elif llm_routing_ran:
                    # LLM routing explicitly decided no tools are needed — honour that decision
                    self.llm_client.configure_tools([], None)
                    logger.info("[TOOL SELECTION] method=none — respecting LLM routing decision to answer directly")
                else:
                    logger.info(f"[TOOL SELECTION] method=full_catalog tools={[t['toolSpec']['name'] for t in (self.mcp_tools_config or [])]}")
                if hint_text:
                    # Append the playbook / output-format hint to the user message
                    user_block = msgs[-1]["content"]
                    user_block.append({"text": f"\n\n{hint_text}"})
                    logger.info("Injected playbook hint into user message")

            self.llm_client.system = self._system_prompt

            # Snapshot msgs before the LLM loop mutates it (appends assistant/tool messages).
            # Needed for a potential retry with the full tool catalog.
            msgs_snapshot = copy.deepcopy(msgs)

            # Reset Motor connections so they bind to this event loop.
            self._tool_response_cache.reset_connection()
            # Tool calls use per-endpoint clients (self.endpoint_clients) that
            # open/close their own sessions on each call, so we do NOT need the
            # top-level self.mcp_client context manager here.  Opening it would
            # create sessions to every endpoint and the close can hang.
            result = await self.llm_client.invoke_bedrock_with_tools_text(messages=msgs)

            # If the pattern cache routed to the wrong tool and all tool results
            # came back empty, retry using LLM routing so it can pick the right tool.
            if was_cache_hit and self._tool_results_all_empty(result.get("history", [])):
                self.message_handler(
                    "Pattern cache hit yielded no data — retrying with LLM tool routing",
                    status="Tool Routing",
                )
                logger.warning(
                    "Cache-routed query returned empty tool results; "
                    "falling back to LLM routing (cache pattern may be wrong)"
                )
                msgs.clear()
                msgs.extend(msgs_snapshot)
                # Use LLM routing to select the right tools for this retry
                retry_tools, retry_hint = [], None
                if self.tool_router is not None:
                    try:
                        retry_tools, retry_hint = await self.tool_router.route_via_llm(question)
                    except Exception as e:
                        logger.warning(f"LLM routing on retry failed, using full catalog: {e}")
                if retry_tools:
                    self.llm_client.configure_tools(retry_tools, self._execute_mcp_tool_cached_async)
                    if retry_hint:
                        msgs[-1]["content"].append({"text": f"\n\n{retry_hint}"})
                else:
                    self.llm_client.configure_tools(
                        self.mcp_tools_config, self._execute_mcp_tool_cached_async
                    )
                self._tool_response_cache.reset_connection()
                result = await self.llm_client.invoke_bedrock_with_tools_text(messages=msgs)

            # Record a PII-free playbook for LLM-routed interactions (non-fatal).
            # Only fires when LLM routing set a pattern; cache hits already have a playbook.
            if self.tool_router is not None and self.tool_router._last_pattern and llm_routing_ran:
                try:
                    await self.tool_router.record_pattern(
                        history=result.get("history", []),
                        response_text=result.get("response_text", ""),
                        jsondata=result.get("jsondata"),
                        question=question,
                    )
                except Exception as _log_err:
                    logger.warning("Pattern recording failed: %s", _log_err)

            return result

        try:
            invoke_result = asyncio.run(_invoke(messages))
            self.message_handler("Response ready, sending to UI...", status="Finalizing")
        except ClientError as error:
            error_code = error.response['Error']['Code']
            print(f"Bedrock error: {error_code} - {error.response['Error']['Message']}")
            if error_code == 'ValidationException':
                return "Error: Input validation failed", None, self.history, False
            if error_code in ['ExpiredTokenException', 'ExpiredToken']:
                raise
            return f"Error: {error.response['Error']['Message']}", None, self.history, False
        except Exception as e:
            print(f"Unexpected error invoking Bedrock: {e}")
            return f"Error: {str(e)}", None, self.history, False

        # If Bedrock detected corrupt history (missing toolResult), clear it before returning.
        if invoke_result.get("clear_history"):
            self.history = []
            logger.warning("History cleared due to corrupt toolResult state.")
            return invoke_result.get("error", "Conversation history was corrupt and has been cleared. Please try again."), None, [], True

        self.history = invoke_result.get("history", messages)
        self.history = self._trim_history(self.history)
        answer = invoke_result.get("response_text", "No response generated")
        jsondata = invoke_result.get("jsondata", None)
        usage = invoke_result.get("usage_last") or invoke_result.get("usage") or {}
        self._last_usage = usage if isinstance(usage, dict) else {}

        if self._last_usage:
            in_tokens = int(self._last_usage.get("inputTokens", 0) or 0)
            out_tokens = int(self._last_usage.get("outputTokens", 0) or 0)
            total_tokens = in_tokens + out_tokens
            # Report context pressure from trimmed history so % aligns with trimming behavior.
            history_tokens = self._estimate_history_tokens(self.history or [])
            context_pressure_tokens = max(history_tokens, in_tokens)
            percent_used = (context_pressure_tokens / max(1, self.max_context_tokens)) * 100
            self.message_handler(
                (
                    f"Token usage - input: {in_tokens}, output: {out_tokens}, total: {total_tokens}, "
                    f"context_estimate: {context_pressure_tokens}, used: {percent_used:.1f}%"
                ),
                status="Token Usage",
            )

        # Stash last query state so save_pattern() can be triggered manually.
        self._last_question = question
        self._last_answer = answer
        self._last_jsondata = jsondata

        return answer, jsondata, self.history, False

    @staticmethod
    def _tool_results_all_empty(history: list) -> bool:
        """Return True if at least one tool was called and every result was trivially empty.

        Triggers the full-catalog retry when a cache-hit route selected the wrong tool.
        Returns False (no retry) when no tool calls were made or any result has real data.
        """
        tool_results = []
        for msg in history:
            if msg.get("role") != "user":
                continue
            for block in msg.get("content", []):
                if "toolResult" in block:
                    tool_results.append(block["toolResult"])

        if not tool_results:
            return False  # No tool calls at all — LLM answered directly; don't retry

        _EMPTY_LITERALS = {"[]", "{}", "null", "none", "\"[]\"", "\"{}\"", "[]\n", "{}\n"}
        for tr in tool_results:
            if tr.get("status") == "error":
                continue  # errors count as non-empty for retry purposes
            for item in tr.get("content", []):
                text = item.get("text", "").strip()
                if len(text) > 20 and text.lower() not in _EMPTY_LITERALS:
                    return False  # At least one result has real data
        return True  # All tool results were trivially empty

    def save_pattern(self) -> bool:
        """Persist the routing pattern from the last query into the pattern cache.

        Should be called explicitly (e.g. via a UI button) after the user
        confirms the output is correct.  Returns True if a pattern was saved.
        """
        if not self.enable_ai_tool_routing or self.tool_router is None:
            return False
        if not getattr(self, '_last_question', None) or not getattr(self, '_last_answer', None):
            return False
        if self._last_answer.startswith("Error"):
            return False
        try:
            asyncio.run(
                self.tool_router.record_pattern(
                    list(self.history),
                    self._last_answer,
                    jsondata=self._last_jsondata,
                    question=self._last_question,
                )
            )
            return True
        except Exception as exc:
            logger.warning(f"Pattern save failed: {exc}")
            return False
            return False

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring"""
        return {
            "caching_enabled": {
                "bedrock": self.llm_client.enable_cache_points,
                "mcp_tools": self.enable_mcp_tool_caching,
                "responses": self.enable_response_caching
            },
            "endpoints_configured": len(self.mcp_endpoint_configs),
            "tools_configured": len(self.mcp_tools_config) if self.mcp_tools_config else 0,
        }
