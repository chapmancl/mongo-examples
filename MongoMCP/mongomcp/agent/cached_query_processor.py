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

import fastmcp

class CachedQueryProcessor:
    """Enhanced QueryProcessor with comprehensive caching support
    
    Implements caching at multiple levels:
    1. Bedrock message caching with cache points
    2. MCP tool discovery caching
    3. MCP tool response caching
    4. Conversation history caching
    """

    _CACHE_VERSION = "v3"  # Bump when tool discovery cache schema changes to auto-invalidate stale entries

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
        self.enable_mcp_tool_caching = getattr(settings, "ENABLE_MCP_TOOL_CACHING", True)
        self.enable_response_caching = getattr(settings, "ENABLE_RESPONSE_CACHING", True)

        self.generate_toolconfig()
    
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
        self.generate_toolconfig()
        self.message_handler("All caches cleared", status="Cache Cleared")

    async def _execute_mcp_tool_cached_async(self, tool_name: str, tool_input: dict) -> str:
        """Async MCP tool execution with cache support for BedrockClient tool callbacks."""
        
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

        if endpoint_key is not None and endpoint_key in self.mongo_collection_info:
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
        for name, config, tools, collection_info in results:
            self.mcp_endpoint_configs[name] = config
            bedrock_tools.extend(tools)
            self.mongo_collection_info[name] = collection_info

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
        try:
            # spin up a thread for each endpoint call since requests is blocking and we want concurrency here, 
            # especially if there are many endpoints or slow responses. FastMCP sessions require async context and was slow 
            # so I pulled it out to an API call with requests instead of using the session tool discovery
            resp = await asyncio.to_thread(
                requests.get, f"{self.settings.mongo_mcp_root}/{name}/llm_tools", headers=self._headers
            )
            resp.raise_for_status()
            tools = resp.json().get("tools", [])
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
            collection_info = resp.json().get("collection_info", {})
        except Exception as e:
            self.message_handler(f"Error fetching collection info for {name}: {e}", status="Error")

        return name, config, tools, collection_info

    def _apply_cached_state(self, cached: dict) -> None:
        """Restore full instance state — endpoints, tools, and LLM client — from a cached discovery payload."""
        self.mcp_endpoints = cached.get("endpoints", [])
        self.mcp_endpoint_configs = cached.get("endpoint_configs", {})
        self.mongo_collection_info = cached.get("collection_info", {})
        self.mcp_client = fastmcp.Client({"mcpServers": self.mcp_endpoint_configs}) if self.mcp_endpoint_configs else None
        self._configure_llm_client(cached.get("tools", []))

    def _configure_llm_client(self, bedrock_tools: list) -> None:
        """Set the system prompt, register tools on the LLM client, and initialize the ToolRouter."""
        self._system_prompt = [
            {"text": t}
            for t in getattr(self.settings, "BEDROCK_SYSTEM_PROMPT_TEXTS", [])
        ]
        self._system_prompt.append({"text": json.dumps(self.mongo_collection_info)})
        self.llm_client.system = self._system_prompt
        self.mcp_tools_config = bedrock_tools
        self.llm_client.configure_tools(self.mcp_tools_config, self._execute_mcp_tool_cached_async)
        self.tool_router = ToolRouter(
            tool_catalog=bedrock_tools,
            llm_client=self.llm_client,
            message_handler=self.message_handler,
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

            async with endpoint_client:
                result = await endpoint_client.session.send_request(
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
        """Keep only the most recent LLM_MAX_HISTORY messages.
        Always trims to a user-role message at the start so Bedrock
        doesn't reject the conversation.
        """
        max_msgs = getattr(self.settings, 'LLM_MAX_HISTORY', 20)
        if len(history) <= max_msgs:
            return history
        trimmed = history[-max_msgs:]
        # Advance to first user message to satisfy Converse API ordering rules
        for i, msg in enumerate(trimmed):
            if msg.get("role") == "user":
                return trimmed[i:]
        return trimmed

    def query_with_mcp_tools(self, question: str, history: Optional[list] = None) -> tuple:
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
            tuple: (answer: str, jsondata: dict|None, history: list)
        """
        if history:
            self.history = history
        if self.history is None:
            self.history = []

        self.history = self._trim_history(self.history)
        self.generate_toolconfig()

        messages = self.history
        messages.append({"role": "user", "content": [{"text": question}]})
        self.llm_client.message_handler = self.message_handler

        async def _invoke(msgs):
            # Reset Motor connections so they bind to this event loop.
            self._tool_response_cache.reset_connection()
            if self.mcp_client is not None:
                async with self.mcp_client:
                    return await self.llm_client.invoke_bedrock_with_tools_text(messages=msgs)
            return await self.llm_client.invoke_bedrock_with_tools_text(messages=msgs)

        try:
            invoke_result = asyncio.run(_invoke(messages))
        except ClientError as error:
            error_code = error.response['Error']['Code']
            print(f"Bedrock error: {error_code} - {error.response['Error']['Message']}")
            if error_code == 'ValidationException':
                return "Error: Input validation failed", None, self.history
            if error_code in ['ExpiredTokenException', 'ExpiredToken']:
                raise
            return f"Error: {error.response['Error']['Message']}", None, self.history
        except Exception as e:
            print(f"Unexpected error invoking Bedrock: {e}")
            return f"Error: {str(e)}", None, self.history

        self.history = invoke_result.get("history", messages)
        answer = invoke_result.get("response_text", "No response generated")
        jsondata = invoke_result.get("jsondata", None)
        return answer, jsondata, self.history

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
