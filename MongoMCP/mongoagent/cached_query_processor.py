import json
import traceback
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError
import requests
import asyncio
import mcp.types as mt
from .webui_bedrock_client import WebUiBedrockClient

#from .cache_utils import (
#    SimpleCache,
#    create_cache_key,
#    get_or_compute_async,
#)
from mongomcp.mongo_cache import (
    MongoSessionCache,
    create_cache_key,
    get_or_compute_async,
)

import fastmcp

class CachedQueryProcessor:
    """Enhanced QueryProcessor with comprehensive caching support
    
    Implements caching at multiple levels:
    1. Bedrock message caching with cache points
    2. MCP tool discovery caching
    3. MCP tool response caching
    4. Conversation history caching
    """

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
        self._system_prompt = []
        self._headers = {
            'Authorization': f'Bearer {settings.AUTH_TOKEN}',
            'Content-Type': 'application/json' 
        }        
        # Initialize caching system


        #self._tool_discovery_cache = SimpleCache(
        #    getattr(settings, "TOOL_DISCOVERY_CACHE_TTL", 300)
        #)
        #self._tool_response_cache = SimpleCache(
        #    getattr(settings, "TOOL_RESPONSE_CACHE_TTL", 60)
        #)
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

    def invoke_bedrock_with_tools(self, prompt: str) -> str:
        """
        Invoke Bedrock with MCP tools support and caching enabled
        
        Args:
            prompt: User prompt to send to the LLM
            
        Returns:
            str: Final assistant response
        """
        # Get tools dynamically from MCP server discovery (with caching)
        self.generate_toolconfig()
        
        # Prepare the conversation messages
        messages = self.history
        messages.append({
            "role": "user",
            "content": [{                
                "text": prompt
            }]
        })

        # Ensure Bedrock progress updates use the current handler instance.
        self.llm_client.message_handler = self.message_handler

        try:
            invoke_result = asyncio.run(
                self._invoke_bedrock_with_mcp_session(messages)
            )
        except ClientError as error:
            error_code = error.response['Error']['Code']
            print(f"Bedrock error: {error_code} - {error.response['Error']['Message']}")
            if error_code == 'ValidationException':
                return "Error: Input validation failed"
            if error_code in ['ExpiredTokenException', 'ExpiredToken']:
                raise
            return f"Error: {error.response['Error']['Message']}"
        except Exception as e:
            print(f"Unexpected error in invoke_bedrock_with_tools: {e}")
            return f"Error: {str(e)}"

        self.history = invoke_result.get("history", messages)
        return invoke_result.get("response_text", "No response generated")

    async def _invoke_bedrock_with_mcp_session(self, messages: list) -> dict:
        """Keep a single MCP session open while Bedrock executes tool calls."""
        # Reset Motor connections so they bind to this event loop, not the
        # closed loop used during discovery.
        self._tool_response_cache.reset_connection()
        if self.mcp_client is not None:
            async with self.mcp_client:
                return await self.llm_client.invoke_bedrock_with_tools_text(
                    messages=messages,
                )

        return await self.llm_client.invoke_bedrock_with_tools_text(
            messages=messages,
        )

    async def _execute_mcp_tool_cached_async(self, tool_name: str, tool_input: dict) -> str:
        """Async MCP tool execution with cache support for BedrockClient tool callbacks."""
        # Fast path: serve endpoint-scoped collection info from in-memory cache
        # instead of calling MCP again.
        suffix = "_get_collection_info"
        if tool_name.endswith(suffix):
            endpoint_key = tool_name[: -len(suffix)]
            if endpoint_key in self.mongo_collection_info:
                self.message_handler(
                    f"Using cached collection info for {endpoint_key}",
                    status="Tool Cache"
                )
                return json.dumps(self.mongo_collection_info.get(endpoint_key, {}), default=str)

        # Single-endpoint compatibility for unprefixed tool names.
        if tool_name == "get_collection_info" and self.mcp_endpoints and len(self.mcp_endpoints) == 1:
            endpoint_key = self.mcp_endpoints[0]
            if endpoint_key in self.mongo_collection_info:
                self.message_handler(
                    f"Using cached collection info for {endpoint_key}",
                    status="Tool Cache"
                )
                return json.dumps(self.mongo_collection_info.get(endpoint_key, {}), default=str)

        if not self.enable_response_caching:
            return await self._call_mcp_tool(tool_name, tool_input)

        cache_key = create_cache_key(tool_name, tool_input)
        result = await get_or_compute_async(
            cache=self._tool_response_cache,
            cache_key=cache_key,
            compute=lambda: self._call_mcp_tool(tool_name, tool_input),
            on_cache_hit=lambda: self.message_handler(f"Using cached response for {tool_name}"),            
        )
        self.message_handler(f"Response for {tool_name}", status="LLM Reasoning...")
        return result

    def discover_mcp_tools(self) -> dict:
        """
        Discover available MCP tools from the server, using a single event loop
        for both the Mongo cache I/O and the async MCP session to avoid Motor
        client binding to a closed loop.
        """
        cache_key = "mcp_tools_discovery"
        try:
            return asyncio.run(self._discover_mcp_tools_async(cache_key))
        except Exception as e:
            self.message_handler(f"Error discovering MCP tools: {e}", status="Error")
            return {"error": str(e), "tools": []}

    def _restore_endpoint_state(self, cached: dict) -> None:
        """Restore instance endpoint state from a previously cached discovery payload."""
        self.mcp_endpoints = cached.get("endpoints", [])
        self.mcp_endpoint_configs = cached.get("endpoint_configs", {})
        self.mongo_collection_info = cached.get("collection_info", {})
        # Rebuild the composite fastmcp client config so _call_mcp_tool can open sessions.
        self.mcp_tools_config = {"mcpServers": {
            name: cfg for name, cfg in self.mcp_endpoint_configs.items()
        }}
        self.mcp_client = fastmcp.Client(self.mcp_tools_config) if self.mcp_endpoint_configs else None

    async def _discover_mcp_tools_async(self, cache_key: str) -> dict:
        """Single-loop async wrapper: cache check → endpoint HTTP fetch → MCP discovery → cache write."""
        self._tool_discovery_cache.reset_connection()
        if self.enable_mcp_tool_caching:
            cached = await self._tool_discovery_cache.get(cache_key)
            if cached is not None:
                self.message_handler("Using cached MCP tools discovery")
                self._restore_endpoint_state(cached)
                return cached

        # Sync HTTP fetch to resolve available endpoint names.
        tools_url = f"{self.settings.mongo_mcp_root}/"
        self.message_handler(f"Discovering MCP endpoints from {tools_url}", status="Discovering Tools")
        try:
            response = requests.get(tools_url, headers=self._headers)
            response.raise_for_status()
            jdoc = response.json()
            self.mcp_endpoints = jdoc.get("available_tools", [])
        except requests.RequestException as e:
            self.message_handler(f"Error making web request to {tools_url}: {e}", status="Error")

        result = await self._discover_endpoint_tools()

        if self.enable_mcp_tool_caching and "error" not in result:
            # Persist endpoint metadata alongside the tool list so it can be restored on cache hit.
            result["endpoints"] = self.mcp_endpoints
            result["endpoint_configs"] = self.mcp_endpoint_configs
            result["collection_info"] = self.mongo_collection_info
            await self._tool_discovery_cache.set(cache_key, result)

        return result
    
    async def _discover_endpoint_tools(self) -> dict:
        """
        Async method to discover multiple mcp server endpoints and associated MCP tools
        this builds the required config for the fastmcp client and gets the tool and resource lists from each server assembling them together.
        
        Returns:
            dict: Dictionary containing available endpoints, tools, and their schemas
        """
        try:
            root_frmt = f"{self.settings.mongo_mcp_root}/{{}}/mcp"
            self.mcp_tools_config = {"mcpServers": {}}
            self.mcp_endpoint_configs = {}
            self.endpoint_clients = {}
            tools = []
            resources = []

            for name in self.mcp_endpoints:                
                endpoint = root_frmt.format(name)
                endpoint_config = {
                    "url": endpoint,
                    "transport": "http",
                    "headers": {"Authorization": f"Bearer {self.settings.AUTH_TOKEN}"}
                }
                self.mcp_tools_config["mcpServers"][name] = endpoint_config
                self.mcp_endpoint_configs[name] = endpoint_config
                # we're going to call get_collection info here so we don't need to have the LLM do it later.
                # this is a web API request that is separate from the MCP tool calls so we can cache 
                # the results for LLM access later
                self.mongo_collection_info[name] = {}
                try:
                    response = requests.get(f"{self.settings.mongo_mcp_root}/{name}/collection_info", headers=self._headers)
                    response.raise_for_status()
                    jdoc = response.json()
                    collection_info = jdoc.get("collection_info", None)
                    if collection_info:
                        self.mongo_collection_info[name] = collection_info
                except Exception as e:
                    await self.async_message_handler(f"Error getting collection info for {name}: {e}", status="Error")
                    traceback.print_exc()

            # Use one composite session to discover tools/resources across all endpoints.
            self.mcp_client = fastmcp.Client(self.mcp_tools_config)
            async with self.mcp_client as session:
                await session.ping()
                try:
                    tools_response = await session.list_tools()
                    await self.async_message_handler(
                        f"Discovered {len(tools_response)} tools from MCP server at {self.mcp_endpoints}",
                        status="Tools Discovered"
                    )
                    for t in tools_response:
                        tools.append({
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.inputSchema,
                            "annotation": t.annotations
                        })
                except Exception as e:
                    await self.async_message_handler(f"Error listing tools: {e}", status="Error")
                    traceback.print_exc()
                '''
                # List available resources
                try:
                    resources_response = await session.list_resources()
                    resources.extend([
                        {
                            "uri": resource.uri,
                            "name": resource.name,
                            "description": resource.description,
                            "mime_type": resource.mimeType
                        }
                        for resource in resources_response
                    ])
                except Exception as e:
                    await self.async_message_handler(f"Error listing resources: {e}", status="Error")
                '''
                    
            return {
                "tools": tools,
                "resources": resources
            }
        
        except Exception as e:
            await self.async_message_handler(f"Failed to discover MCP tools: {e}", status="Error")   
            traceback.print_exc()                     
            return {"error": str(e), "tools": [], "resources": []}

        
    def generate_toolconfig(self) -> list:
        """
        Get Bedrock-formatted tools from MCP server discovery with caching
        
        Returns:
            list: List of tools in Bedrock toolSpec format
        """
        if not self.mcp_tools_config:            
            mcp_info = self.discover_mcp_tools()
            bedrock_tools = []
            
            if "error" in mcp_info:
                self.message_handler(f"MCP discovery failed {mcp_info['error']}", status="Error")
                self.message_handler(str(mcp_info), status="Error")                

            for tool in mcp_info.get("tools", []):
                bedrock_tool = {
                    "toolSpec": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "inputSchema": {
                            "json": tool["input_schema"]
                        }
                    }
                }
                bedrock_tools.append(bedrock_tool)
            
            # build out a set of toplevel directives for the LLM based on the collection info from each server, 
            # this will help guide the LLM to use vector search appropriately and understand the capabilities of each collection.
            # this also controls the output format
            if self.mongo_collection_info is not None:
                self._system_prompt = [
                    {"text": prompt_text}
                    for prompt_text in getattr(self.settings, "BEDROCK_SYSTEM_PROMPT_TEXTS", [])
                ]
                self._system_prompt.append({"text": json.dumps(self.mongo_collection_info)})

            self.llm_client.system = self._system_prompt

            self.mcp_tools_config = bedrock_tools        
            self.llm_client.configure_tools(self.mcp_tools_config, self._execute_mcp_tool_cached_async)
            self.message_handler(f"Using {len(bedrock_tools)} tools discovered from MCP server", status="Tools Ready")

        return self.mcp_tools_config


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
 
    def query_with_mcp_tools(self, question: str, history: list or None = None) -> tuple:
        """
        Query LLM with MCP tool support using Bedrock's Converse API with caching
        
        Args:
            question: User question or full prompt 
            history: optional list of historical questions and assistant answers
            
        Returns:
            tuple: (assistant response (str), updated history (list))
        """
        
        # Update history if provided
        if history:
            self.history = history
        if self.history is None:
            self.history = []
        
        # Invoke Bedrock with MCP tools and caching
        assistant_message = self.invoke_bedrock_with_tools(question)
        return assistant_message, self.history

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring"""
        return {
            "tool_discovery_cache_size": len(self._tool_discovery_cache._cache),
            "tool_response_cache_size": len(self._tool_response_cache._cache),
            "caching_enabled": {
                "bedrock": self.llm_client.enable_cache_points,
                "mcp_tools": self.enable_mcp_tool_caching,
                "responses": self.enable_response_caching
            }
        }
