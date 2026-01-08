import json
import boto3
from botocore.exceptions import ClientError
import re
import time
import traceback
import hashlib
from typing import Dict, Any, Optional
import requests
import asyncio
import settings
import fastmcp
from fastmcp.client.transports import StreamableHttpTransport

class SimpleCache:
    """Simple in-memory cache with TTL support"""
    def __init__(self, default_ttl=300):
        self._cache = {}
        self._default_ttl = default_ttl
    
    def get(self, key: str):
        if key in self._cache:
            data, timestamp, ttl = self._cache[key]
            if time.time() - timestamp < ttl:
                return data
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        ttl = ttl or self._default_ttl
        self._cache[key] = (value, time.time(), ttl)
    
    def clear(self):
        self._cache.clear()
    
    def remove_pattern(self, pattern: str):
        """Remove keys matching a pattern"""
        keys_to_remove = [k for k in self._cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self._cache[key]

class CachedQueryProcessor:
    """Enhanced QueryProcessor with comprehensive caching support
    
    Implements caching at multiple levels:
    1. Bedrock message caching with cache points
    2. MCP tool discovery caching
    3. MCP tool response caching
    4. Conversation history caching
    """

    def __init__(self):
        """Initializes the CachedQueryProcessor with caching configuration"""
        # Conversation history (starts empty)
        self.history = None
        
        # AWS session objects
        self.bedrock_client = None
        self._create_bedrock_client()
        
        self.mcp_client = None
        self.mcp_tools_config = None
        self.mongo_tools = None
        self.mongo_collection_info = {}
        self._system_prompt = ""
        self._headers = {
            'Authorization': f'Bearer {settings.AUTH_TOKEN}',
            'Content-Type': 'application/json' 
        }        
        
        # Initialize caching system
        self._init_caches()
        self.get_bedrock_tools_from_mcp()
        
    def _init_caches(self):
        """Initialize cache systems with configurable TTLs"""
        # Cache settings from settings or defaults
        tool_discovery_ttl = getattr(settings, 'TOOL_DISCOVERY_CACHE_TTL', 300)
        tool_response_ttl = getattr(settings, 'TOOL_RESPONSE_CACHE_TTL', 60)
        
        # Initialize caches
        self._tool_discovery_cache = SimpleCache(tool_discovery_ttl)
        self._tool_response_cache = SimpleCache(tool_response_ttl)
        
        # Cache control flags
        self.enable_bedrock_caching = getattr(settings, 'ENABLE_BEDROCK_CACHING', True)
        self.enable_mcp_tool_caching = getattr(settings, 'ENABLE_MCP_TOOL_CACHING', True)
        self.enable_response_caching = getattr(settings, 'ENABLE_RESPONSE_CACHING', True)
        
    def clear_all_caches(self):
        """Clear all caches - useful for testing or when data changes"""
        self._tool_discovery_cache.clear()
        self._tool_response_cache.clear()
        self.mcp_tools_config = None
        self.get_bedrock_tools_from_mcp()
        print("All caches cleared")
        
    def _create_bedrock_client(self) -> None:
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws_region
        )
    
    def _create_cache_key(self, tool_name: str, tool_input: dict) -> str:
        """Create a deterministic cache key from tool name and input"""
        # Sort the input dict to ensure consistent key generation
        sorted_input = json.dumps(tool_input, sort_keys=True)
        input_hash = hashlib.md5(sorted_input.encode()).hexdigest()
        return f"{tool_name}:{input_hash}"

    def invoke_bedrock_with_tools(self, prompt: str, max_iterations=10) -> str:
        """
        Invoke Bedrock with MCP tools support and caching enabled
        
        Args:
            prompt: User prompt to send to the LLM
            max_iterations: Maximum number of tool call iterations
            
        Returns:
            str: Final assistant response
        """
        # Get tools dynamically from MCP server discovery (with caching)
        tools = self.get_bedrock_tools_from_mcp()
        
        # Prepare the conversation messages
        messages = self.history
        messages.append({
            "role": "user",
            "content": [{                
                "text": prompt
            }]
        })
        
        # Smart cache point management - only add to key messages to stay under 4 block limit
        if self.enable_bedrock_caching:
            self._manage_cache_points(messages)
        
        # Tool configuration for Bedrock
        tool_config = {
            "tools": tools
        }
        
        for iteration in range(max_iterations):
            try:
                # Invoke Bedrock using the Converse API
                response = self.bedrock_client.converse(
                    system=self._system_prompt,
                    modelId=settings.LLM_MODEL_ID,
                    messages=messages,
                    toolConfig=tool_config
                )
                
                # Get the assistant's response
                assistant_message = response['output']['message']
                
                # Display LLM reasoning
                if assistant_message.get("content"):
                    for content in assistant_message["content"]:
                        if content.get("text"):
                            print(f"LLM: {content['text']}")
                
                messages.append(assistant_message)
                
                # Check if the assistant wants to use tools
                if 'content' in assistant_message:
                    tool_calls = []
                    text_content = []
                    
                    for content in assistant_message['content']:
                        if 'toolUse' in content:
                            tool_calls.append(content['toolUse'])
                        elif 'text' in content:
                            text_content.append(content['text'])
                    
                    # If there are tool calls, execute them
                    if tool_calls:
                        tool_results = []
                        
                        for tool_call in tool_calls:
                            tool_name = tool_call['name']
                            tool_input = tool_call['input']
                            tool_use_id = tool_call['toolUseId']
                            
                            # Execute the MCP tool call (with caching)
                            try:
                                tool_result = self._execute_mcp_tool_cached(tool_name, tool_input)
                                tool_results.append({
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": str(tool_result)}]
                                    }
                                })
                            except Exception as e:
                                print(f"Error executing MCP tool {tool_name}: {e}")
                                tool_results.append({
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": f"Error: {str(e)}"}],
                                        "status": "error"
                                    }
                                })
                        
                        # Add tool results to the conversation
                        if tool_results:
                            tool_message = {"role": "user", "content": tool_results}
                            messages.append(tool_message)                
                            continue  # Continue the conversation loop
                    
                    # If no tool calls, return the text response
                    if text_content:
                        self.history = messages
                        return " ".join(text_content)
                
                # If we get here, there was no content to process
                return "No response generated"
                
            except ClientError as error:
                error_code = error.response['Error']['Code']
                print(f"Bedrock error: {error_code} - {error.response['Error']['Message']}")
                if error_code == 'ValidationException':
                    return "Error: Input validation failed"
                elif error_code in ['ExpiredTokenException', 'ExpiredToken']:
                    raise
                else:
                    return f"Error: {error.response['Error']['Message']}"
            except Exception as e:
                print(f"Unexpected error in invoke_bedrock_with_tools: {e}")
                return f"Error: {str(e)}"
        
        self.history = messages
        return "Maximum iterations reached without completion"

    def _manage_cache_points(self, messages):
        """
        Smart cache point management to stay under Bedrock's 4 cache block limit
        Only adds cache points to the most important messages
        """
        cache_point = {"cachePoint": {"type": "default"}}
        cache_points_added = 0
        max_cache_points = 4
        
        # Remove any existing cache points first
        for message in messages:
            if 'content' in message:
                message['content'] = [
                    content for content in message['content'] 
                    if 'cachePoint' not in content
                ]
        
        # Strategy: Add cache points to recent user messages and key assistant responses
        # Work backwards from the most recent message
        for i in range(len(messages) - 1, -1, -1):
            if cache_points_added >= max_cache_points:
                break
                
            message = messages[i]
            
            # Add cache points to recent user messages (every 2nd message)
            if (message.get('role') == 'user' and 
                cache_points_added < max_cache_points and 
                (len(messages) - i) % 2 == 1):
                message['content'].append(cache_point.copy())
                cache_points_added += 1
            
            # Add cache points to assistant messages with tool results
            elif (message.get('role') == 'assistant' and 
                  cache_points_added < max_cache_points and
                  any('toolUse' in str(content) for content in message.get('content', []))):
                message['content'].append(cache_point.copy())
                cache_points_added += 1
        
        if cache_points_added > 0:
            print(f"Added {cache_points_added} cache points to conversation")

    def _execute_mcp_tool_cached(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute an MCP tool call with caching support
        
        Args:
            tool_name: Name of the MCP tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            str: Tool execution result
        """
        # Check if caching is disabled
        if not self.enable_response_caching:
            return self._execute_mcp_tool_direct(tool_name, tool_input)
        
        # Create cache key
        cache_key = self._create_cache_key(tool_name, tool_input)
        
        # Try to get from cache first
        cached_result = self._tool_response_cache.get(cache_key)
        if cached_result is not None:
            print(f"Using cached response for {tool_name}")
            return cached_result
        
        # Cache miss - execute the tool
        try:
            print(f"Executing MCP tool: {tool_name} with args: {tool_input}")
            result = self._execute_mcp_tool_direct(tool_name, tool_input)
            
            # Cache the result
            self._tool_response_cache.set(cache_key, result)
            
            return result
        except Exception as e:
            print(f"Error calling MCP server for {tool_name}: {e}")
            raise

    def _execute_mcp_tool_direct(self, tool_name: str, tool_input: dict) -> str:
        """Execute MCP tool without caching"""
        return asyncio.run(self._call_mcp_tool(tool_name, tool_input))

    async def message_handler(self, message):
        """Handle incoming messages from the server."""
        if isinstance(message, Exception):
            print(f"Error in message handler: {message}")
            return    
        print(f"Received message from server: {message}")
    
    def discover_mcp_tools(self) -> dict:
        """
        Discover available MCP tools from the server with caching
        
        Returns:
            dict: Dictionary containing available tools and their schemas
        """
        # Check if caching is disabled
        if not self.enable_mcp_tool_caching:
            return self._discover_mcp_tools_direct()
        
        cache_key = "mcp_tools_discovery"
        
        # Try to get from cache first
        cached_tools = self._tool_discovery_cache.get(cache_key)
        if cached_tools is not None:
            print("Using cached MCP tools discovery")
            return cached_tools
        
        # Cache miss - discover tools
        try:
            tools_data = self._discover_mcp_tools_direct()
            
            # Cache the results
            self._tool_discovery_cache.set(cache_key, tools_data)
            
            return tools_data
        except Exception as e:
            print(f"Error discovering MCP tools: {e}")
            return {"error": str(e), "tools": []}

    def _discover_mcp_tools_direct(self) -> dict:
        """Discover MCP tools without caching"""
        try:
            #call the root url and get the available tools
            tools_url = f"{settings.mongo_mcp_root}/"
            print(f"Discovering MCP tools from {tools_url}")

            # Make web request to tools_url and return dict data
            try:
                response = requests.get(tools_url, headers=self._headers)
                response.raise_for_status()
                jdoc = response.json()
                available_tools = jdoc.get("available_tools", [])
                self.mongo_tools = available_tools
                print(available_tools)
                
            except requests.RequestException as e:
                print(f"Error making web request to {tools_url}: {e}")
                #return {"error": str(e), "tools": [], "resources": []}

            if self.mongo_tools:
                return asyncio.run(self._discover_multi_mcptools())
            else:
                return asyncio.run(self._discover_mcp_tools_async())
            
        except Exception as e:
            print(f"Error discovering MCP tools: {e}")
            return {"error": str(e), "tools": []}
    
    async def _discover_multi_mcptools(self) -> dict:
        """
        Async method to discover MCP tools from a multiple mcp server endpoints in mongo_tools
        
        Returns:
            dict: Dictionary containing available endpoints, tools, and their schemas
        """
        try:
            root_frmt = f"{settings.mongo_mcp_root}/{{}}/mcp"
            self.mcp_tools_config = { }
            tools = []
            resources = []

            for name in self.mongo_tools:
                print(f"Found MCP server at {name}")
                endpoint = root_frmt.format(name)
                self.mcp_tools_config[name] = {
                    "url": endpoint,
                    "headers": {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}
                }
                # we're going to call get_collection infor here too so we don't need to have the LLM do it later.
                self.mongo_collection_info[name] = {}
                try:
                    response = requests.get(f"{settings.mongo_mcp_root}/{name}/collection_info", headers=self._headers)
                    response.raise_for_status()
                    jdoc = response.json()
                    collection_info = jdoc.get("collection_info", None)
                    if collection_info:
                        self.mongo_collection_info[name] = collection_info
                except Exception as e:
                    print(f"Error getting collection info for {name}: {e}")
                    traceback.print_exc()

            self.mcp_client = fastmcp.Client(self.mcp_tools_config)
            async with self.mcp_client as session:
                await session.ping()                    
                try:
                    tools_response = await session.list_tools()     
                    print(f"Discovered {len(tools_response)} tools from MCP server at {self.mongo_tools}")
                    for t in tools_response:
                        tools.append({
                            "name": f"{t.name}",
                            "description": t.description,
                            "input_schema": t.inputSchema,
                            "annotation": t.annotations
                        })
                    
                except Exception as e:
                    print(f"Error listing tools: {e}")
                    traceback.print_exc()
                
                # List available resources                     
                try:
                    resources_response = await session.list_resources()
                    resources.extend([
                        {
                            "uri": resource.uri,
                            "name": f"{resource.name}",
                            "description": resource.description,
                            "mime_type": resource.mimeType
                        }
                        for resource in resources_response
                    ])
                except Exception as e:
                    print(f"Error listing resources: {e}")
                    
            return {
                "tools": tools,
                "resources": resources
            }
        
        except Exception as e:
            print(f"Failed to discover MCP tools: {e}")   
            traceback.print_exc()                     
            return {"error": str(e), "tools": [], "resources": []}

    async def _discover_mcp_tools_async(self) -> dict:
        """
        Async method to discover MCP tools from a single server using persistent session
        
        Returns:
            dict: Dictionary containing available tools and their schemas
        """
        try:
            #self.mcp_client = fastmcp.Client(self.mcp_tools_config)
            async with self.mcp_client as session:
                print("Pinging MCP server...")
                await session.ping()
                print("MCP server is reachable")
            
                # List available tools with error handling
                tools = []
                try:
                    tools_response = await session.list_tools()     
                    
                    print(f"Discovered {len(tools_response)} tools from MCP server")
                    for t in tools_response:
                        tools.append({
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.inputSchema,
                            "annotation": t.annotations
                        })                    
                except Exception as e:
                    print(f"Error listing tools: {e}")
                
                # List available resources 
                resources = []
                try:
                    resources_response = await session.list_resources()
                    resources = [
                        {
                            "uri": resource.uri,
                            "name": resource.name,
                            "description": resource.description,
                            "mime_type": resource.mimeType
                        }
                        for resource in resources_response
                    ]
                except Exception as e:
                    print(f"Error listing resources: {e}")
                
                return {
                    "tools": tools,
                    "resources": resources
                }
            
        except Exception as e:
            print(f"Failed to discover MCP tools: {e}")
            traceback.print_exc()           
            return {"error": str(e), "tools": [], "resources": []}
    
    def get_bedrock_tools_from_mcp(self) -> list:
        """
        Get Bedrock-formatted tools from MCP server discovery with caching
        
        Returns:
            list: List of tools in Bedrock toolSpec format
        """
        if self.mcp_tools_config is None:            
            mcp_info = self.discover_mcp_tools()
            bedrock_tools = []
            
            if "error" in mcp_info:
                print(f"MCP discovery failed {mcp_info['error']}")
                print(mcp_info)                

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
            
            if self.mongo_collection_info is not None:
                self._system_prompt = [
                    {"text":"***IMPORTANT: always use vector_search before aggregate_query"},
                    {"text":"Only use vector_search with collections that have a search_indexes.type=vectorSearch."},
                    {"text": json.dumps(self.mongo_collection_info)}
                ]

            self.mcp_tools_config = bedrock_tools        
            print(f"Using {len(bedrock_tools)} tools discovered from MCP server")

        return self.mcp_tools_config


    async def _call_mcp_tool(self, toolname: str, tool_input: dict) -> str:
        """Initialize a stateless session for tool calls."""
        try:
            #self.mcp_client = fastmcp.Client(self.mcp_tools_config)
            async with self.mcp_client as session:
                result = await session.call_tool(toolname, tool_input)
            return result.content[0].text
        except Exception as e:
            print(f"Failed MCP {toolname} call: {e}")
            traceback.print_exc()
            raise
 
    def query_claude_with_mcp_tools(self, question: str, history: list or None = None) -> tuple:
        """
        Query Claude with MCP tool support using Bedrock's Converse API with caching
        
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

    def invalidate_cache_for_collection(self, collection_name: str):
        """Invalidate caches related to a specific collection"""
        self._tool_response_cache.remove_pattern(collection_name)
        print(f"Invalidated caches for collection: {collection_name}")

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring"""
        return {
            "tool_discovery_cache_size": len(self._tool_discovery_cache._cache),
            "tool_response_cache_size": len(self._tool_response_cache._cache),
            "caching_enabled": {
                "bedrock": self.enable_bedrock_caching,
                "mcp_tools": self.enable_mcp_tool_caching,
                "responses": self.enable_response_caching
            }
        }

    def _get_clean_input(self, prompt: str = "Question: ") -> str:
        """Get clean input from user, handling buffer issues and multi-line text"""
        import sys
        
        # Flush any pending output first
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Clear input buffer by reading any leftover characters
        if sys.stdin.isatty():  # Only do this for interactive terminals
            try:
                import select
                import tty
                import termios
                
                # Check if there's data waiting in stdin
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    # Read and discard any leftover input
                    old_settings = termios.tcgetattr(sys.stdin)
                    try:
                        tty.setcbreak(sys.stdin.fileno())
                        while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                            sys.stdin.read(1)
                    finally:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except (ImportError, OSError):
                # Fallback for systems without select/tty/termios
                pass
        
        # Get input and clean it
        try:
            user_input = input(prompt).strip()
            
            # Handle cases where input might contain multiple lines
            # If the input seems truncated (no proper ending), prompt for continuation
            while user_input and not self._is_complete_input(user_input):
                print("Input appears incomplete. Continue or press Enter to submit:")
                continuation = input("... ").strip()
                if not continuation:
                    break
                user_input += " " + continuation
                
            return user_input
            
        except EOFError:
            # Handle Ctrl+D gracefully
            raise KeyboardInterrupt
    
    def _is_complete_input(self, text: str) -> bool:
        """Check if input appears to be complete (simple heuristic)"""
        if not text:
            return True
            
        # Check for common indicators of complete input
        complete_indicators = ['.', '?', '!', '"', "'", ')', '}', ']']
        incomplete_indicators = ['(', '{', '[', '"', "'"]
        
        # If it ends with a complete indicator, likely complete
        if text[-1] in complete_indicators:
            return True
            
        # If it ends abruptly mid-word or with incomplete indicators, likely incomplete
        if text[-1] in incomplete_indicators:
            return False
            
        # Default to complete
        return True

    def run(self) -> None:
        """Runs an interactive loop with caching support"""
        print("Enhanced QueryProcessor with Caching enabled")
        print("Enter questions (Press Ctrl+C to stop):")
        print("Commands:")
        print("  clear - Clear conversation history and caches")
        print("  cache stats - Show cache statistics")
        print("  cache clear - Clear all caches")
        print("  <question> - Claude query with MCP tool support and caching")
        print("Note: For multi-line input, the system will detect incomplete input and prompt for continuation.")
        
        try:
            while True:
                # Use the clean input method instead of direct input()
                user_input = self._get_clean_input("Question: ")
                answer = "unknown"
                
                if not user_input:
                    answer = "Not a valid question"
                elif user_input.startswith("clear"):
                    self.history = None
                    self.clear_all_caches()
                    answer = "History and caches cleared..."
                elif user_input.startswith("cache stats"):
                    stats = self.get_cache_stats()
                    answer = f"Cache Statistics: {json.dumps(stats, indent=2)}"
                elif user_input.startswith("cache clear"):
                    self.clear_all_caches()
                    answer = "All caches cleared"
                else:
                    # Claude query with MCP tool support and caching
                    answer, history = self.query_claude_with_mcp_tools(user_input)                    
                    answer = None  # Already displayed during processing
                    
                if answer:
                    print(f"Answer: {answer}")
                    
        except ClientError as error:
            error_code = error.response['Error']['Code']
            if error_code in ['ExpiredTokenException', 'ExpiredToken']:
                print("AWS Token has expired!", error)
            elif error_code == 'ValidationException':
                self.history = None 
                self.clear_all_caches()
                print("Too much history, clearing...", error)
                self.run()
            else:
                print("Some other AWS client error occurred:", error.response)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting...")
            print("Final cache stats:", self.get_cache_stats())

    def write_dict_to_json_file(self, data_dict: Dict[str, Any], filename: str) -> None:
        """
        Write a Python dictionary to a JSON file
        
        Args:
            data_dict: Dictionary to write to JSON file
            filename: Path to output JSON file
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)
            print(f"Dictionary successfully written to {filename}")
        except Exception as e:
            print(f"Error writing dictionary to JSON file: {e}")

def main():    
    processor = CachedQueryProcessor()
    processor.query_claude_with_mcp_tools("What is the claim status for nicole weber?")
    processor.run()

if __name__ == "__main__":
    main()
