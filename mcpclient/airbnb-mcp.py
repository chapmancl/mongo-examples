import boto3
from botocore.exceptions import ClientError
import asyncio
import settings
import fastmcp

class QueryProcessor:
    """A class to process user queries using MCP server based search and Claude LLM via Bedrock converse.
    
    Manages configuration, MongoDB Atlas vector search, and AWS Bedrock interactions
    to retrieve and aggregate facts based on user questions.
    """

    def __init__(self):
        """Initializes the QueryProcessor with configuration from settings.py.
        
        Sets up Bedrock and MongoDB clients, and connects to the vector collection.
        """
        # Conversation history (starts empty)
        self.history = None
        # AWS session objects
        self.bedrock_client = None
        self._create_bedrock_client()
        
        self.mcp_client = fastmcp.Client(settings.mong_mcp)
        self.mcp_tools_config = None
        
    def _create_bedrock_client(self) -> None:
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws_region
        )

    def invoke_bedrock_with_tools(self, prompt: str, max_iterations=5) -> str:
        """
        Invoke Bedrock with MCP tools support, handling tool calls iteratively
        
        Args:
            prompt: User prompt to send to the LLM
            max_iterations: Maximum number of tool call iterations
            
        Returns:
            str: Final assistant response
        """
        # Get tools dynamically from MCP server discovery
        tools = self.get_bedrock_tools_from_mcp()
        print(f"Using {len(tools)} tools discovered from MCP server")
        
        # Prepare the conversation messages
        messages = self.history
        messages.append({
            "role": "user",
            "content": [{                
                "text": prompt
                }]
        })
        
        # Tool configuration for Bedrock
        tool_config = {
            "tools": tools
            #"toolChoice": {"auto": {}}
        }
        
        for iteration in range(max_iterations):
            try:
                # Invoke Bedrock using the Converse API
                response = self.bedrock_client.converse(
                    modelId=settings.LLM_MODEL_ID,
                    messages=messages,
                    toolConfig=tool_config
                )
                
                # Get the assistant's response
                assistant_message = response['output']['message']
                # so we can see what the LLM is reasoning
                print(assistant_message["content"][0].get("text","no LLM message"))
                # enable caching?
                #assistant_message["content"].append({"cachePoint" : {"type": "default"}})                
                messages.append(assistant_message)
                
                # Check if the assistant wants to use tools
                if 'content' in assistant_message:
                    tool_calls = []
                    text_content = []
                    
                    for content in assistant_message['content']:
                        if 'toolUse' in content:
                            tool_calls.append(content['toolUse'])
                            #print(f"Tool call detected: {content['toolUse']}")
                        elif 'text' in content:
                            text_content.append(content['text'])
                    
                    # If there are tool calls, execute them
                    if tool_calls:
                        tool_results = []
                        
                        for tool_call in tool_calls:
                            tool_name = tool_call['name']
                            tool_input = tool_call['input']
                            tool_use_id = tool_call['toolUseId']
                            
                            # Execute the MCP tool call
                            try:
                                tool_result = self._execute_mcp_tool(tool_name, tool_input)
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
                            messages.append({"role": "user", "content": tool_results})                
                            continue  # Continue the conversation loop
                    
                    # If no tool calls, return the text response
                    if text_content:
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
    
    async def message_handler(self, message):
        """Handle incoming messages from the server."""
        if isinstance(message, Exception):
            print(f"Error in message handler: {message}")
            return    
        print(f"Received message from server: {message}")
    
    
    def discover_mcp_tools(self) -> dict:
        """
        Discover available MCP tools from the server
        
        Returns:
            dict: Dictionary containing available tools and their schemas
        """
        try:            
            return asyncio.run(self._discover_mcp_tools_async())
        except Exception as e:
            print(f"Error discovering MCP tools: {e}")
            return {"error": str(e), "tools": []}
    
    async def _discover_mcp_tools_async(self) -> dict:
        """
        Async method to discover MCP tools from the server using persistent session
        
        Returns:
            dict: Dictionary containing available tools and their schemas
        """
        try:
            #session = await self._initialize_mcp_session()
            async with self.mcp_client as session:
                # Ping the server to ensure connection
                print("Pinging MCP server...")
                await session.ping()
                print("MCP server is reachable")
            
                # List available tools with error handling
                tools = []
                try:
                    tools_response = await session.list_tools()     
                    
                    print(f"Discovered {len(tools_response)} tools from MCP server")
                    for t in tools_response:
                        tools.append(
                            {
                                "name": t.name,
                                "description": t.description,
                                "input_schema": t.inputSchema,
                                "annotation": t.annotations
                            }
                        )                    
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
            return {"error": str(e), "tools": [], "resources": []}
    
    def get_bedrock_tools_from_mcp(self) -> list:
        """
        Get Bedrock-formatted tools from MCP server discovery
        
        Returns:
            list: List of tools in Bedrock toolSpec format
        """
        if self.mcp_tools_config is None:            
            mcp_info = self.discover_mcp_tools()
            bedrock_tools = []
            
            if "error" in mcp_info:
                print(f"MCP discovery failed, using fallback tools: {mcp_info['error']}")
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
            self.mcp_tools_config = bedrock_tools

        return self.mcp_tools_config
    
    

    def _execute_mcp_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute an MCP tool call using the AWS MCP server
        
        Args:
            tool_name: Name of the MCP tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            str: Tool execution result
        """
        try:
            print(f"Executing MCP tool: {tool_name} with args: {tool_input}")
            return asyncio.run(self._call_mcp_tool(tool_name,tool_input))            
    
        except Exception as e:
            print(f"Error calling MCP server for {tool_name}: {e}")
    
    async def _call_mcp_tool(self, toolname: str, tool_input: dict ) -> str:
        """Initialize a persistent MCP session for tool calls."""
        try:
            async with self.mcp_client as session:
                result = await session.call_tool(toolname, tool_input)
                return result.content[0].text
        except Exception as e:
            print(f"Failed MCP {toolname} call: {e}")
            raise
 
    def query_claude_with_mcp_tools(self, question: str, history: list or None = None) -> tuple:
        """
        Query Claude with MCP tool support using Bedrock's Converse API
        
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
        
        # Invoke Bedrock with MCP tools
        assistant_message = self.invoke_bedrock_with_tools(question)
        return assistant_message, self.history

    def run(self) -> None:
        """Runs an interactive loop on the command line to handle user questions.
        
        Supports direct Claude queries (prefixed with 'ask'), MCP tool queries (prefixed with 'mcp'),
        or vector-backed fact retrieval.
        """
        print("Enter questions (Press Ctrl+C to stop):")
        print("Commands:")
        print("  clear - Clear conversation history")
        print("  <question> -  Claude query with MCP tool support")
        try:
            while True:
                # Get user input and strip whitespace
                user_input = input("Question: ").strip()
                answer = "unknown"  # Default answer if no processing occurs
                if not user_input:
                    answer = "Not a valid question"
                elif user_input.startswith("clear"):
                    self.history = None
                    self.mcp_tools_config = None
                    answer = "history and tools cleared..."
                else:
                    # Claude query with MCP tool support
                    answer, history = self.query_claude_with_mcp_tools(user_input)                    
                    # clear the answer, it was already displayed
                    answer = None
                if answer:
                    print(f"Answer: {answer}")
        except ClientError as error:
            error_code = error.response['Error']['Code']
            if error_code in ['ExpiredTokenException', 'ExpiredToken']:
                print("AWS Token has expired!", error)
            elif error_code == 'ValidationException':
                # if input exceeds token limit just drop it all
                self.history = None 
                print("too much history, clearing...", error)
                self.run()
            else:
                # Log other errors                 
                print("Some other AWS client error occurred:", error.response)
        except KeyboardInterrupt:
            # Handle user interruption
            print("\nKeyboard interrupt received, exiting...")

def main():    
    processor = QueryProcessor()
    processor.run()

if __name__ == "__main__":
    main()
