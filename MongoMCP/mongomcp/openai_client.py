"""
MongoDB AI/LLM client functions using OpenAI and Azure OpenAI

"""

import datetime
import json
import re
import asyncio
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
import logging
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()  
        return json.JSONEncoder.default(self, obj)


class OpenAIClient:
    """
    OpenAI Client for MCP tool calls and LLM invocations
    Handles the LLM interactions and tool integrations for the LLM.
    Supports both OpenAI and Azure OpenAI.
    """
    def __init__(self, settings, use_azure: bool = False):
        self.settings = settings
        self.use_azure = use_azure
        
        # Initialize the appropriate client
        if use_azure:
            # Azure OpenAI configuration
            self.client = AsyncAzureOpenAI(
                api_key=getattr(settings, 'AZURE_OPENAI_API_KEY', None),
                api_version=getattr(settings, 'AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
                azure_endpoint=getattr(settings, 'AZURE_OPENAI_ENDPOINT', None)
            )
            self.model_id = getattr(settings, 'AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')
            self.embedding_model_id = getattr(settings, 'AZURE_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-ada-002')
        else:
            # Standard OpenAI configuration
            self.client = AsyncOpenAI(
                api_key=getattr(settings, 'OPENAI_API_KEY', None)
            )
            self.model_id = getattr(settings, 'OPENAI_MODEL_ID', 'gpt-4-turbo-preview')
            self.embedding_model_id = getattr(settings, 'OPENAI_EMBEDDING_MODEL_ID', 'text-embedding-3-large')
        
        self.mcp_tools = None
        self.mcp_call = None
        self.llm_setup = False
        
        # Invoke behavior is configured on the client instance, not per call.
        self.max_iterations = getattr(settings, 'LLM_MAX_ITERATIONS', 10)
        self.system = None
        self.message_handler = None
        self.show_response_progress = True
        
        
    def configure_tools(self, tools_config, tool_handler: Optional[Callable] = None):
        """
        Configure MCP tools for OpenAI client.

        tool_handler should accept (toolname, tool_input).
        If not provided, subclasses can override _call_mcp_tool.
        """
        self.mcp_tools = tools_config
        self.mcp_call = tool_handler
        self.llm_setup = True

    def _emit_progress(self, message_handler: Optional[Callable], message: str, status: str = "Processing") -> None:
        """Emit optional progress updates without impacting request flow."""
        if not message_handler:
            return
        try:
            message_handler(message, status=status)
        except Exception:
            # Progress updates should never fail the main LLM flow.
            return

    def _try_parse_json(self, json_string):
        """ try to parse a string to json, return json obj or nothing """
        try:
            # Remove markdown code fences
            json_string = re.sub(r'^```json\s*', '', json_string)
            json_string = re.sub(r'^```\s*', '', json_string)
            json_string = re.sub(r'\s*```$', '', json_string)

            # Find the JSON object (between first { and last })
            start_idx = json_string.find('{')
            end_idx = json_string.rfind('}')

            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                return None
            else:
                json_string = json_string[start_idx:end_idx+1]
                json_string = json_string.strip()
                data = json.loads(json_string)
                return data
        except (json.JSONDecodeError, ValueError):
            return None
        return None

    def _convert_mcp_tools_to_openai_format(self) -> List[Dict[str, Any]]:
        """
        Convert MCP tools from Bedrock format to OpenAI function calling format.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        if not self.mcp_tools:
            return []
        
        openai_tools = []
        for tool in self.mcp_tools:
            if 'toolSpec' in tool:
                spec = tool['toolSpec']
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": spec.get('name', ''),
                        "description": spec.get('description', ''),
                        "parameters": spec.get('inputSchema', {})
                    }
                }
                openai_tools.append(openai_tool)
        
        return openai_tools

    def _convert_bedrock_messages_to_openai(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Bedrock message format to OpenAI message format.

        Args:
            messages: List of messages in Bedrock format

        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []

        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', [])

            # Handle user messages
            if role == 'user':
                # Check if this is a tool result message
                if content and isinstance(content, list) and any('toolResult' in c for c in content):
                    # Convert tool results to OpenAI format
                    for item in content:
                        if 'toolResult' in item:
                            tool_result = item['toolResult']
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_result.get('toolUseId', ''),
                                "content": tool_result.get('content', [{}])[0].get('text', '')
                            })
                else:
                    # Regular user message
                    text_content = ""
                    for item in content:
                        if 'text' in item:
                            text_content += item['text']
                        # Skip cache points
                        if 'cachePoint' not in item:
                            pass

                    if text_content:
                        openai_messages.append({
                            "role": "user",
                            "content": text_content
                        })

            # Handle assistant messages
            elif role == 'assistant':
                text_content = ""
                tool_calls = []

                for item in content:
                    if 'text' in item:
                        text_content += item['text']
                    elif 'toolUse' in item:
                        tool_use = item['toolUse']
                        tool_calls.append({
                            "id": tool_use.get('toolUseId', ''),
                            "type": "function",
                            "function": {
                                "name": tool_use.get('name', ''),
                                "arguments": json.dumps(tool_use.get('input', {}))
                            }
                        })

                msg_dict = {"role": "assistant"}
                if text_content:
                    msg_dict["content"] = text_content
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls

                openai_messages.append(msg_dict)

        return openai_messages

    def _convert_openai_message_to_bedrock(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI message format back to Bedrock format.

        Args:
            message: Message in OpenAI format

        Returns:
            Message in Bedrock format
        """
        role = message.get('role', '')

        if role == 'assistant':
            content = []

            # Add text content if present
            if 'content' in message and message['content']:
                content.append({"text": message['content']})

            # Add tool calls if present
            if 'tool_calls' in message:
                for tool_call in message['tool_calls']:
                    function = tool_call.get('function', {})
                    content.append({
                        "toolUse": {
                            "toolUseId": tool_call.get('id', ''),
                            "name": function.get('name', ''),
                            "input": json.loads(function.get('arguments', '{}'))
                        }
                    })

            return {
                "role": "assistant",
                "content": content
            }

        elif role == 'tool':
            return {
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": message.get('tool_call_id', ''),
                        "content": [{"text": message.get('content', '')}]
                    }
                }]
            }

        else:  # user or system
            return {
                "role": role,
                "content": [{"text": message.get('content', '')}]
            }

    async def invoke_openai_text(self, prompt: str, system: Optional[str] = None) -> str:
        """Plain text invocation with no tool config — single user turn, returns the assistant text.

        Useful for lightweight tasks (e.g. tool routing, summarisation) that do not need
        the full MCP tool loop.

        Args:
            prompt: The user message text.
            system:  Optional system prompt string.

        Returns:
            The assistant's response text, or an empty string on failure.
        """
        messages = [{"role": "user", "content": prompt}]

        if system:
            messages.insert(0, {"role": "system", "content": system})

        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"invoke_openai_text failed: {e}")
            return ""

    async def invoke_openai_with_tools(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke OpenAI with MCP tools support.

        Args:
            request: Unified request payload with keys:
                - messages: list of Bedrock conversation messages (will be converted)
            Client-level options used by this method:
                - self.system
                - self.max_iterations
                - self.message_handler

        Returns:
            dict: Structured payload (history/usage/response/error).
        """
        if not isinstance(request, dict):
            return {
                "history": [],
                "usage": None,
                "error": "Invalid request: request must be a dict",
            }

        messages = request.get("messages")

        if not isinstance(messages, list):
            return {
                "history": [],
                "usage": None,
                "error": "Invalid request: 'messages' must be a list",
            }

        if len(messages) == 0:
            return {
                "history": messages,
                "usage": None,
                "error": "Invalid request: at least one message is required",
            }

        # Convert Bedrock messages to OpenAI format
        openai_messages = self._convert_bedrock_messages_to_openai(messages)

        # Add system message if configured
        if self.system:
            openai_messages.insert(0, {"role": "system", "content": self.system})

        # Convert MCP tools to OpenAI format
        openai_tools = self._convert_mcp_tools_to_openai_format()

        usage = {
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0
        }
        return_obj = {
            "history": messages,
            "usage": usage
        }

        if not openai_tools:
            return_obj["error"] = "No MCP tools configured. Tool discovery may have failed."
            return return_obj

        for iteration in range(self.max_iterations):
            try:
                self._emit_progress(self.message_handler, f"Invoking OpenAI (iteration {iteration + 1})", status="LLM Thinking...")

                # Invoke OpenAI
                t0 = time.monotonic()
                response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=openai_messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )
                elapsed_ms = (time.monotonic() - t0) * 1000
                self._emit_progress(
                    self.message_handler,
                    f"OpenAI completed in {elapsed_ms:.0f}ms",
                    status="LLM Response Received",
                )

                # Aggregate usage statistics
                if response.usage:
                    usage["inputTokens"] += response.usage.prompt_tokens or 0
                    usage["outputTokens"] += response.usage.completion_tokens or 0
                    usage["totalTokens"] += response.usage.total_tokens or 0
                return_obj["usage"] = usage

                # Get the assistant's response
                assistant_message = response.choices[0].message

                # Show response text if available
                if assistant_message.content and self.show_response_progress:
                    self._emit_progress(
                        self.message_handler,
                        f"LLM: {assistant_message.content[0:150]}...",
                        status="LLM Response"
                    )

                # Add assistant message to conversation
                openai_messages.append(assistant_message.model_dump(exclude_unset=True))

                # Convert back to Bedrock format and add to history
                bedrock_message = self._convert_openai_message_to_bedrock(assistant_message.model_dump())
                messages.append(bedrock_message)
                return_obj["history"] = messages

                # Check if max iterations reached
                if iteration + 1 >= self.max_iterations:
                    break

                # Check if the assistant wants to use tools
                if assistant_message.tool_calls:
                    tool_results_openai = []
                    tool_results_bedrock = []

                    for tool_call in assistant_message.tool_calls:
                        function = tool_call.function
                        tool_name = function.name
                        tool_input = json.loads(function.arguments)
                        tool_call_id = tool_call.id

                        self._emit_progress(self.message_handler, f"Calling tool: {tool_name}", status="Tool Execution")

                        # Execute the MCP tool call
                        try:
                            tool_result = await self._call_mcp_tool(tool_name, tool_input)
                            result_len = len(str(tool_result))
                            self._emit_progress(self.message_handler, f"Tool {tool_name} returned {result_len} chars", status="Tool Complete")

                            # OpenAI format
                            tool_results_openai.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": str(tool_result)
                            })

                            # Bedrock format
                            tool_results_bedrock.append({
                                "toolResult": {
                                    "toolUseId": tool_call_id,
                                    "content": [{"text": str(tool_result)}]
                                }
                            })
                        except Exception as e:
                            logger.error(f"Error executing MCP tool {tool_name}: {e}")
                            tool_results_openai.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": f"Error: {str(e)}"
                            })
                            tool_results_bedrock.append({
                                "toolResult": {
                                    "toolUseId": tool_call_id,
                                    "content": [{"text": f"Error: {str(e)}"}],
                                    "status": "error"
                                }
                            })

                    # Add tool results to conversations
                    if tool_results_openai:
                        openai_messages.extend(tool_results_openai)
                        messages.append({"role": "user", "content": tool_results_bedrock})
                        return_obj["history"] = messages

                        total_result_chars = sum(len(tr.get('content', '')) for tr in tool_results_openai)
                        self._emit_progress(
                            self.message_handler,
                            f"Sending {len(tool_results_openai)} tool result(s) ({total_result_chars} chars) back to OpenAI...",
                            status="Tool Results",
                        )
                        continue  # Continue the conversation loop

                # If no more tool calls, we're done
                self._emit_progress(self.message_handler, "No more tool calls, preparing final response...", status="Finalizing")
                return_obj["stats"] = {"total_itterations": iteration + 1, "max_itterations": self.max_iterations}

                if assistant_message.content:
                    return_obj["response"] = assistant_message.content
                return return_obj

            except Exception as e:
                logger.error(f"Unexpected error in invoke_openai_with_tools: {e}")
                return_obj["error"] = str(e)
                return return_obj

        # If max iterations reached without completion
        logger.error(f"invoke_openai_with_tools reached maximum iterations: {self.max_iterations}")
        return_obj["error"] = f"Maximum iterations ({self.max_iterations}) reached without completion"
        return return_obj

    async def _call_mcp_tool(
        self,
        toolname: str,
        tool_input: dict,
    ) -> str:
        """Execute MCP tool call via configured callback by default."""
        try:
            call_fn = self.mcp_call
            if call_fn is None:
                raise NotImplementedError(
                    "No MCP tool callback configured. Provide configure_tools(..., tool_handler) "
                    "or override _call_mcp_tool in a subclass."
                )

            result = await call_fn(toolname, tool_input)
            if isinstance(result, dict):
                return json.dumps(result, cls=DateTimeEncoder, indent=2)
            return str(result)
        except Exception as e:
            print(f"Failed MCP {toolname} call: {e}")
            traceback.print_exc()
            raise

    async def generate_embedding(self, text: str) -> list:
        """Generates an embedding for the input text using the given model.

        Args:
            text: Input text to embed.

        Returns:
            list: Embedding vector (list of floats) produced by the model.
        """
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model_id,
                input=text
            )
            # Extract the embedding vector from the response
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def generate_embeddings_batch(self, texts: List[str]) -> List[list]:
        """Generates embeddings for multiple input texts using the given model.

        Args:
            texts: List of input texts to embed.

        Returns:
            List[list]: List of embedding vectors (each is a list of floats) produced by the model.
        """
        if not texts:
            return []

        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model_id,
                input=texts
            )
            # Extract all embedding vectors from the response
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise


class ServerOpenAIClient(OpenAIClient):
    """Server-side OpenAI client with prompt/context input formatting."""

    def _format_invoke_request(
        self,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        request_messages = messages

        if request_messages is None:
            final_prompt = prompt or ""
            if context:
                final_prompt = final_prompt + f"\nUse the following data for Context: {context}"
            request_messages = [{
                "role": "user",
                "content": [{
                    "text": final_prompt
                }]
            }]
        elif prompt or context:
            appended_prompt = prompt or ""
            if context:
                appended_prompt = appended_prompt + f"\nUse the following data for Context: {context}"
            request_messages.append({
                "role": "user",
                "content": [{
                    "text": appended_prompt
                }]
            })

        return {
            "messages": request_messages,
        }

    async def invoke_openai_with_tools(
        self,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        request = self._format_invoke_request(
            prompt=prompt,
            context=context,
            messages=messages,
        )
        return await super().invoke_openai_with_tools(
            request=request,
        )
