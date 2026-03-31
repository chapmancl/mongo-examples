"""
MongoDB AI/LLM client functions 

"""

import datetime
import json
import re
import asyncio
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
import logging
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()  
        return json.JSONEncoder.default(self, obj)


class BedrockClient:
    """
    Bedrock Client for MCP tool calls and LLM invocations
    Handles the LLM interactions and tool integrations for the LLM.
    """
    def __init__(self, settings):
        self.settings = settings
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=self.settings.aws_region,
            config=BotoConfig(
                read_timeout=120,       # seconds to wait for a response chunk
                connect_timeout=10,     # seconds to establish connection
                retries={"max_attempts": 2, "mode": "adaptive"},
            ),
        )
        self.mcp_tools = None
        self.mcp_call = None
        self.llm_setup = False
        # Invoke behavior is configured on the client instance, not per call.
        self.max_iterations = self.settings.LLM_MAX_ITERATIONS
        self.enable_cache_points = getattr(self.settings, "ENABLE_CACHE_POINTS", True)
        self.max_cache_points = 4 # Bedrock has a max of cache points per conversation this was 4, but we can adjust if needed.
        self.system = None
        self.message_handler = None
        self.show_response_progress = True
        
        
    def configure_tools(self, tools_config, tool_handler: Optional[Callable] = None):
        """
        Configure MCP tools for Bedrock client.

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

    def manage_bedrock_cache_points(self, messages: List[Dict[str, Any]], max_cache_points: int = 4) -> int:
        """
        Add cache points to selected messages while respecting Bedrock limits.

        Returns:
            int: Number of cache points added.
        """
        cache_point = {"cachePoint": {"type": "default"}}
        cache_points_added = 0

        # Remove any existing cache points first.
        for message in messages:
            if "content" in message:
                message["content"] = [
                    content
                    for content in message["content"]
                    if "cachePoint" not in content
                ]

        # Add cache points to recent user messages and tool-heavy assistant responses.
        for idx in range(len(messages) - 1, -1, -1):
            if cache_points_added >= max_cache_points:
                break

            message = messages[idx]
            is_recent_user_message = (
                message.get("role") == "user"
                and (len(messages) - idx) % 2 == 1
            )
            is_tool_assistant_message = (
                message.get("role") == "assistant"
                and any("toolUse" in str(content) for content in message.get("content", []))
            )

            if (is_recent_user_message or is_tool_assistant_message) and "content" in message:
                message["content"].append(cache_point.copy())
                cache_points_added += 1

        return cache_points_added

    def _try_parse_json(self, json_string):
        """ try to parse a string to json, return json obj or nothing """
        # I hate using try/catch as logic, but here we are. is there a better way?
        try:
            # Remove markdown code fences
            json_string = re.sub(r'^```json\s*', '', json_string)
            json_string = re.sub(r'^```\s*', '', json_string)
            json_string = re.sub(r'\s*```$', '', json_string)

            # Find the JSON object (between first { and last })
            start_idx = json_string.find('{')
            end_idx = json_string.rfind('}')

            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                #logger.info("No valid JSON object found in response")
                return None
            else:
                json_string = json_string[start_idx:end_idx+1]
                json_string = json_string.strip()
                data = json.loads(json_string)
                return data
        except json.JSONDecodeError as e:
            return None
        except ValueError as e:
            return None
        return None

    async def invoke_bedrock_text(self, prompt: str, system: Optional[str] = None) -> str:
        """Plain text invocation with no tool config — single user turn, returns the assistant text.

        Useful for lightweight tasks (e.g. tool routing, summarisation) that do not need
        the full MCP tool loop.  Uses asyncio.to_thread so it is safe to await from async
        code without blocking the event loop.

        Args:
            prompt: The user message text.
            system:  Optional system prompt string.

        Returns:
            The assistant's response text, or an empty string on failure.
        """
        converse_input: Dict[str, Any] = {
            "modelId": self.settings.LLM_MODEL_ID,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
        }
        try:
            response = await asyncio.to_thread(self.bedrock_client.converse, **converse_input)
            text = ""
            for block in response.get("output", {}).get("message", {}).get("content", []):
                if "text" in block:
                    text += block["text"]
            return text
        except Exception as e:
            logger.warning(f"invoke_bedrock_text failed: {e}")
            return ""

    # Keep this method as the core Bedrock execution path.
    # It accepts a unified request payload so each subclass can own
    # prompt/context/history formatting for its own call surface.
    async def invoke_bedrock_with_tools(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke Bedrock with MCP tools support and optional cache points.

        Args:
            request: Unified Bedrock request payload with keys:
                - messages: list of Bedrock conversation messages
            Client-level options used by this method:
                - self.system
                - self.max_iterations
                - self.message_handler
                - self.enable_cache_points
                - self.max_cache_points

        Returns:
            dict: Structured payload (history/usage/response/error).
        """
        if not isinstance(request, dict):
            return {
                "history": [],
                "usage": None,
                "error": "Invalid Bedrock request: request must be a dict",
            }

        messages = request.get("messages")        
        
        if not isinstance(messages, list):
            return {
                "history": [],
                "usage": None,
                "error": "Invalid Bedrock request: 'messages' must be a list",
            }

        if len(messages) == 0:
            return {
                "history": messages,
                "usage": None,
                "error": "Invalid Bedrock request: at least one message is required",
            }

        if self.enable_cache_points:
            cache_points_added = self.manage_bedrock_cache_points(messages, max_cache_points=self.max_cache_points)
            if cache_points_added > 0:
                self._emit_progress(
                    self.message_handler,
                    f"Added {cache_points_added} cache points to conversation",
                    status="Processing"
                )
        
        # Tool configuration for Bedrock
        tool_config = {"tools": self.mcp_tools} if self.mcp_tools else None

        usage = None
        return_obj = {
            "history": messages,
            "usage": usage
        }

        if tool_config is None:
            return_obj["error"] = "No MCP tools configured. Tool discovery may have failed."
            return return_obj

        # subtract 1 or else we would end on a tool response        
        for iteration in range(self.max_iterations):
            try:
                self._emit_progress(self.message_handler, f"Invoking Bedrock (iteration {iteration + 1})", status="LLM Thinking...")

                # Invoke Bedrock using the Converse API
                converse_input = {
                    "modelId": self.settings.LLM_MODEL_ID,
                    "messages": messages,
                    "toolConfig": tool_config,
                }
                if self.system is not None:
                    converse_input["system"] = self.system

                t0 = time.monotonic()
                response = self.bedrock_client.converse(**converse_input)
                elapsed_ms = (time.monotonic() - t0) * 1000
                self._emit_progress(
                    self.message_handler,
                    f"Bedrock completed in {elapsed_ms:.0f}ms",
                    status="LLM Response Received",
                )

                # Aggregate usage statistics
                itt_used = response['usage']
                if usage is None: 
                    usage = itt_used
                else:
                    for k,v in itt_used.items(): usage[k] += v
                return_obj["usage"] = usage

                # Get the assistant's response
                assistant_message = response['output']['message']                
                if assistant_message.get("content"):
                    for content in assistant_message["content"]:
                        if content.get("text"):
                            if self.show_response_progress:
                                self._emit_progress(
                                    self.message_handler,
                                    f"LLM: {content['text'][0:150]}...",
                                    status="LLM Response"
                                )
                            break

                messages.append(assistant_message)
                return_obj["history"] = messages

                # if this is the final itteration, return what we have, but don't do the tool call.
                # just think it makes sense to end after the last LLM response
                if iteration + 1 >= self.max_iterations:
                    break
                
                # Check if the assistant wants to use tools
                if 'content' in assistant_message:
                    tool_calls = []

                    for content in assistant_message['content']:
                        if 'toolUse' in content:
                            tool_calls.append(content['toolUse'])
                        # don't care about the text content right now its already recorded
                        #elif 'text' in content:
                        #    text_content.append(content['text'])
                    
                    # If there are tool calls, execute them
                    if tool_calls:
                        tool_results = []
                        
                        for tool_req in tool_calls:
                            tool_name = tool_req['name']
                            tool_input = tool_req['input']
                            tool_use_id = tool_req['toolUseId']
                            self._emit_progress(self.message_handler, f"Calling tool: {tool_name}", status="Tool Execution")
                            # Execute the MCP tool call (with caching)
                            try:
                                tool_result = await self._call_mcp_tool(tool_name, tool_input)
                                result_len = len(str(tool_result))
                                self._emit_progress(self.message_handler, f"Tool {tool_name} returned {result_len} chars", status="Tool Complete")
                                tool_results.append({
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": str(tool_result)}]
                                    }
                                })
                            except Exception as e:
                                # don't fail here, the LLM can usually find a work around
                                # just log it and keep going
                                logger.error(f"Error executing MCP tool {tool_name}: {e}")
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
                            return_obj["history"] = messages
                            total_result_chars = sum(len(str(tr)) for tr in tool_results)
                            self._emit_progress(
                                self.message_handler,
                                f"Sending {len(tool_results)} tool result(s) ({total_result_chars} chars) back to Bedrock...",
                                status="Tool Results",
                            )
                            continue  # Continue the conversation loop
                    
                    # If no more tool calls, then we're done and return the response
                    self._emit_progress(self.message_handler, "No more tool calls, preparing final response...", status="Finalizing")
                    return_obj["stats"] = {"total_itterations": iteration + 1, "max_itterations": self.max_iterations}     
                    # Always pass the raw assistant text through unchanged.
                    # JSON extraction is handled downstream via [JSON_DATA_START]
                    # tags only — no brace-counting.
                    if len(messages) > 0 and messages[-1]["role"] == "assistant":
                        msg = messages[-1]["content"][0]["text"]
                        return_obj["response"] = msg
                    return return_obj
                
                # If we get here, there was no content to process
                return_obj["error"] = "No response generated"
                return return_obj
                
            except ClientError as error:
                error_code = error.response['Error']['Code']
                logger.error(f"Bedrock error: {error_code} - {error.response['Error']['Message']}")
                if error_code == 'ValidationException':
                    return_obj["error"] = f"Input validation failed {error.response['Error']['Message']}"
                elif error_code in ['ExpiredTokenException', 'ExpiredToken']:
                    raise Exception("credentials have expired", error)
                else:
                    return_obj["error"] = error.response['Error']['Message']
                return return_obj
            except Exception as e:
                logger.error(f"Unexpected error in invoke_bedrock_with_tools: {e}")
                return_obj["error"] = str(e)
                return return_obj

        # If max iterations reached without completion
        logger.error(f"invoke_bedrock_with_tools reached maximum iterations: {self.max_iterations}")
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
        body = json.dumps({"inputText": text})        
        # Invoke the Bedrock embedding model (e.g., Titan Embeddings) specified in config
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.bedrock_client.invoke_model(
                modelId= self.settings.EMBEDDING_MODEL_ID, #"amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="application/json",
                body=body
            )
        )
        # Parse the response and extract the embedding vector
        return json.loads(response["body"].read())["embedding"]


class ServerBedrockClient(BedrockClient):
    """Server-side Bedrock client with prompt/context input formatting."""

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

    async def invoke_bedrock_with_tools(
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
        return await super().invoke_bedrock_with_tools(
            request=request,
        )
