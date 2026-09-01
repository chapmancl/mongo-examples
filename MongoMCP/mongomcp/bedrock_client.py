"""
MongoDB AI/LLM client functions 

"""

import datetime
import json
import os
import re
import asyncio
import hashlib
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
import logging
import httpx
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from bson.binary import Binary, BinaryVectorDtype

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Known Bedrock Converse output-token ceilings (inferenceConfig.maxTokens) by model family.
# Passing a value above the model's ceiling makes Converse reject the request, so the
# configured LLM_MAX_OUTPUT_TOKENS is clamped to the matching entry below.
_MODEL_MAX_OUTPUT_TOKENS = {
    "opus-4-8": 128000,
    "opus-4-7": 128000,
    "opus-4-6": 128000,
    "opus-5": 128000,
    "sonnet-4-6": 128000,
    "sonnet-5": 128000,
    "fable-5": 128000,
    "opus-4-5": 64000,
    "sonnet-4-5": 64000,
    "haiku-4-5": 64000,
    "opus-4-1": 32000,
}
# Conservative fallback when the model id matches no known family (safe for all Claude models).
_DEFAULT_MODEL_MAX_OUTPUT_TOKENS = 8192


def _resolve_max_output_tokens(model_id: Optional[str], configured: Optional[int]) -> int:
    """Return the per-turn maxTokens to send, clamped to the model's known ceiling.

    ``configured`` is the operator's requested cap (LLM_MAX_OUTPUT_TOKENS). It is clamped
    down to the ceiling for ``model_id``'s family so an over-large value can never trigger
    a Bedrock ValidationException. Falls back to 8192 when the value or model is unknown.
    """
    try:
        want = int(configured) if configured else _DEFAULT_MODEL_MAX_OUTPUT_TOKENS
    except (TypeError, ValueError):
        want = _DEFAULT_MODEL_MAX_OUTPUT_TOKENS
    if want <= 0:
        want = _DEFAULT_MODEL_MAX_OUTPUT_TOKENS
    mid = (model_id or "").lower()
    ceiling = next((cap for key, cap in _MODEL_MAX_OUTPUT_TOKENS.items() if key in mid), None)
    if ceiling is None:
        return min(want, _DEFAULT_MODEL_MAX_OUTPUT_TOKENS)
    return min(want, ceiling)

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

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Rough text-to-token estimate used for overflow preflight."""
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)

    def _estimate_content_tokens(self, content: Any) -> int:
        """Estimate token footprint of a Bedrock content payload."""
        if isinstance(content, str):
            return self._estimate_text_tokens(content)
        if isinstance(content, list):
            return sum(self._estimate_content_tokens(item) for item in content)
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return self._estimate_text_tokens(text)
            return self._estimate_text_tokens(json.dumps(content, ensure_ascii=False, default=str))
        return self._estimate_text_tokens(str(content))

    def _estimate_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate tokens for all current conversation messages."""
        total = 0
        for msg in messages or []:
            total += 6  # per-message overhead
            total += self._estimate_content_tokens(msg.get("content", []))
        return total

    def _estimate_system_tokens(self) -> int:
        """Estimate tokens for current system prompt blocks."""
        return self._estimate_content_tokens(self.system or [])

    def _estimate_total_context_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total context size for the next model call."""
        return self._estimate_system_tokens() + self._estimate_messages_tokens(messages)

    def _cache_min_prefix_tokens(self) -> int:
        """Minimum cacheable-prefix size for the current model. Anthropic on Bedrock caches
        only prefixes at/above this length (1,024 for Opus/Sonnet, 2,048 for Haiku); a
        cachePoint on a shorter prefix is silently ignored."""
        mid = (getattr(self.settings, "LLM_MODEL_ID", "") or "").lower()
        return 2048 if "haiku" in mid else 1024

    @staticmethod
    def _tool_overflow_notice(tool_name: str, estimated_added_tokens: int, current_tokens: int, max_tokens: int) -> str:
        """Return overflow-safe tool message with paging guidance."""
        return (
            f"Tool '{tool_name}' executed successfully, but the full result was omitted because it would overflow "
            f"the model context (current~{current_tokens}, add~{estimated_added_tokens}, max={max_tokens}). "
            "Please rework the request to page results into smaller chunks and use the memory layer to store and "
            "retrieve those chunks across turns."
        )

    def manage_bedrock_cache_points(self, messages: List[Dict[str, Any]], max_cache_points: int = 4) -> int:
        """Checkpoint the stable conversation-history prefix for Bedrock prompt caching.

        The system prompt and tool schemas are checkpointed separately (and unconditionally)
        in invoke_bedrock_with_tools, so they are ALWAYS cached. This method adds at most one
        checkpoint on the message history, and only once that history is large enough to be
        worth its own cache write — new/short conversations get none (they still benefit from
        the always-on system+tools cache).

        The checkpoint goes on the LAST message (the current user turn). Its content is fixed
        for this request (the query text doesn't change, and we only ever APPEND assistant/
        tool messages after it), so the whole prior context is reused from cache next turn and
        across this turn's tool-loop iterations.

        Returns the number of history cache points added.
        """
        # Drop any cache points carried over in resent history — they move forward each turn.
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [
                    c for c in content
                    if not (isinstance(c, dict) and "cachePoint" in c)
                ]

        if max_cache_points <= 0 or not messages:
            return 0

        # Gate: skip the history checkpoint until the message history exceeds the model's
        # minimum cacheable size, so short conversations don't spend a (largely redundant)
        # cache write on top of the system+tools checkpoint.
        if self._estimate_messages_tokens(messages) < self._cache_min_prefix_tokens():
            return 0

        last_content = messages[-1].get("content")
        if isinstance(last_content, list):
            last_content.append({"cachePoint": {"type": "default"}})
            return 1
        return 0

    @staticmethod
    def _deserialize_stringified_arrays(tool_input: dict) -> dict:
        """Normalize tool inputs where Claude has stringified array/object values.

        Claude occasionally encodes array or object parameters as JSON strings
        (e.g. entities='["X","Y"]' instead of entities=["X","Y"]). This pass
        detects any string value that parses as a JSON array or object and
        replaces it with the parsed value, so downstream handlers always receive
        the correct native type.
        """
        if not isinstance(tool_input, dict):
            return tool_input
        result = {}
        for k, v in tool_input.items():
            if isinstance(v, str) and len(v) >= 2 and v[0] in ('[', '{'):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, (list, dict)):
                        result[k] = parsed
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            result[k] = v
        return result

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
            # system + tools are always checkpointed below; this count is the history point.
            self._emit_progress(
                self.message_handler,
                f"Prompt cache: system + tools checkpointed; history checkpoints: {cache_points_added}",
                status="Processing"
            )
        
        # Tool configuration for Bedrock
        tool_config = {"tools": self.mcp_tools} if self.mcp_tools else None

        usage = None
        # Per-tool execution times (toolUseId -> ms) accumulated across every tool
        # iteration of this turn. Surfaced to the UI (activity-feed sum + per-tool ms
        # in the tool-response bubble); never injected into the model payload.
        tool_timings: Dict[str, int] = {}
        # Wall-clock time spent inside tool-execution batches this turn. Tools within a
        # batch run concurrently, so this is <= the sum of per-tool times.
        turn_wall_ms = 0
        return_obj = {
            "history": messages,
            "usage": usage,
            "tool_timings": tool_timings,
        }

        if tool_config is None:
            return_obj["error"] = "No MCP tools configured. Tool discovery may have failed."
            return return_obj

        # subtract 1 or else we would end on a tool response        
        _iter_warning_injected = False
        for iteration in range(self.max_iterations):
            try:
                # Warn the LLM when only 5 iterations remain so it can wrap up.
                iterations_remaining = self.max_iterations - iteration
                if iterations_remaining <= 5 and not _iter_warning_injected:
                    _iter_warning_injected = True
                    self._emit_progress(
                        self.message_handler,
                        f"Iteration limit warning: {iterations_remaining} of {self.max_iterations} iterations remaining",
                        status="Iteration Warning",
                    )
                    messages.append({
                        "role": "user",
                        "content": [{
                            "text": (
                                f"\n\n[Iteration Warning] You have {iterations_remaining} of "
                                f"{self.max_iterations} LLM iterations remaining in this request. "
                                "You must finish within these iterations. "
                                "Stop calling tools, save any important state to memory now using "
                                "the memory intake tool, and return a concise summary response to "
                                "the user so they can continue in a new request if needed."
                            )
                        }]
                    })

                self._emit_progress(self.message_handler, f"Invoking Bedrock (iteration {iteration + 1})", status="LLM Thinking...")

                # Recompute tool_config each iteration so tools activated mid-conversation
                # (e.g. via activate_data_domain) become callable on the next model turn.
                # Append a cachePoint after the tool schemas so the (large, mostly-stable)
                # system+tools prefix is cached every request and across this turn's loop.
                # When a data domain toggles the tool list, only this checkpoint re-warms;
                # the system checkpoint (placed before tools in the prefix) still hits.
                if self.mcp_tools:
                    _tools_blocks = list(self.mcp_tools)
                    if self.enable_cache_points:
                        _tools_blocks.append({"cachePoint": {"type": "default"}})
                    tool_config = {"tools": _tools_blocks}

                # Invoke Bedrock using the Converse API
                converse_input = {
                    "modelId": self.settings.LLM_MODEL_ID,
                    "messages": messages,
                    "toolConfig": tool_config,
                    # Cap output generously so a turn emitting SEVERAL large tool calls (e.g. a
                    # fan-out of agent_run_prompt with big prompts) isn't truncated mid-last-block.
                    # Without this, converse used the model default and the final toolUse's input
                    # JSON got cut off -> that sub-agent launched with missing required args.
                    # _resolve_max_output_tokens clamps the configured value to the model ceiling.
                    "inferenceConfig": {
                        "maxTokens": _resolve_max_output_tokens(
                            self.settings.LLM_MODEL_ID,
                            getattr(self.settings, "LLM_MAX_OUTPUT_TOKENS", None)
                            or os.environ.get("LLM_MAX_OUTPUT_TOKENS"),
                        ),
                    },
                }
                if self.system is not None:
                    # Checkpoint the end of the system prompt so it's ALWAYS cached (it's large
                    # and byte-identical every request). Placed before tools in the prefix, so
                    # it keeps hitting even when a data-domain toggle changes the tool list.
                    _sys_blocks = list(self.system)
                    if self.enable_cache_points and _sys_blocks:
                        _sys_blocks.append({"cachePoint": {"type": "default"}})
                    converse_input["system"] = _sys_blocks
                else:
                    if iteration == 0:
                        logger.warning("invoke_bedrock_with_tools: NO system prompt set on this client")

                t0 = time.monotonic()
                response = self.bedrock_client.converse(**converse_input)
                elapsed_ms = (time.monotonic() - t0) * 1000
                self._emit_progress(
                    self.message_handler,
                    f"Bedrock completed in {elapsed_ms / 1000:.3f}s",
                    status="LLM Response Received",
                )

                # Aggregate usage statistics
                itt_used = response['usage']
                if usage is None: 
                    usage = itt_used
                else:
                    for k,v in itt_used.items(): usage[k] += v
                return_obj["usage"] = usage
                #return_obj["usage_last"] = itt_used

                # Get the assistant's response
                assistant_message = response['output']['message']                
                # A truncated turn (stopReason=max_tokens) can leave the LAST toolUse block's
                # input incomplete -> that sub-agent fails validation. Surface it clearly; the
                # maxTokens bump above is the primary prevention.
                if response.get("stopReason") == "max_tokens":
                    logger.warning(
                        "Bedrock stopReason=max_tokens (iteration %d) — the final tool call may be "
                        "truncated/incomplete; raise LLM_MAX_OUTPUT_TOKENS or emit fewer parallel calls.",
                        iteration + 1,
                    )
                    self._emit_progress(
                        self.message_handler,
                        "Response hit the max output-token limit — the last tool call may be truncated. "
                        "Emit fewer/smaller parallel tool calls, or raise LLM_MAX_OUTPUT_TOKENS.",
                        status="Max Tokens Truncation",
                    )
                if assistant_message.get("content"):
                    for content in assistant_message["content"]:
                        if content.get("text"):
                            if self.show_response_progress:
                                self._emit_progress(
                                    self.message_handler,
                                    content['text'],
                                    status="LLM Reasoning"
                                )
                            break

                messages.append(assistant_message)
                return_obj["history"] = messages

                # if this is the final itteration, return what we have, but don't do the tool call.
                # just think it makes sense to end after the last LLM response
                if iteration + 1 >= self.max_iterations:
                    # The model may have asked to call a tool on this final turn, but we
                    # will NOT execute it. Persisting an unanswered toolUse block makes the
                    # NEXT request fail with "Expected toolResult blocks ...", so strip any
                    # dangling toolUse from the trailing assistant message before returning.
                    self._strip_dangling_tool_use_from_last(messages)
                    return_obj["history"] = messages
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
                        max_context_tokens = int(getattr(self.settings, "LLM_MAX_CONTEXT_TOKENS", 200000))
                        current_usage_tokens = int((itt_used or {}).get("inputTokens", 0)) + int((itt_used or {}).get("outputTokens", 0))
                        estimated_context_tokens = self._estimate_total_context_tokens(messages)
                        context_baseline_tokens = max(current_usage_tokens, estimated_context_tokens)

                        # Announce all pending calls upfront, then dispatch concurrently.
                        for tool_req in tool_calls:
                            self._emit_progress(self.message_handler, f"Calling tool: {tool_req['name']}", status="Tool Execution")

                        async def _exec_one(tool_req):
                            tool_name = tool_req['name']
                            tool_use_id = tool_req['toolUseId']
                            t0 = time.monotonic()
                            try:
                                tool_input = self._deserialize_stringified_arrays(tool_req['input'])
                                result = await self._call_mcp_tool(tool_name, tool_input)
                                result_text = str(result)
                                elapsed_ms = int((time.monotonic() - t0) * 1000)
                                self._emit_progress(
                                    self.message_handler,
                                    f"Tool {tool_name} returned {len(result_text)} chars in {elapsed_ms} ms",
                                    status="Tool Complete",
                                )
                                return tool_use_id, tool_name, result_text, None, elapsed_ms
                            except Exception as e:
                                elapsed_ms = int((time.monotonic() - t0) * 1000)
                                logger.error(f"Error executing MCP tool {tool_name}: {e}")
                                return tool_use_id, tool_name, None, e, elapsed_ms

                        batch_t0 = time.monotonic()
                        raw_results = await asyncio.gather(
                            *[_exec_one(tr) for tr in tool_calls],
                            return_exceptions=True,
                        )
                        batch_wall_ms = int((time.monotonic() - batch_t0) * 1000)

                        # Post-pass: token-overflow check (pure arithmetic, no I/O).
                        # return_exceptions=True means BaseException instances can appear
                        # as result values — handle them so no toolUseId is ever missing.
                        tool_results = []
                        projected_additional_tokens = 0
                        for i, item in enumerate(raw_results):
                            if isinstance(item, BaseException):
                                tool_use_id = tool_calls[i]['toolUseId']
                                tool_name = tool_calls[i]['name']
                                logger.error(f"Unhandled error in parallel tool {tool_name}: {item}")
                                tool_results.append({
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": f"Error: {str(item)}"}],
                                        "status": "error",
                                    }
                                })
                                continue
                            tool_use_id, tool_name, tool_result_text, exc, elapsed_ms = item
                            tool_timings[tool_use_id] = elapsed_ms
                            if exc is not None:
                                tool_results.append({
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": f"Error: {str(exc)}"}],
                                        "status": "error",
                                    }
                                })
                                continue
                            candidate_block = {
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": tool_result_text}],
                                }
                            }
                            added_tokens = 6 + self._estimate_content_tokens(candidate_block)
                            projected_total_tokens = context_baseline_tokens + projected_additional_tokens + added_tokens

                            # If full tool output would overflow model context, send a compact notice instead.
                            if projected_total_tokens > max_context_tokens:
                                self._emit_progress(
                                    self.message_handler,
                                    f"Tool result for {tool_name} would overflow context; sending overflow notice only",
                                    status="Tool Overflow",
                                )
                                tool_result_text = self._tool_overflow_notice(
                                    tool_name=tool_name,
                                    estimated_added_tokens=added_tokens,
                                    current_tokens=context_baseline_tokens + projected_additional_tokens,
                                    max_tokens=max_context_tokens,
                                )
                                candidate_block = {
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": tool_result_text}],
                                    }
                                }
                                added_tokens = 6 + self._estimate_content_tokens(candidate_block)

                            projected_additional_tokens += added_tokens
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": tool_result_text}],
                                }
                            })
                                
                        
                        # Add tool results to the conversation
                        if tool_results:
                            tool_message = {"role": "user", "content": tool_results}
                            messages.append(tool_message)              
                            return_obj["history"] = messages
                            total_result_chars = sum(len(str(tr)) for tr in tool_results)
                            tool_block_context_tokens = self._estimate_total_context_tokens(messages)
                            tool_block_percent = (tool_block_context_tokens / max(1, max_context_tokens)) * 100
                            tool_block_input = int((itt_used or {}).get("inputTokens", 0) or 0)
                            tool_block_output = int((itt_used or {}).get("outputTokens", 0) or 0)
                            tool_block_total = tool_block_input + tool_block_output
                            self._emit_progress(
                                self.message_handler,
                                f"Sending {len(tool_results)} tool result(s) ({total_result_chars} chars) back to Bedrock...",
                                status="Tool Results",
                            )
                            self._emit_progress(
                                self.message_handler,
                                (
                                    "Token usage after tool block - "
                                    f"input: {tool_block_input}, output: {tool_block_output}, total: {tool_block_total}, "
                                    f"context_estimate: {tool_block_context_tokens}, used: {tool_block_percent:.1f}%"
                                ),
                                status="Token Usage",
                            )
                            turn_wall_ms += batch_wall_ms
                            self._emit_progress(
                                self.message_handler,
                                f"Tool time (wall clock) — this step: {batch_wall_ms} ms across {len(tool_results)} call(s); turn total: {turn_wall_ms} ms",
                                status="Tool Timing",
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
                error_msg = error.response['Error']['Message']
                logger.error(f"Bedrock error: {error_code} - {error_msg}")
                if error_code == 'ValidationException':
                    # Attempt to repair the conversation history in place before giving up.
                    # For missing/mismatched toolResult ids, try targeted id-injection first;
                    # then fall back to full structural canonicalisation, which enforces
                    # toolUse/toolResult adjacency AND drops empty-content messages that
                    # Bedrock rejects ("The content field ... is empty").
                    repaired = 0
                    if 'toolResult' in error_msg or 'toolUse' in error_msg:
                        repaired = self._repair_missing_tool_results(messages, error_msg)
                    if not repaired:
                        repaired = self._canonicalize_tool_cycles(messages)
                    if repaired:
                        logger.warning(
                            "Repaired %d history block(s); retrying iteration %d",
                            repaired, iteration + 1,
                        )
                        continue  # retry this iteration with patched history
                    # Nothing repairable. Clear history for structural corruption; otherwise
                    # surface the validation error verbatim.
                    if ('toolResult' in error_msg or 'toolUse' in error_msg
                            or 'content field' in error_msg):
                        logger.error("Conversation history is corrupt and could not be repaired. Clearing history.")
                        return_obj["history"] = messages
                        return_obj["clear_history"] = True
                        return_obj["error"] = "Conversation history was corrupt and has been cleared. Please retry your question."
                    else:
                        return_obj["error"] = f"Input validation failed {error_msg}"
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
        

    def _repair_missing_tool_results(self, messages: list, error_msg: str) -> int:
        """
        Inject synthetic toolResult blocks for any toolUseIds that Bedrock reports
        as missing a result.

        Strategy:
        1. Parse the orphaned IDs from Bedrock's error message.
        2. Skip IDs that already have a toolResult anywhere in history.
        3. For each orphaned ID, find the assistant message that contains the
           matching toolUse block and insert a user/toolResult message immediately
           after it (positional repair). This handles the messages.0 case where
           appending at the end would not satisfy Bedrock's ordering requirement.
        4. Fall back to appending at the end for any IDs whose assistant message
           cannot be located.

        Returns the number of synthetic blocks injected (0 = could not repair).
        """
        import re as _re

        named_ids = set(_re.findall(r'tooluse_[A-Za-z0-9]+', error_msg))
        if not named_ids:
            return 0

        # Collect IDs that already have a toolResult in history.
        answered: set = set()
        for msg in messages:
            if msg.get('role') != 'user':
                continue
            for block in msg.get('content', []):
                if 'toolResult' in block:
                    answered.add(block['toolResult'].get('toolUseId', ''))

        target_ids = named_ids - answered
        if not target_ids:
            return 0

        def _synthetic(tid: str) -> dict:
            return {
                "toolResult": {
                    "toolUseId": tid,
                    "content": [{"text": "Tool execution failed or result was lost. Please proceed without this result."}],
                    "status": "error",
                }
            }

        injected = 0
        remaining = set(target_ids)

        # Pass 1: positional repair — insert result message right after the
        # assistant message that owns the orphaned toolUse block.
        i = 0
        while i < len(messages) and remaining:
            msg = messages[i]
            if msg.get('role') == 'assistant':
                owned = [
                    b['toolUse']['toolUseId']
                    for b in msg.get('content', [])
                    if isinstance(b, dict) and 'toolUse' in b
                    and b['toolUse'].get('toolUseId') in remaining
                ]
                if owned:
                    synthetic_blocks = [_synthetic(tid) for tid in owned]
                    # Insert a user/toolResult turn right after this assistant message.
                    next_msg = messages[i + 1] if i + 1 < len(messages) else None
                    if (next_msg and next_msg.get('role') == 'user'
                            and isinstance(next_msg.get('content'), list)
                            and all(isinstance(b, dict) and 'toolResult' in b
                                    for b in next_msg['content'])):
                        # Merge into the existing tool-result turn.
                        next_msg['content'].extend(synthetic_blocks)
                    else:
                        messages.insert(i + 1, {'role': 'user', 'content': synthetic_blocks})
                        i += 1  # skip past the just-inserted message
                    for tid in owned:
                        remaining.discard(tid)
                    injected += len(owned)
            i += 1

        # Pass 2: fallback — append remaining IDs that had no matching assistant message.
        if remaining:
            synthetic_blocks = [_synthetic(tid) for tid in remaining]
            if messages and messages[-1].get('role') == 'user':
                existing = messages[-1].get('content', [])
                if all(isinstance(b, dict) and 'toolResult' in b for b in existing):
                    messages[-1]['content'].extend(synthetic_blocks)
                else:
                    messages.append({'role': 'user', 'content': synthetic_blocks})
            else:
                messages.append({'role': 'user', 'content': synthetic_blocks})
            injected += len(synthetic_blocks)

        return injected


    @staticmethod
    def _strip_dangling_tool_use_from_last(messages: list) -> None:
        """Remove unanswered toolUse blocks from the trailing assistant message.

        Called when the tool loop stops (e.g. max iterations) right after the model
        asked to use a tool we will not execute. Persisting that toolUse with no
        following toolResult makes Bedrock reject the whole history on the next
        request ("Expected toolResult blocks ...").
        """
        if not messages:
            return
        last = messages[-1]
        if last.get("role") != "assistant":
            return
        content = last.get("content")
        if not isinstance(content, list):
            return
        kept = [b for b in content if not (isinstance(b, dict) and "toolUse" in b)]
        if len(kept) == len(content):
            return  # nothing dangling
        if not any(isinstance(b, dict) and "text" in b for b in kept):
            kept.append({"text": (
                "I reached the tool-call limit for this request before finishing. "
                "Please continue in a new message."
            )})
        last["content"] = kept

    @staticmethod
    def _canonicalize_tool_cycles(messages: list) -> int:
        """Enforce Bedrock's toolUse/toolResult adjacency invariant, in place.

        Bedrock requires every assistant toolUse block to be answered by a
        toolResult with the same id in the IMMEDIATELY following user message, and
        every toolResult to correspond to a toolUse in the immediately preceding
        assistant message. Trimming or interrupted turns can break this even when
        a matching block exists elsewhere in history. This pass:
          * strips assistant toolUse blocks not answered by the next message,
          * strips user toolResult blocks not requested by the previous message,
          * drops messages left with no content.
        Returns the number of blocks removed (0 = already canonical).
        """
        removed = 0
        n = len(messages)
        for i, msg in enumerate(messages):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            role = msg.get("role")
            if role == "assistant":
                next_ids = set()
                if i + 1 < n:
                    nxt = messages[i + 1]
                    if nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
                        next_ids = {
                            b["toolResult"].get("toolUseId")
                            for b in nxt["content"]
                            if isinstance(b, dict) and isinstance(b.get("toolResult"), dict)
                        }
                kept = []
                for b in content:
                    if isinstance(b, dict) and "toolUse" in b:
                        tid = (b.get("toolUse") or {}).get("toolUseId")
                        if tid not in next_ids:
                            removed += 1
                            continue
                    kept.append(b)
                msg["content"] = kept
            elif role == "user":
                prev_ids = set()
                if i - 1 >= 0:
                    prv = messages[i - 1]
                    if prv.get("role") == "assistant" and isinstance(prv.get("content"), list):
                        prev_ids = {
                            b["toolUse"].get("toolUseId")
                            for b in prv["content"]
                            if isinstance(b, dict) and isinstance(b.get("toolUse"), dict)
                        }
                kept = []
                for b in content:
                    if isinstance(b, dict) and "toolResult" in b:
                        tid = (b.get("toolResult") or {}).get("toolUseId")
                        if tid not in prev_ids:
                            removed += 1
                            continue
                    kept.append(b)
                msg["content"] = kept
        # Always drop messages left with no content blocks (empty list, None, or a
        # non-list). Bedrock rejects these with "The content field in the Message
        # object at messages.N is empty."
        before = len(messages)
        messages[:] = [
            m for m in messages
            if isinstance(m.get("content"), list) and m.get("content")
        ]
        removed += before - len(messages)
        return removed

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

    async def generate_embedding(self, text: str, model_id: Optional[str] = None, is_query: bool = True) -> list:
        """Generates an embedding for the input text using the given model.
        
        Args:
            text: Input text to embed.
            model_id: The ID of the model to use. Defaults to self.settings.EMBEDDING_MODEL_ID.
                      Routes to Voyage AI for "voyage-*" models, Bedrock for "amazon.*" models.
            is_query: True (default) embeds as a QUERY (search text); pass False for stored
                      DOCUMENTS (memory/strategy/schema content) so Voyage uses input_type=
                      "document" for better asymmetric retrieval. Ignored for Bedrock models.
        
        Returns:
            list: Embedding vector (list of floats) produced by the model.
        """
        if model_id is None:
            model_id = self.settings.EMBEDDING_MODEL_ID

        # Resolve the effective model up front so the cache key is identical between the
        # cache-check and the cache-store, and across the query/document + voyage/bedrock
        # paths (query embeds resolve to QUERY_EMBEDDING_MODEL_ID). Keying by model means two
        # different embedding models never collide on the same text and return each other's
        # vectors.
        if model_id.startswith("voyage-"):
            effective_model = self._resolve_voyage_model(model_id, is_query)
        else:
            effective_model = model_id

        # Cache QUERY embeddings only (deterministic + high reuse on the search path).
        # Document embeds are one-off stored content — low reuse.
        use_cache = is_query and self._cache_enabled("EMBED_CACHE")
        if use_cache:
            cached = await self._embed_cache_get(effective_model, text)
            if cached is not None:
                return cached

        if model_id.startswith("voyage-"):
            enabled, _linger_s, _max_batch = self._embed_cfg()
            if enabled:
                # Coalesce concurrent single embeds into ONE batched Voyage call.
                result = await self._coalesced_embed(text, model_id, is_query=is_query)
            else:
                result = await self.generate_voyage_embeddings(text, model_id=model_id, is_query=is_query)
        else:
            logger.debug(f"Generating embedding using Bedrock model {model_id} for input text of length {len(text)}")
            # amazon.* — use Bedrock. Titan returns the vector at the TOP level of the response
            # body ({"embedding": [...], "inputTextTokenCount": N}), NOT nested under data[0].
            body = json.dumps({"inputText": text})
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.bedrock_client.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body
                )
            )
            data = json.loads(response["body"].read())
            result = {
                "embedding_model": model_id,
                "vector": data["embedding"],
            }

        if use_cache and isinstance(result, dict) and result.get("vector"):
            # Embeddings are immutable for a given (model, text) — safe to cache long.
            # Default 30 days; override via EMBED_CACHE_TTL (seconds).
            await self._embed_cache_put(effective_model, text, result, self._cache_ttl("EMBED_CACHE_TTL", 2592000))
        return result


    def _get_voyage_client(self) -> "httpx.AsyncClient":
        """Lazily create a SHARED, pooled httpx client for Voyage embedding/rerank calls.

        Reused across all calls on this instance (memory intakes funnel through the shared
        llm_client) so we don't open a fresh TLS connection per embedding — that per-call
        client churn caused ConnectTimeout storms under sub-agent fan-out. Connection reuse +
        a bounded pool + a generous connect timeout make concurrent embeds resilient.
        """
        c = getattr(self, "_voyage_http", None)
        if c is None or c.is_closed:
            limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
            timeout = httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0)
            c = self._voyage_http = httpx.AsyncClient(limits=limits, timeout=timeout)
        return c

    def _embed_semaphore(self) -> "asyncio.Semaphore":
        """Bound concurrent embedding requests (default 8; override via EMBED_MAX_CONCURRENCY).

        Per-instance → per-worker cap, so it scales naturally with uvicorn --workers.
        """
        s = getattr(self, "_embed_sema", None)
        if s is None:
            limit = getattr(self.settings, "EMBED_MAX_CONCURRENCY", None) or os.environ.get("EMBED_MAX_CONCURRENCY") or 8
            s = self._embed_sema = asyncio.Semaphore(int(limit))
        return s

    # ------------------------------------------------------------------
    # Embedding / rerank result cache (mcp_config.mcp_cache)
    # ------------------------------------------------------------------
    # Query embeddings are deterministic — the same query text always yields the same
    # vector — so we cache them keyed by sha256(text) as the document _id (auto-indexed,
    # no extra index needed). Vectors are stored as BSON binData float32 (subtype 9) to
    # keep docs compact and avoid float-array bloat. A TTL index on expires_at lets Mongo
    # auto-purge stale entries. Reranks are cached by (model, query, candidate docs, top_k)
    # since a rerank score depends on all of them. Cache lives in the backend server's one
    # long-lived event loop, so the Motor client stays loop-stable (no per-turn reset).

    def _cache_enabled(self, env_name: str) -> bool:
        raw = getattr(self.settings, env_name, None)
        if raw is None:
            raw = os.environ.get(env_name, "1")
        return str(raw).strip().lower() not in ("0", "false", "no", "off")

    def _cache_ttl(self, env_name: str, default: int = 300) -> int:
        raw = getattr(self.settings, env_name, None) or os.environ.get(env_name) or default
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return default

    async def _get_cache_collection(self):
        """Lazy Motor handle to mcp_config.mcp_cache with a TTL index (created once)."""
        mc = getattr(self, "_cache_mongo", None)
        if mc is None:
            from .mongodb_client import MongoDBClient
            mc = self._cache_mongo = MongoDBClient(settings=self.settings)
        await mc.ensure_connection()
        col = mc.get_collection("mcp_cache")
        if not getattr(self, "_cache_ttl_index_ready", False):
            try:
                # A TTL index on expires_at may already exist under another name (e.g.
                # MongoSessionCache's mcp_cache_entry_ttl). Reuse it rather than creating a
                # duplicate, which Mongo rejects with IndexOptionsConflict.
                existing = await col.index_information()
                has_ttl = any(
                    spec.get("expireAfterSeconds") is not None
                    and any(field == "expires_at" for field, _dir in spec.get("key", []))
                    for spec in existing.values()
                )
                if not has_ttl:
                    await col.create_index("expires_at", expireAfterSeconds=0, name="cache_ttl")
            except Exception as e:
                logger.debug("mcp_cache TTL index ensure failed (non-fatal): %s", e)
            self._cache_ttl_index_ready = True
        return col

    @staticmethod
    def _embed_cache_id(model: str, text: str) -> str:
        # Key by (model, text): the same text embedded by different models yields different
        # vectors, so the model MUST be part of the identity or caches collide.
        h = hashlib.sha256()
        h.update((model or "").encode("utf-8")); h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return "emb:" + h.hexdigest()

    async def _embed_cache_get(self, model: str, text: str) -> Optional[dict]:
        try:
            col = await self._get_cache_collection()
            doc = await col.find_one({"_id": self._embed_cache_id(model, text)})
        except Exception as e:
            logger.debug("embed cache get failed (non-fatal): %s", e)
            return None
        if not doc or doc.get("vector") is None:
            return None
        vec = doc["vector"]
        try:
            vector = list(vec.as_vector().data)   # binData float32 -> list[float]
        except AttributeError:
            vector = list(vec)                    # legacy: stored as a plain array
        return {"embedding_model": doc.get("embedding_model"), "vector": vector}

    async def _embed_cache_put(self, model: str, text: str, result: dict, ttl: int) -> None:
        try:
            vector = result.get("vector")
            if not vector:
                return
            col = await self._get_cache_collection()
            now = datetime.datetime.now(datetime.timezone.utc)
            _id = self._embed_cache_id(model, text)
            await col.update_one(
                {"_id": _id},
                {"$set": {
                    "type": "embedding",
                    "embedding_model": result.get("embedding_model"),
                    "vector": Binary.from_vector([float(x) for x in vector], BinaryVectorDtype.FLOAT32),
                    "expires_at": now + datetime.timedelta(seconds=ttl),
                    "updated_at": now,
                }},
                upsert=True,
            )
        except Exception as e:
            logger.warning("embed cache put FAILED: _id=%s err=%r", self._embed_cache_id(model, text), e)

    @staticmethod
    def _rerank_cache_id(model: str, query: str, documents: List[str], top_k: Optional[int]) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8")); h.update(b"\x00")
        h.update(query.encode("utf-8")); h.update(b"\x00")
        for d in documents:
            h.update(str(d).encode("utf-8")); h.update(b"\x00")
        h.update(str(top_k).encode("utf-8"))
        return "rrk:" + h.hexdigest()

    async def _rerank_cache_get(self, cache_id: str, documents: List[str]) -> Optional[List[dict]]:
        try:
            col = await self._get_cache_collection()
            doc = await col.find_one({"_id": cache_id})
        except Exception as e:
            logger.debug("rerank cache get failed (non-fatal): %s", e)
            return None
        if not doc or doc.get("results") is None:
            return None
        # Rehydrate the document text from the caller's list (we only cache index+score).
        out = []
        for r in doc["results"]:
            idx = r.get("index")
            entry = {"index": idx, "relevance_score": r.get("relevance_score")}
            if isinstance(idx, int) and 0 <= idx < len(documents):
                entry["document"] = documents[idx]
            out.append(entry)
        return out

    async def _rerank_cache_put(self, cache_id: str, results: List[dict], ttl: int) -> None:
        try:
            col = await self._get_cache_collection()
            now = datetime.datetime.now(datetime.timezone.utc)
            slim = [{"index": r.get("index"), "relevance_score": r.get("relevance_score")} for r in results]
            await col.update_one(
                {"_id": cache_id},
                {"$set": {
                    "type": "rerank",
                    "results": slim,
                    "expires_at": now + datetime.timedelta(seconds=ttl),
                    "updated_at": now,
                }},
                upsert=True,
            )
            logger.debug("rerank cache WRITE ok: _id=%s entries=%d ttl=%d", cache_id, len(slim), ttl)
        except Exception as e:
            logger.warning("rerank cache put FAILED: _id=%s err=%r", cache_id, e)

    # ------------------------------------------------------------------
    # Embedding request-coalescing (micro-batching)
    # ------------------------------------------------------------------
    # Many independent concurrent generate_embedding() callers (e.g. fan-out sub-agent
    # intakes) are collected into ONE batched Voyage call. Each caller awaits a future;
    # a short linger window (or hitting max-batch) triggers a flush that fires the batch
    # and resolves every future. All on the single event loop — no locks needed (buffer
    # mutations happen between awaits). Grouped by (resolved_model_id, input_type) since a
    # Voyage batch is one model + one input_type.

    def _embed_cfg(self):
        """(enabled, linger_seconds, max_batch) — from settings/env, cached. Voyage caps batch at 1000."""
        cfg = getattr(self, "_embed_cfg_cache", None)
        if cfg is None:
            raw_enabled = getattr(self.settings, "EMBED_BATCH", None)
            if raw_enabled is None:
                raw_enabled = os.environ.get("EMBED_BATCH", "0")
            enabled = str(raw_enabled).strip().lower() not in ("0", "false", "no", "off")
            linger_ms = int(getattr(self.settings, "EMBED_BATCH_LINGER_MS", None) or os.environ.get("EMBED_BATCH_LINGER_MS") or 20)
            max_batch = int(getattr(self.settings, "EMBED_MAX_BATCH", None) or os.environ.get("EMBED_MAX_BATCH") or 128)
            max_batch = max(1, min(max_batch, 1000))  # Voyage API hard cap
            cfg = self._embed_cfg_cache = (enabled, max(0.0, linger_ms / 1000.0), max_batch)
        return cfg

    def _resolve_voyage_model(self, model_id: Optional[str], is_query: bool) -> str:
        """Mirror generate_voyage_embeddings model resolution so batched + single paths agree."""
        if is_query:
            model_id = self.settings.QUERY_EMBEDDING_MODEL_ID
        if model_id is None:
            model_id = self.settings.EMBEDDING_MODEL_ID
        if not str(model_id).startswith("voyage-"):
            model_id = "voyage-4"
        return model_id

    async def _coalesced_embed(self, text: str, model_id: Optional[str], is_query: bool) -> dict:
        """Buffer this embed and await its slot in the next batched Voyage call."""
        if not hasattr(self, "_embed_buffers"):
            self._embed_buffers: Dict[tuple, list] = {}
            self._embed_timers: Dict[tuple, Any] = {}
        resolved = self._resolve_voyage_model(model_id, is_query)
        input_type = "query" if is_query else "document"
        key = (resolved, input_type)
        _, linger_s, max_batch = self._embed_cfg()
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        buf = self._embed_buffers.setdefault(key, [])
        buf.append((text, fut))
        if len(buf) >= max_batch:
            self._cancel_embed_timer(key)
            loop.create_task(self._flush_embed(key))
        elif key not in self._embed_timers:
            self._embed_timers[key] = loop.call_later(linger_s, self._fire_embed_flush, key)
        return await fut

    def _fire_embed_flush(self, key):
        self._embed_timers.pop(key, None)
        asyncio.get_event_loop().create_task(self._flush_embed(key))

    def _cancel_embed_timer(self, key):
        t = self._embed_timers.pop(key, None)
        if t is not None:
            t.cancel()

    async def _flush_embed(self, key):
        """Drain up to max_batch pending embeds for `key`, fire one batch, resolve futures."""
        self._cancel_embed_timer(key)
        buf = self._embed_buffers.get(key)
        if not buf:
            return
        _, _, max_batch = self._embed_cfg()
        # Drain atomically (no await between slice + delete on the single loop thread).
        items = buf[:max_batch]
        del buf[:len(items)]
        if buf:  # leftover beyond max_batch — schedule another drain
            asyncio.get_event_loop().create_task(self._flush_embed(key))
        model_id, input_type = key
        texts = [t for t, _ in items]
        try:
            results = await self.generate_voyage_embeddings_batch(
                texts, model_id=model_id, is_query=(input_type == "query"),
            )
        except Exception as e:
            for _, fut in items:
                if not fut.done():
                    fut.set_exception(e)
            return
        n = len(results)
        for i, (_, fut) in enumerate(items):
            if fut.done():
                continue
            if i < n:
                fut.set_result(results[i])
            else:
                fut.set_exception(RuntimeError("voyage batch returned fewer results than inputs"))

    async def generate_voyage_embeddings(self, text: str, model_id: Optional[str] = None, is_query: bool = True) -> list:
        """Generates an embedding for the input text using the Voyage AI API.
        https://www.mongodb.com/docs/api/doc/atlas-embedding-and-reranking-api/operation/operation-createembedding
        Args:
            text: Input text to embed.
            model_id: Voyage model to use. Defaults to self.settings.EMBEDDING_MODEL_ID.
            is_query: Whether the embedding is for a query. Defaults to True.
        
        Returns:
            list: Embedding vector (list of floats) produced by the model.
        """
        api_key = self.settings.mongo_voyage_apikey()
        if is_query:
            model_id = self.settings.QUERY_EMBEDDING_MODEL_ID   
        if model_id is None:                                     
            model_id = self.settings.EMBEDDING_MODEL_ID
        if not model_id.startswith("voyage-"):
            logger.debug(f"Model ID {model_id} for generate_voyage_embeddings is not a Voyage model. Defaulting to {model_id}.")
            model_id = "voyage-4"  # default to voyage-4 if not specified or incorrectly specified
            
        
        # voyage distinguishes between query and document embeddings for better performance
        input_type = "query" if is_query else "document"
        
        logger.debug(f"Using {input_type} embedding model: {model_id}")
        
        max_retries = 6
        base_delay = 2.0  # seconds
        client = self._get_voyage_client()
        sema = self._embed_semaphore()
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                # Cap concurrent in-flight embedding requests (fan-out storms caused
                # ConnectTimeout); hold the slot only for the POST, release during backoff.
                async with sema:
                    response = await client.post(
                        "https://ai.mongodb.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "input": text,
                            "model": model_id,
                            "input_type": input_type
                        },
                    )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    httpx.PoolTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError) as e:
                last_exc = e
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Voyage transport error {type(e).__name__} (attempt {attempt+1}/{max_retries}) — retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = float(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                logger.warning(
                    f"Voyage {response.status_code} — waiting {retry_after:.1f}s (attempt {attempt+1}/{max_retries})"
                )
                await asyncio.sleep(retry_after)
                continue
            response.raise_for_status()
            data = response.json()
            return {
                "embedding_model": model_id,
                "vector": data["data"][0]["embedding"]
            }
        raise RuntimeError(f"generate_voyage_embeddings: exceeded {max_retries} retries (last error: {last_exc})")

    async def generate_voyage_embeddings_batch(
        self,
        texts: List[str],
        model_id: Optional[str] = None,
        is_query: bool = False,
        batch_size: int = 1000,
    ) -> List[dict]:
        """Generate embeddings for a list of texts using the Voyage AI API.

        Splits *texts* into batches of up to *batch_size* (max 1000 per API limit),
        sends each batch in a single request, and returns results in input order.
        https://www.mongodb.com/docs/api/doc/atlas-embedding-and-reranking-api/operation/operation-createembedding
        Args:
            texts: List of strings to embed.
            model_id: Voyage model to use. Defaults to EMBEDDING_MODEL_ID (document)
                      or QUERY_EMBEDDING_MODEL_ID (query).
            is_query: When True, uses the query embedding model and input_type="query".
            batch_size: Items per API call. Capped at 1000 (API maximum).

        Returns:
            List of dicts, one per input text, each with keys:
                - "embedding_model": str
                - "vector": list[float]
        """
        api_key = self.settings.mongo_voyage_apikey()
        if is_query:
            model_id = model_id or self.settings.QUERY_EMBEDDING_MODEL_ID
        if model_id is None:
            model_id = self.settings.EMBEDDING_MODEL_ID
        if not model_id.startswith("voyage-"):
            model_id = "voyage-4"

        input_type = "query" if is_query else "document"
        batch_size = min(batch_size, 1000)

        results: List[dict] = [None] * len(texts)

        max_retries = 6
        base_delay = 2.0

        client = self._get_voyage_client()
        sema = self._embed_semaphore()
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start: batch_start + batch_size]
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    async with sema:
                        response = await client.post(
                            "https://ai.mongodb.com/v1/embeddings",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "input": batch,
                                "model": model_id,
                                "input_type": input_type,
                            },
                        )
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                        httpx.PoolTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError) as e:
                    last_exc = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Voyage batch transport error {type(e).__name__} "
                        f"(batch {batch_start}, attempt {attempt + 1}/{max_retries}) — retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = float(
                        response.headers.get("Retry-After", base_delay * (2 ** attempt))
                    )
                    logger.warning(
                        f"Voyage batch {response.status_code} — waiting {retry_after:.1f}s "
                        f"(batch {batch_start}, attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                data = response.json()
                # data["data"] is ordered by "index" within the batch
                for item in data["data"]:
                    global_idx = batch_start + item["index"]
                    results[global_idx] = {
                        "embedding_model": model_id,
                        "vector": item["embedding"],
                    }
                break  # success — move to next batch
            else:
                raise RuntimeError(
                    f"generate_voyage_embeddings_batch: exceeded {max_retries} retries "
                    f"on batch starting at index {batch_start} (last error: {last_exc})"
                )

        return results

    async def rerank(
        self,
        query: str,
        documents: List[str],        
        top_k: Optional[int] = 10,
        truncation: bool = True,
    ) -> List[dict]:
        """Rerank *documents* against *query* using the Voyage AI reranker API.

        Returns a list of dicts sorted by descending relevance_score:
            [{"index": <original_index>, "document": <str>, "relevance_score": <float>}, ...]

        Raises httpx.HTTPStatusError on API errors.
        """
        model = getattr(self.settings, "RERANK_MODEL_ID", "rerank-2.5")
        # Cache by (model, query, candidate docs, top_k) — a rerank score depends on all of
        # them. On a hit we rehydrate document text from the caller's list (only index+score
        # are stored) so the cached doc stays small.
        use_cache = self._cache_enabled("RERANK_CACHE") and bool(documents)
        cache_id = None
        if use_cache:
            cache_id = self._rerank_cache_id(model, query, documents, top_k)
            cached = await self._rerank_cache_get(cache_id, documents)
            if cached is not None:
                logger.debug("rerank cache HIT: _id=%s docs=%d", cache_id, len(documents))
                return cached
            logger.debug("rerank cache MISS: _id=%s docs=%d top_k=%s", cache_id, len(documents), top_k)
        else:
            logger.debug("rerank cache DISABLED: enabled=%s docs=%d", self._cache_enabled("RERANK_CACHE"), len(documents))
        api_key = self.settings.mongo_voyage_apikey()
        payload: Dict[str, Any] = {
            "query": query,
            "documents": documents,
            "model": model,
            "truncation": truncation,
            "top_k": top_k
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://ai.mongodb.com/v1/rerank",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        results = data.get("data", [])
        if use_cache and results:
            # 30-day default: rerank scores are deterministic for a fixed (model, query,
            # docs, top_k). Override via RERANK_CACHE_TTL (seconds).
            await self._rerank_cache_put(cache_id, results, self._cache_ttl("RERANK_CACHE_TTL", 2592000))
        elif use_cache:
            logger.warning("rerank cache NOT written: empty results (data keys=%s)", list(data.keys()))
        return results


class ServerBedrockClient(BedrockClient):
    """Server-side Bedrock client with prompt/context input formatting."""

    def __init__(self, settings):
        super().__init__(settings)
        instructions = getattr(settings, "agent_instructions", "")
        if instructions:
            self.system = [{"text": instructions}]
        else:
            logger.warning("ServerBedrockClient: agent_instructions EMPTY — system prompt NOT set")

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
