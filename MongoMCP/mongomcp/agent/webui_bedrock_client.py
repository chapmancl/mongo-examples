import json
from typing import Any, Dict, List, Optional
from ..bedrock_client import BedrockClient

JSON_DATA_START = '[JSON_DATA_START]'
JSON_DATA_END = '[JSON_DATA_END]'

def _extract_json_block(text: str):
    """Extract JSON from [JSON_DATA_START]...[JSON_DATA_END] tags.
    Returns (parsed_json, clean_text) or (None, original_text) if no tags found."""
    start_idx = text.find(JSON_DATA_START)
    end_idx = text.find(JSON_DATA_END)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None, text.strip()
    json_str = text[start_idx + len(JSON_DATA_START):end_idx].strip()
    clean = (text[:start_idx] + text[end_idx + len(JSON_DATA_END):]).strip()
    try:
        parsed = json.loads(json_str)
        return parsed, clean
    except (json.JSONDecodeError, TypeError):
        return None, text.strip()

class WebUiBedrockClient(BedrockClient):
    """Web UI Bedrock client with text-oriented response normalization helpers."""
    def _format_invoke_request(
        self,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        request_messages = messages if messages is not None else []

        if prompt or context:
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
        """WebUiBedrockClient override: formats the request then delegates to
        BedrockClient.invoke_bedrock_with_tools() (base class) via super().
        Returns the raw response dict — call invoke_bedrock_with_tools_text() instead
        if you want the normalized {response_text, jsondata, history} payload.
        """
        request = self._format_invoke_request(
            prompt=prompt,
            context=context,
            messages=messages,
        )
        return await super().invoke_bedrock_with_tools(  # BedrockClient (base class)
            request=request,
        )

    async def invoke_bedrock_with_tools_text(
        self,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Invoke Bedrock and return a normalized {response_text, jsondata, history} dict."""
        raw = await self.invoke_bedrock_with_tools(
            prompt=prompt,
            context=context,
            messages=messages,
        )
        return self.normalize_bedrock_response(raw, fallback_history=messages or [])

    async def _call_mcp_tool(
        self,
        toolname: str,
        tool_input: dict,
    ) -> str:
        """Web UI override point for MCP tool execution behavior."""
        return await super()._call_mcp_tool(toolname, tool_input)

    def normalize_bedrock_response(
        self,
        response_obj: Dict[str, Any],
        fallback_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        history = response_obj.get("history", fallback_history or [])

        if response_obj.get("error"):
            return {
                "response_text": f"Error: {response_obj['error']}",
                "jsondata": None,
                "history": history,
            }

        # Get the raw assistant text from history
        raw_text = None
        if history and history[-1].get("role") == "assistant":
            text_parts = [
                c["text"] for c in history[-1].get("content", [])
                if isinstance(c, dict) and "text" in c
            ]
            if text_parts:
                raw_text = " ".join(text_parts)

        # Use the response field as fallback source
        if not raw_text and "response" in response_obj:
            raw_text = str(response_obj["response"])

        if not raw_text:
            return {
                "response_text": "No response generated",
                "jsondata": None,
                "history": history,
            }

        # Only extract JSON via [JSON_DATA_START]...[JSON_DATA_END] tags.
        # All other JSON stays in the markdown as-is for display.
        tag_json, clean_text = _extract_json_block(raw_text)
        return {
            "response_text": clean_text,
            "jsondata": tag_json,  # None if no tags found
            "history": history,
        }
