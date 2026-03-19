import json
from typing import Any, Dict, List, Optional
from mongomcp.bedrock_client import BedrockClient

JSON_DATA_START = '[JSON_DATA_START]'
JSON_DATA_END = '[JSON_DATA_END]'

def _remove_json_block(text: str):
    """Strip [JSON_DATA_START]...[JSON_DATA_END] from text.
    Returns clean_text with the JSON block removed."""
    start_idx = text.find(JSON_DATA_START)
    end_idx = text.find(JSON_DATA_END)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        clean = (text[:start_idx] + text[end_idx + len(JSON_DATA_END):]).strip()
        return clean
    return text.strip()

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

        # Always prefer the raw assistant text from history — it preserves the
        # [JSON_DATA_START] tags that _try_parse_json may have consumed upstream.
        raw_text = None
        if history and history[-1].get("role") == "assistant":
            text_parts = [
                c["text"] for c in history[-1].get("content", [])
                if isinstance(c, dict) and "text" in c
            ]
            if text_parts:
                raw_text = " ".join(text_parts)

        if "response" in response_obj:
            response = response_obj["response"]
            if isinstance(response, (dict, list)):
                # _try_parse_json already parsed the JSON block — use it as jsondata
                # and strip the JSON block from the raw history text
                clean_text = _remove_json_block(raw_text) if raw_text else ""
                return {
                    "response_text": clean_text or raw_text or "",
                    "jsondata": response,
                    "history": history,
                }
            # Plain string response — extract JSON block if present
            clean_text = _remove_json_block(str(response))
            return {
                "response_text": clean_text,
                "jsondata": None,
                "history": history,
            }

        return {
            "response_text": "No response generated",
            "jsondata": None,
            "history": history,
        }

    async def invoke_bedrock_with_tools_text(
        self,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Preferred public entry point for callers (e.g. CachedQueryProcessor).
        Calls self.invoke_bedrock_with_tools() [WebUiBedrockClient override → BedrockClient base]
        then passes the raw result through normalize_bedrock_response() to extract
        cleaned text and any structured JSON data block.
        Returns: {response_text: str, jsondata: dict|None, history: list}
        """
        response_obj = await self.invoke_bedrock_with_tools(  # WebUiBedrockClient override
            prompt=prompt,
            context=context,
            messages=messages,
        )
        return self.normalize_bedrock_response(response_obj, fallback_history=messages)
