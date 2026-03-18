import json
from typing import Any, Dict, List, Optional
from mongomcp.bedrock_client import BedrockClient

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
        request = self._format_invoke_request(
            prompt=prompt,
            context=context,
            messages=messages,
        )
        return await super().invoke_bedrock_with_tools(
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
                "history": history,
            }

        if "response" in response_obj:
            response = response_obj["response"]
            if isinstance(response, (dict, list)):
                response = json.dumps(response)
            return {
                "response_text": str(response),
                "history": history,
            }

        if history and history[-1].get("role") == "assistant":
            text_content = [
                content["text"]
                for content in history[-1].get("content", [])
                if isinstance(content, dict) and "text" in content
            ]
            if text_content:
                return {
                    "response_text": " ".join(text_content),
                    "history": history,
                }

        return {
            "response_text": "No response generated",
            "history": history,
        }

    async def invoke_bedrock_with_tools_text(
        self,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Invoke Bedrock tools flow and return normalized text response payload."""
        response_obj = await self.invoke_bedrock_with_tools(
            prompt=prompt,
            context=context,
            messages=messages,
        )
        return self.normalize_bedrock_response(response_obj, fallback_history=messages)
