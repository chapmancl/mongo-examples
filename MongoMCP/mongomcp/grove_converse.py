"""
Grove (Converse) transport — a drop-in replacement for the
boto3 ``bedrock-runtime`` client's ``converse`` method.

Calls the Grove/Anthropic gateway:

    POST {base_url}/anthropic/v1/messages
    Headers: api-key: <GROVE_API_KEY>
             anthropic-version: 2023-06-01

The request is translated from Bedrock Converse format to Anthropic Messages
API format. The response is translated back to Bedrock Converse format so that
``BedrockClient`` callers need no changes.

Bedrock → Anthropic request translation:
  modelId              → model
  inferenceConfig.maxTokens → max_tokens  (default 4096)
  messages[].content[].text → messages[].content[].text  (same)
  messages[].content[].toolUse → messages[].content[].tool_use  (id/input/name)
  messages[].content[].toolResult → messages[].content[].tool_result
  system[].text blocks → system string (joined)
  toolConfig.tools[].toolSpec → tools[].{name, description, input_schema}

Anthropic → Bedrock response translation:
  content[].text       → output.message.content[].text
  content[].tool_use   → output.message.content[].toolUse  (toolUseId/name/input)
  stop_reason          → stopReason  ("end_turn" → "end_turn", "tool_use" → "tool_use")
  usage.input_tokens   → usage.inputTokens
  usage.output_tokens  → usage.outputTokens
"""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Anthropic Messages API version required by the gateway
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Bedrock → Anthropic request helpers
# ---------------------------------------------------------------------------

def _strip_cache_points(content: Any) -> Any:
    """Remove Bedrock ``cachePoint`` blocks — not supported by Anthropic API."""
    if isinstance(content, list):
        return [
            _strip_cache_points(item)
            for item in content
            if not (isinstance(item, dict) and "cachePoint" in item)
        ]
    return content


def _bedrock_content_to_anthropic(content: Any) -> Any:
    """Convert a single Bedrock content block to Anthropic format."""
    if not isinstance(content, dict):
        return content
    # text block — identical in both APIs
    if "text" in content:
        return {"type": "text", "text": content["text"]}
    # toolUse → tool_use
    if "toolUse" in content:
        tu = content["toolUse"]
        return {
            "type": "tool_use",
            "id": tu.get("toolUseId", ""),
            "name": tu.get("name", ""),
            "input": tu.get("input", {}),
        }
    # toolResult → tool_result
    if "toolResult" in content:
        tr = content["toolResult"]
        inner = tr.get("content", [])
        # Flatten inner content to a single string for Anthropic
        if isinstance(inner, list):
            text_parts = [
                c["text"] for c in inner if isinstance(c, dict) and "text" in c
            ]
            result_content = " ".join(text_parts) if text_parts else str(inner)
        else:
            result_content = str(inner)
        return {
            "type": "tool_result",
            "tool_use_id": tr.get("toolUseId", ""),
            "content": result_content,
        }
    return content


def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate Bedrock messages list to Anthropic format."""
    converted = []
    for msg in messages:
        content = msg.get("content", [])
        content = _strip_cache_points(content)
        if isinstance(content, list):
            content = [_bedrock_content_to_anthropic(c) for c in content]
        elif isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({"role": msg["role"], "content": content})
    return converted


def _convert_tools(tool_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate Bedrock toolConfig.tools list to Anthropic tools format."""
    tools = []
    for entry in tool_config.get("tools", []):
        spec = entry.get("toolSpec", entry)
        tools.append({
            "name": spec.get("name", ""),
            "description": spec.get("description", ""),
            "input_schema": spec.get("inputSchema", {}).get("json", spec.get("inputSchema", {})),
        })
    return tools


def _extract_system(system: List[Dict[str, Any]]) -> str:
    """Join Bedrock system content text blocks into a single system string."""
    parts = []
    for block in _strip_cache_points(system):
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Anthropic → Bedrock response helpers
# ---------------------------------------------------------------------------

def _anthropic_content_to_bedrock(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single Anthropic response content block to Bedrock format.

    Returns ``None`` for block types that have no Bedrock equivalent (e.g.
    ``thinking`` / ``redacted_thinking``) so the caller can filter them out.
    """
    btype = block.get("type", "")
    if btype == "text":
        return {"text": block.get("text", "")}
    if btype == "tool_use":
        return {
            "toolUse": {
                "toolUseId": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            }
        }
    # thinking / redacted_thinking — Anthropic-only, no Bedrock equivalent.
    # Drop them so they never appear as raw dict strings in the chat output.
    if btype in ("thinking", "redacted_thinking"):
        return None
    # Unknown fallback — drop rather than corrupt assistant text with repr strings.
    logger.debug("Dropping unknown Anthropic content block type=%r", btype)
    return None


def _anthropic_stop_reason(stop_reason: Optional[str]) -> str:
    """Map Anthropic stop_reason values to Bedrock stopReason values."""
    mapping = {
        "end_turn": "end_turn",
        "tool_use": "tool_use",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
    }
    return mapping.get(stop_reason or "", stop_reason or "end_turn")


def _anthropic_to_bedrock_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a full Anthropic Messages API response to Bedrock Converse shape."""
    anthropic_content = data.get("content", [])
    # Filter out None — _anthropic_content_to_bedrock returns None for block
    # types that have no Bedrock equivalent (thinking, redacted_thinking, etc.)
    bedrock_content = [
        b for b in (_anthropic_content_to_bedrock(block) for block in anthropic_content)
        if b is not None
    ]

    anthropic_usage = data.get("usage", {})
    input_tokens = anthropic_usage.get("input_tokens", 0)
    output_tokens = anthropic_usage.get("output_tokens", 0)

    return {
        "output": {
            "message": {
                "role": data.get("role", "assistant"),
                "content": bedrock_content,
            }
        },
        "stopReason": _anthropic_stop_reason(data.get("stop_reason")),
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GroveConverseClient:
    """Mimics ``boto3.client('bedrock-runtime')`` for the calls this repo uses.

    Translates Bedrock Converse API calls to the Grove/Anthropic gateway at:

        POST {base_url}/anthropic/v1/messages

    Authentication uses an ``api-key`` header (not Bearer).
    Only ``converse`` is implemented — the single method the LLM loop depends
    on. Calls are synchronous (matching boto3) so existing
    ``asyncio.to_thread`` / direct-call sites work unchanged.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        verify: Optional[str] = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.verify = verify if verify else True
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
            "api-key": self.api_key,
        }

    def converse(
        self,
        modelId: str,  # noqa: N803 - matches boto3's parameter name for drop-in compatibility
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, Any]]] = None,
        toolConfig: Optional[Dict[str, Any]] = None,  # noqa: N803
        inferenceConfig: Optional[Dict[str, Any]] = None,  # noqa: N803
        **_ignored: Any,
    ) -> Dict[str, Any]:
        """Send a request to the Grove/Anthropic gateway.

        Accepts the same arguments as ``boto3 bedrock-runtime.converse()`` and
        returns the same Bedrock-shaped response dict so all existing callers
        work without modification.
        """
        inference = inferenceConfig or {}
        max_tokens: int = inference.get("maxTokens", _DEFAULT_MAX_TOKENS)

        body: Dict[str, Any] = {
            "model": modelId,
            "max_tokens": max_tokens,
            "messages": _convert_messages(messages),
        }

        if system:
            body["system"] = _extract_system(system)

        if toolConfig:
            converted_tools = _convert_tools(toolConfig)
            if converted_tools:
                body["tools"] = converted_tools

        # Optional inference params
        if "temperature" in inference:
            body["temperature"] = inference["temperature"]
        if "topP" in inference:
            body["top_p"] = inference["topP"]
        if "stopSequences" in inference:
            body["stop_sequences"] = inference["stopSequences"]

        url = f"{self.base_url}/anthropic/v1/messages"
        logger.debug("Grove POST %s model=%s", url, modelId)

        response = self._session.post(
            url,
            json=body,
            headers=self._headers(),
            verify=self.verify,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"Grove call failed [{response.status_code}] "
                f"model={modelId}: {response.text[:1000]}"
            )

        return _anthropic_to_bedrock_response(response.json())

    @classmethod
    def from_settings(cls, settings) -> "GroveConverseClient":
        """Build a client from a Settings-like object."""
        return cls(
            base_url=getattr(settings, "LLM_ENDPOINT", ""),
            api_key=getattr(settings, "LLM_PROVIDER_API_KEY", ""),
            verify=getattr(settings, "CERTIFICATE_PATH", None),
        )
    