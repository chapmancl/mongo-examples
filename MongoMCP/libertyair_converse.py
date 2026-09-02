"""
LibertyAIR Bedrock (Converse) transport — a drop-in replacement for the
boto3 ``bedrock-runtime`` client's ``converse`` method.

Per LibertyAIR docs (https://libertyair.lmig.com/docs/routes/amazon-bedrock/),
the Bedrock route is a native Converse passthrough:

    POST {base}/use-cases/bedrock/deployments/{deployment_name}

It accepts the same request body the existing code already builds
(``messages`` / ``system`` / ``inferenceConfig`` / ``toolConfig``) and returns
the same response shape (``output.message`` / ``usage`` / ``stopReason``), so
``BedrockClient`` only needs to swap which object ``self.bedrock_client`` is.

EA direction (Frameworks & SDKs Decision Record) retires direct Bedrock in
favour of routing all inference through LibertyAIR, which also provides cost
tracking and automatic retry — so cache points are intentionally stripped here
(``cachePoint`` is undocumented on this route and unused by the gem-ai
reference implementation).
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

try:
    from .libertyair_token import LibertyAIRTokenProvider
except ImportError:  # allow standalone import (e.g. smoke test) without the package
    from libertyair_token import LibertyAIRTokenProvider

logger = logging.getLogger(__name__)


def _strip_cache_points(content: Any) -> Any:
    """Return content with any Bedrock ``cachePoint`` blocks removed.

    LibertyAIR's Bedrock route does not document cache-point support, so we
    drop them defensively rather than risk a ValidationException.
    """
    if isinstance(content, list):
        return [
            _strip_cache_points(item)
            for item in content
            if not (isinstance(item, dict) and "cachePoint" in item)
        ]
    return content


def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized = []
    for msg in messages:
        new_msg = dict(msg)
        if "content" in new_msg:
            new_msg["content"] = _strip_cache_points(new_msg["content"])
        sanitized.append(new_msg)
    return sanitized


class LibertyAIRConverseClient:
    """Mimics ``boto3.client('bedrock-runtime')`` for the calls this repo uses.

    Only ``converse`` is implemented — that is the single method the LLM loop
    depends on. Calls are synchronous (matching boto3) so existing
    ``asyncio.to_thread`` / direct-call sites work unchanged.
    """

    def __init__(
        self,
        base_url: str,
        token_provider: LibertyAIRTokenProvider,
        troux_id: str,
        verify: Optional[str] = None,
        timeout: int = 120,
        policy_id: str = "LibertyAirMedium",
    ):
        self.base_url = base_url.rstrip("/")
        self.token_provider = token_provider
        self.troux_id = troux_id
        self.verify = verify if verify else True
        self.timeout = timeout
        self.policy_id = policy_id
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token_provider.get_token()}",
            "Content-Type": "application/json",
            "user": self.troux_id,
            # Optional per gem-ai reference; documented examples require only `user`.
            "use-case": self.troux_id,
            "x-policy-id": self.policy_id,
            "x-request-timeout": str(self.timeout),
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
        """Send a Bedrock Converse request through LibertyAIR.

        Returns the parsed JSON response, matching the boto3 Converse response
        shape: ``{"output": {...}, "usage": {...}, "stopReason": "..."}``.
        """
        body: Dict[str, Any] = {"messages": _sanitize_messages(messages)}
        if system is not None:
            body["system"] = _strip_cache_points(system)
        if toolConfig is not None:
            body["toolConfig"] = toolConfig
        if inferenceConfig is not None:
            body["inferenceConfig"] = inferenceConfig

        url = f"{self.base_url}/use-cases/bedrock/deployments/{quote(modelId, safe='')}"
        response = self._session.post(
            url,
            json=body,
            headers=self._headers(),
            verify=self.verify,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"LibertyAIR Bedrock call failed [{response.status_code}] "
                f"model={modelId}: {response.text[:1000]}"
            )
        return response.json()

    @classmethod
    def from_settings(cls, settings, token_provider: Optional[LibertyAIRTokenProvider] = None) -> "LibertyAIRConverseClient":
        """Build a client from an LMIGSettings-like object."""
        provider = token_provider or LibertyAIRTokenProvider.from_settings(settings)
        return cls(
            base_url=getattr(settings, "LIBERTY_AIR_URL", ""),
            token_provider=provider,
            troux_id=getattr(settings, "LIBERTY_TROUX_ID", ""),
            verify=getattr(settings, "CERTIFICATE_PATH", None),
        )
