# OpenAi Tester
import os
import sys
import requests
import logging
import datetime
import json
from urllib.parse import quote
from typing import Any, Dict, List, Optional

try:
    from .libertyair_token import LibertyAIRTokenProvider
except ImportError:  # allow standalone import (e.g. smoke test) without the package
    from libertyair_token import LibertyAIRTokenProvider

logger = logging.getLogger(__name__)

'''
Handles getting embeddings through Liberty AIR Azure OpenAi
'''
class LibertyAIREmbeddingClient:
    """Mimics ``boto3.client('bedrock-runtime')`` for the calls this repo uses.

    Only ``embedding`` is implemented — that is the single method the LLM loop
    depends on. Calls are synchronous (matching boto3) so existing
    ``asyncio.to_thread`` / direct-call sites work unchanged.
    """

    def __init__(
        self,
        base_url: str,
        embedding_model_id: str,
        token_provider: LibertyAIRTokenProvider,
        troux_id: str,
        verify: Optional[str] = None,
        timeout: Optional[int] = 120
        
    ):
        self.base_url = base_url.rstrip("/")
        self.embedding_model_id = embedding_model_id
        self.token_provider = token_provider
        self.troux_id = troux_id
        self.verify = verify if verify else True
        self.api_version = "2024-10-21"
        self.timeout = timeout
        self.policy_id = "LibertyAirMedium"
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token_provider.get_token()}",
            "Content-Type": "application/json",
            "user": self.troux_id,
            # Optional per gem-ai reference; documented examples require only `user`.
            "use-case": self.troux_id,
            #"x-policy-id": self.policy_id,
            "x-request-timeout": str(self.timeout),
        }

    def get_embedding(
        self,
        texts: str,
        model_id: str,
        verbose: Optional[str] = False
    ) -> list:
        """Send a text to create embeddings through LibertyAIR.
        Returns vector array.
        """
        model_id = model_id.replace("openai-","")
        url = f"{self.base_url}/use-cases/openai/deployments/{quote(model_id, safe='')}/embeddings"
        if isinstance(texts, str):
            texts = [texts]
        
        response = self._session.post(
            url, 
            params={"api-version": self.api_version}, 
            headers=self._headers(), 
            json={"input": texts}, 
            verify=self.verify,
            timeout=self.timeout)

        if not response.ok:
            raise RuntimeError(
                f"LibertyAIR OpenAI call failed [{response.status_code}] "
                f"model={model_id}: {response.text[:1000]}"
            )
        if verbose:
            print(response.text)
        data = response.json()
        result = []
        for item in data['data']:
            #logger.info(f'openai - index: {item["index"]},embed: {item["embedding"][:5]}')
            result.append(item["embedding"])
        if verbose:
            logger.info(f'Model: {data["model"]}, Usage: prompt: {data["usage"]["prompt_tokens"]}, tot: {data["usage"]["total_tokens"]}')
        return result

    @classmethod
    def from_settings(cls, settings, token_provider: Optional[LibertyAIRTokenProvider] = None) -> "LibertyAIREmbeddingClient":
        """Build a client from an LMIGSettings-like object."""
        provider = token_provider or LibertyAIRTokenProvider.from_settings(settings)
        return cls(
            base_url=getattr(settings, "LIBERTY_AIR_URL", ""),
            embedding_model_id=getattr(settings, "EMBEDDING_MODEL_ID", ""),
            token_provider=provider,
            troux_id=getattr(settings, "LIBERTY_TROUX_ID", ""),
            verify=getattr(settings, "CERTIFICATE_PATH", None)
        )

if __name__ == "__main__":
    blah = "blah"
