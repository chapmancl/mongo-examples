"""
Azure AD OAuth2 client-credentials token provider for LibertyAIR.

LibertyAIR is fronted by Azure AD. Callers obtain a bearer token via the
client-credentials grant, then send it as `Authorization: Bearer <token>` on
every model call. Tokens are cached per scope and refreshed before expiry.

Ported from the gem-ai reference (apps/agent-runtime/src/lib/token-cache.ts).
Kept dependency-light (requests + stdlib) so it can run in a standalone smoke
test without importing the full server stack.
"""

import logging
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Refresh this many seconds before the token actually expires, so an in-flight
# request never races the expiry boundary. Matches the gem-ai 5-minute buffer.
EXPIRY_BUFFER_SECONDS = 5 * 60


class LibertyAIRTokenProvider:
    """Thread-safe, expiry-aware Azure AD client-credentials token cache."""

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        scope: str,
        verify: Optional[str] = None,
        timeout: int = 30,
    ):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        # `verify` is the path to the internal CA bundle (or None to use defaults).
        self.verify = verify
        self.timeout = timeout
        self.token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0

    def get_token(self, force_refresh: bool = False) -> str:
        """Return a valid bearer token, fetching a fresh one if needed.

        A single lock serialises concurrent refreshes so only one network call
        is made when the cache is cold or stale.
        """
        now = time.monotonic()
        if not force_refresh and self._access_token and now < self._expires_at:
            return self._access_token

        with self._lock:
            # Re-check inside the lock: another thread may have refreshed while
            # we waited for it.
            now = time.monotonic()
            if not force_refresh and self._access_token and now < self._expires_at:
                return self._access_token
            self._access_token, self._expires_at = self._fetch_token()
            return self._access_token

    def _fetch_token(self) -> tuple[str, float]:
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scope,
        }
        response = requests.post(
            self.token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            verify=self.verify if self.verify else True,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"LibertyAIR token fetch failed [{response.status_code}]: {response.text[:500]}"
            )
        payload = response.json()
        access_token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 3600))
        expires_at = time.monotonic() + max(0, expires_in - EXPIRY_BUFFER_SECONDS)
        logger.info("Fetched LibertyAIR token (expires_in=%ss, scope=%s)", expires_in, self.scope)
        return access_token, expires_at

    @classmethod
    def from_settings(cls, settings) -> "LibertyAIRTokenProvider":
        """Build a provider from an LMIGSettings-like object."""
        return cls(
            tenant_id=getattr(settings, "AZURE_TENANT_ID", ""),
            client_id=getattr(settings, "LIBERTY_AIR_CLIENT", ""),
            client_secret=getattr(settings, "LIBERTY_AIR_SECRET", ""),
            scope=getattr(settings, "LIBERTY_CLIENT_SCOPE", ""),
            verify=getattr(settings, "CERTIFICATE_PATH", None),
        )
