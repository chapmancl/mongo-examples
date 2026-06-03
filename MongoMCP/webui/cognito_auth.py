"""
Cognito JWT token provider for machine-to-machine auth against API Gateway.

Configure via environment variables:
    COGNITO_CLIENT_ID   — Cognito App Client ID (from Terraform output cognito_client_id)
    COGNITO_USERNAME    — Cognito user account username
    COGNITO_PASSWORD    — Cognito user account password

If these are not set, aws_settings.get_auth_token() falls back to AUTH_TOKEN.
"""
import time
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Refresh when less than this many seconds remain (5 minutes buffer)
_REFRESH_BUFFER_SECONDS = 300


class CognitoTokenProvider:
    """Fetches and caches Cognito access tokens, refreshing before expiry."""

    def __init__(self, region: str, client_id: str, username: str, password: str):
        self._region = region
        self._client_id = client_id
        self._username = username
        self._password = password
        self._token: Optional[str] = None
        self._expires_at: float = 0
        self._client = boto3.client("cognito-idp", region_name=region)

    def get_token(self) -> str:
        """Return a valid access token, re-fetching if within the refresh buffer."""
        if self._token and time.time() < self._expires_at - _REFRESH_BUFFER_SECONDS:
            return self._token
        return self._fetch_token()

    def _fetch_token(self) -> str:
        try:
            resp = self._client.initiate_auth(
                AuthFlow="USER_PASSWORD_AUTH",
                ClientId=self._client_id,
                AuthParameters={
                    "USERNAME": self._username,
                    "PASSWORD": self._password,
                },
            )
            result = resp["AuthenticationResult"]
            self._token = result["AccessToken"]
            expires_in = result.get("ExpiresIn", 3600)
            self._expires_at = time.time() + expires_in
            logger.info("Cognito token refreshed, expires in %ds", expires_in)
            return self._token
        except ClientError as exc:
            logger.error("Failed to fetch Cognito token: %s", exc)
            raise
