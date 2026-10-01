"""Generic forward-auth JWT provider.

Verifies a JWT that a fronting reverse-proxy/IdP integration already
authenticated and forwarded on a header. This exact pattern is used by
CorpSecure, Okta (App Proxy / API Access Management), Auth0, Azure AD App
Proxy, Cloudflare Access, oauth2-proxy, and similar - so it is intentionally
free of any vendor-specific defaults; every detail is env-driven (see
webui/auth/config.py).
"""
import logging

import jwt
from jwt import PyJWKClient

from .. import config
from ..identity import AuthError, Identity
from .base import AuthProvider

log = logging.getLogger(__name__)


def _extract_bearer(header_value: str):
    """Return the raw token from a header that may or may not use a Bearer prefix."""
    if not header_value:
        return None
    parts = header_value.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    if len(parts) == 1:
        return parts[0]
    return None


def _get_claim(claims: dict, path: str):
    """Resolve a claim by exact key first, then as a dotted path (a.b.c)."""
    if path in claims:
        return claims[path]
    value = claims
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


class JWTForwardAuthProvider(AuthProvider):
    """Verifies a forwarded RS256 JWT against a JWKS endpoint."""

    def __init__(self):
        self._jwks_client = None

    def _jwks(self) -> PyJWKClient:
        if self._jwks_client is None:
            if not config.AUTH_JWKS_URL:
                raise RuntimeError("AUTH_JWKS_URL is not configured but AUTH_MODE=jwt")
            # PyJWKClient caches keys in-memory and refetches on an unknown kid,
            # which transparently handles IdP key rotation.
            self._jwks_client = PyJWKClient(config.AUTH_JWKS_URL, cache_keys=True)
        return self._jwks_client

    def verify_request(self, headers) -> Identity:
        token = _extract_bearer(headers.get(config.AUTH_HEADER, ""))
        if not token:
            raise AuthError("Missing auth token")

        try:
            signing_key = self._jwks().get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=config.AUTH_ISSUER or None,
                audience=config.AUTH_AUDIENCE or None,
                options={"verify_aud": bool(config.AUTH_AUDIENCE)},
            )
        except AuthError:
            raise
        except Exception as e:  # jwt.* errors, key fetch errors, etc.
            raise AuthError(f"Invalid auth token: {e}")

        identity = Identity(
            sub=claims.get(config.AUTH_CLAIM_SUB, "") or "",
            email=_get_claim(claims, config.AUTH_CLAIM_EMAIL),
            groups=_get_claim(claims, config.AUTH_CLAIM_GROUPS) or [],
            scp=claims.get("scp") or [],
            raw=claims,
        )

        if config.AUTH_REQUIRED_GROUPS:
            if not set(config.AUTH_REQUIRED_GROUPS) & set(identity.groups):
                raise AuthError("Insufficient group membership", status=403)

        return identity
