"""Auth configuration, driven entirely by environment variables.

The public/OSS build runs with AUTH_MODE=disabled (the default) and pulls in
no vendor-specific auth dependencies at runtime. Set AUTH_MODE=jwt to verify a
forwarded JWT - this works behind any IdP/proxy that forwards one (CorpSecure,
Okta, Auth0, Azure AD, oauth2-proxy, Cloudflare Access, ...) - or set
AUTH_PROVIDER to a dotted "module.path:ClassName" to plug in a fully custom
AuthProvider (see auth/providers/base.py).
"""
import os

# "disabled" (default) | "jwt" (built-in forward-auth JWT verifier) | anything,
# provided AUTH_PROVIDER names a custom AuthProvider class.
AUTH_MODE = os.environ.get("AUTH_MODE", "disabled").strip().lower()

# Dotted "module.path:ClassName" of a custom AuthProvider. Overrides AUTH_MODE's
# built-in provider lookup entirely when set.
AUTH_PROVIDER = os.environ.get("AUTH_PROVIDER", "").strip()

# Header the fronting proxy/IdP forwards the verified identity JWT on.
AUTH_HEADER = os.environ.get("AUTH_HEADER", "Authorization")

# JWKS endpoint that publishes the token signing keys.
AUTH_JWKS_URL = os.environ.get("AUTH_JWKS_URL", "").strip()

# Expected issuer ("iss" claim). Empty disables issuer verification.
AUTH_ISSUER = os.environ.get("AUTH_ISSUER", "").strip()

# Expected audience ("aud" claim). Empty disables audience verification (some
# forward-auth proxies, e.g. CorpSecure, omit aud entirely).
AUTH_AUDIENCE = os.environ.get("AUTH_AUDIENCE", "").strip()

# Claim names to read the identity out of the decoded token. Override for IdPs
# that nest claims under a custom namespace (e.g. Auth0 custom claims).
AUTH_CLAIM_SUB = os.environ.get("AUTH_CLAIM_SUB", "sub")
AUTH_CLAIM_EMAIL = os.environ.get("AUTH_CLAIM_EMAIL", "email")
AUTH_CLAIM_GROUPS = os.environ.get("AUTH_CLAIM_GROUPS", "groups")

# Optional comma-separated list of required group claims; empty = any authenticated user.
_required_groups = os.environ.get("AUTH_REQUIRED_GROUPS", "").strip()
AUTH_REQUIRED_GROUPS = [g.strip() for g in _required_groups.split(",") if g.strip()]

# Paths that must never require auth (k8s liveness/readiness probes reach the
# app directly, not through the auth proxy, so they carry no token).
AUTH_EXEMPT_PATHS = tuple(
    p.strip() for p in os.environ.get("AUTH_EXEMPT_PATHS", "/health").split(",") if p.strip()
)


def auth_enabled() -> bool:
    """True when request-level auth verification should be enforced."""
    return AUTH_MODE != "disabled"
