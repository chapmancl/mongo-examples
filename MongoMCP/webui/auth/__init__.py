"""Pluggable authentication layer for the webui.

Public API:
  * ``init_auth(app)``        - install the before_request guard (no-op when disabled)
  * ``current_identity()``    - the verified Identity for the current request, or None
  * ``apply_identity(payload)`` - override client-supplied user_id/username with the
                                  verified token identity (no-op when disabled)

When ``AUTH_MODE=disabled`` (the default, OSS build) nothing internal is imported
and every request passes through untouched, preserving the existing self-declared
identity behavior. Set ``AUTH_MODE=jwt`` for the built-in forward-auth JWT
provider, or ``AUTH_PROVIDER=module.path:ClassName`` for a fully custom one -
see auth/providers/base.py.
"""
import logging

from flask import g, jsonify, request

from . import config

log = logging.getLogger(__name__)


def current_identity():
    """Return the verified Identity for the current request, or None."""
    try:
        return getattr(g, "identity", None)
    except RuntimeError:
        # Outside of an application/request context (e.g. background thread).
        return None


def apply_identity(payload):
    """Overwrite client-supplied identity with the verified token identity.

    Tracking keeps the same fields (user_id, username); when a verified
    identity is present the values become authoritative and the client can no
    longer spoof them. No-op when auth is disabled or no identity is attached.
    """
    identity = current_identity()
    if identity is None or not isinstance(payload, dict):
        return payload
    payload["user_id"] = identity.sub
    payload["username"] = identity.email or identity.sub
    payload["identity_verified"] = True
    return payload


def init_auth(app):
    """Register the auth guard on the Flask app based on AUTH_MODE/AUTH_PROVIDER."""
    if not config.auth_enabled():
        app.logger.warning(
            "Auth disabled (AUTH_MODE=%s); requests use self-declared identity.",
            config.AUTH_MODE,
        )
        return

    # Imported lazily so the OSS build never pulls in JWT/JWKS machinery.
    from .identity import AuthError
    from .providers import resolve_provider

    provider = resolve_provider()

    @app.before_request
    def _authenticate():
        if (request.path or "") in config.AUTH_EXEMPT_PATHS:
            return None
        try:
            g.identity = provider.verify_request(request.headers)
        except AuthError as e:
            return jsonify({"error": str(e)}), e.status
        return None

    app.logger.info(
        "Auth enabled (mode=%s, provider=%s).",
        config.AUTH_MODE,
        type(provider).__name__,
    )
