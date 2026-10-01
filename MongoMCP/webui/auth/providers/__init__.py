"""Auth provider registry + resolution.

Built-in providers are registered by short AUTH_MODE name below. A fully
custom provider can be injected via AUTH_PROVIDER="module.path:ClassName"
without editing this file - that's the hook/injection point for anything that
doesn't fit the generic forward-auth-JWT shape (e.g. SAML, opaque session
tokens looked up against an IdP API, etc.).
"""
import importlib

from .. import config
from .jwt_forward_auth import JWTForwardAuthProvider

_BUILTIN = {
    "jwt": JWTForwardAuthProvider,
}


def resolve_provider():
    """Return an AuthProvider instance for the current config.

    AUTH_PROVIDER (a dotted "module.path:ClassName") always wins when set;
    otherwise AUTH_MODE is looked up in the built-in registry.
    """
    if config.AUTH_PROVIDER:
        module_path, _, attr = config.AUTH_PROVIDER.partition(":")
        if not attr:
            raise RuntimeError("AUTH_PROVIDER must be 'module.path:ClassName'")
        module = importlib.import_module(module_path)
        provider_cls = getattr(module, attr)
        return provider_cls()

    try:
        provider_cls = _BUILTIN[config.AUTH_MODE]
    except KeyError:
        raise RuntimeError(
            f"Unknown AUTH_MODE={config.AUTH_MODE!r}. Use one of {list(_BUILTIN)}, "
            "or set AUTH_PROVIDER=module.path:ClassName for a custom provider."
        )
    return provider_cls()
