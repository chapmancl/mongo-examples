"""Base interface every auth provider must implement."""
from abc import ABC, abstractmethod

from ..identity import Identity


class AuthProvider(ABC):
    """Verifies an inbound request and returns the caller's Identity.

    Implement this and point AUTH_PROVIDER at "module.path:ClassName" to plug
    in any IdP/forward-auth setup (Okta, Auth0, Azure AD, a custom proxy, SAML
    session lookup, ...) without touching webui/auth/__init__.py at all.
    """

    @abstractmethod
    def verify_request(self, headers) -> Identity:
        """Return the verified Identity, or raise AuthError(status=401/403)."""
        raise NotImplementedError
