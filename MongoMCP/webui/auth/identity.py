"""Provider-agnostic verified identity + auth error types.

Every AuthProvider (see providers/base.py) returns an Identity or raises
AuthError; nothing else in the app needs to know which provider produced it.
"""
from dataclasses import dataclass, field
from typing import List, Optional


class AuthError(Exception):
    """Raised by an AuthProvider when a request cannot be authenticated/authorized."""

    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


@dataclass
class Identity:
    """Verified identity extracted from a request by an AuthProvider."""

    sub: str
    email: Optional[str] = None
    groups: List[str] = field(default_factory=list)
    scp: List[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
