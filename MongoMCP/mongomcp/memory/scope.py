"""Scope constants and helpers for the memory subpackage.

Mirrors Go's memservice/scope.go.  Five integer scope levels with 10-step
gaps so future intermediate levels can be inserted without renumbering.
"""

# ---------------------------------------------------------------------------
# Scope constants
# ---------------------------------------------------------------------------
SCOPE_SHARED             = 0   # visible to everyone
SCOPE_AGENT              = 10  # visible to this agent only
SCOPE_USER               = 20  # visible to this user, any agent
SCOPE_USER_SESSION       = 30  # visible to this user in this session (default)
SCOPE_USER_SESSION_AGENT = 40  # visible to this user + session + agent_id

_NAME_MAP = {
    SCOPE_SHARED:             "shared",
    SCOPE_AGENT:              "agent",
    SCOPE_USER:               "user",
    SCOPE_USER_SESSION:       "user_session",
    SCOPE_USER_SESSION_AGENT: "user_session_agent",
}
_PARSE_MAP = {v: k for k, v in _NAME_MAP.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_scope(s: str) -> int:
    """Parse a scope name string to its integer constant.

    Returns -1 for unknown or empty input so callers can apply a default.
    """
    if not s:
        return -1
    return _PARSE_MAP.get(s.lower().strip(), -1)


def scope_name(scope: int) -> str:
    """Return the string name for a scope constant.

    Returns 'unknown' for unrecognised values.
    """
    return _NAME_MAP.get(scope, "unknown")


def collection_for_scope(scope: int) -> str:
    """Return the MongoDB collection name for a given scope int.

    scope >= SCOPE_USER_SESSION (30)  → memory_episodic  (session-bound)
    scope <  SCOPE_USER_SESSION       → memory_semantic   (long-term)

    Uses string literals to avoid circular imports.
    """
    return "memory_episodic" if scope >= SCOPE_USER_SESSION else "memory_semantic"


def shard_key_for_scope(
    memory_type: str,
    scope: int,
    agent_id: str = "",
    username: str = "",
    session_id: str = "",
) -> str:
    """Derive a TurboQuant shard key for a given scope.

    Mirrors Go's ShardKeyForScope() in memservice/scope.go.

    Examples:
        SCOPE_SHARED              → "memory_type|shared"
        SCOPE_AGENT               → "memory_type|agent:agent_id"
        SCOPE_USER                → "memory_type|u:username"
        SCOPE_USER_SESSION        → "memory_type|u:username|s:session_id"
        SCOPE_USER_SESSION_AGENT  → "memory_type|u:username|s:session_id|a:agent_id"
    """
    if scope == SCOPE_SHARED:
        return f"{memory_type}|shared"
    if scope == SCOPE_AGENT:
        return f"{memory_type}|agent:{agent_id}"
    if scope == SCOPE_USER:
        return f"{memory_type}|u:{username}"
    if scope == SCOPE_USER_SESSION:
        return f"{memory_type}|u:{username}|s:{session_id}"
    if scope == SCOPE_USER_SESSION_AGENT:
        return f"{memory_type}|u:{username}|s:{session_id}|a:{agent_id}"
    # Fallback: treat as user_session
    return f"{memory_type}|u:{username}|s:{session_id}"


def build_scope_filter(
    agent_id: str = "",
    username: str = "",
    session_id: str = "",
) -> list:
    """Build MongoDB ``$or`` clauses covering all 5 scope layers plus legacy docs.

    Returns a list suitable for use as the value of ``"$or"`` in a MongoDB
    filter dict.  Mirrors Go's BuildScopeFilter() in memservice/helpers.go.

    A document is visible if ANY clause matches:
      - scope=SHARED (0):               always visible
      - scope=AGENT (10):               agent_id matches
      - scope=USER (20):                username matches
      - scope=USER_SESSION (30):        username matches (session_id is provenance)
      - scope=USER_SESSION_AGENT (40):  username + agent_id match
      - no scope field (legacy):        is_isolated absent or False
    """
    # Legacy docs: no scope field and not explicitly isolated.
    clauses: list = [
        {"scope": SCOPE_SHARED},
        {"scope": {"$exists": False}, "is_isolated": {"$ne": True}},
    ]
    if agent_id:
        clauses.append({"scope": SCOPE_AGENT, "agent_id": agent_id})
    if username:
        clauses.append({"scope": SCOPE_USER, "username": username})
        clauses.append({"scope": SCOPE_USER_SESSION, "username": username})
    if username and agent_id:
        clauses.append({
            "scope": SCOPE_USER_SESSION_AGENT,
            "username": username,
            "agent_id": agent_id,
        })
    return clauses
