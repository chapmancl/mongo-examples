from fastmcp.server.auth import TokenVerifier
#from mcp.server.auth.provider import AccessToken
from fastmcp.server.dependencies import AccessToken
from .mongo_mcp_middleware import MongoMCPMiddleware
import logging
import jwt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoTokenVerifier(TokenVerifier):
    """
    Verifies Bearer tokens and extracts agent identity for logging.

    Two modes controlled by the `strict` parameter:
    - strict=True  (MCP_AUTH_ENABLED=true):  HS256 tokens validated against MongoDB
                   agent_identities; RS256 Cognito tokens trusted (API Gateway pre-validated).
    - strict=False (MCP_AUTH_ENABLED=false): any well-formed JWT is accepted — identity is
                   parsed from the token payload for logging but no signature/DB check is done.
                   Use when the container sits behind API Gateway which already validated the token.
    """
    def __init__(self, mongo_middleware: MongoMCPMiddleware, strict: bool = True):
        self.resource_server_url="https://www.mongodb.com/" # we need the value for the base class, but don't use it
        super().__init__(required_scopes=["read"])
        self.mongo_middleware = mongo_middleware
        self.strict = strict

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            header = jwt.get_unverified_header(token)
            alg = header.get("alg", "")

            if not self.strict or alg == "RS256":
                # Non-strict mode OR Cognito RS256 — trust the token, just extract identity.
                payload = jwt.decode(token, options={"verify_signature": False})
                # Prefer cognito:username for a human-readable identity; fall back to sub/client_id.
                agent_name = (
                    payload.get("cognito:username")
                    or payload.get("username")
                    or payload.get("client_id")
                    or payload.get("sub", "unknown")
                )
                raw_scope = payload.get("scope", "read llm:invoke")
                scopes = raw_scope.split() if isinstance(raw_scope, str) else list(raw_scope)
                if not self.strict:
                    logger.debug(f"Non-strict auth: identity resolved as '{agent_name}'")
                else:
                    logger.debug(f"Cognito RS256 token accepted for agent: {agent_name}")
                return AccessToken(
                    token=token,
                    client_id=agent_name,
                    scopes=scopes,
                    claims={"agent_name": agent_name},
                )

            # Strict HS256 path — verify against MongoDB agent_identities.
            (allowed, agent_rec) = self.mongo_middleware.check_authorization(token)
            if allowed:
                scope = agent_rec.get("scope", ["read"])
                return AccessToken(
                    token=token,
                    client_id=agent_rec["agent_key"],
                    scopes=scope,
                    claims={"agent_name": agent_rec["agent_name"]},
                )
        except Exception as e:
            logger.error(f"Token validation error: {e}")
        return None
