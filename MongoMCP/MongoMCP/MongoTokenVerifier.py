from fastmcp.server.auth import TokenVerifier
#from mcp.server.auth.provider import AccessToken
from fastmcp.server.dependencies import AccessToken
from .MongoMCPMiddleware import MongoMCPMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoTokenVerifier(TokenVerifier):
    """
    Simple class to manually verify Bearer tokens (jwt) against a mongodb collection
    piggybacks off the MongoMCPMiddleware class to do the actual mongo calls. FastAPI needs a TokenVerifier class to handle token verification.
    this is a demonstration implementation; in production you would want to use an actual JWT verification source.
    """
    def __init__(self, mongo_middleware: MongoMCPMiddleware ):
        self.resource_server_url="https://www.mongodb.com/" # we need the value for the base class, but don't use it
        super().__init__(required_scopes=["read"] )
        self.mongo_middleware = mongo_middleware
    
    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            (allowed, agent_rec) = self.mongo_middleware.check_authorization(token)            
            if allowed:
                # get permission scope from mongo
                scope = agent_rec.get("scope", ["read"])
                atoken = AccessToken(
                    token=token,
                    client_id= agent_rec["agent_key"],
                    scopes= scope                 
                )
                return atoken
        except Exception as e:
            logger.error(f"Token validation error: {e}")
        return None
