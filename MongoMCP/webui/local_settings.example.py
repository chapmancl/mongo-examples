import os
from typing import Dict


class LocalSettings:
    def __init__(self):
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.mcp_config_db = "mcp_config"
        self.mcp_config_col = "mcp_tools"
        self.LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
        self.EMBEDDING_MODEL_ID = "voyage-4"
        self.QUERY_EMBEDDING_MODEL_ID = os.getenv(
            "QUERY_EMBEDDING_MODEL_ID",
            "voyage-4-lite" if self.EMBEDDING_MODEL_ID.startswith("voyage-") else self.EMBEDDING_MODEL_ID,
        )
        self.LLM_MAX_ITERATIONS = int(os.getenv("LLM_MAX_ITERATIONS", "15"))
        self.LLM_MAX_HISTORY = int(os.getenv("LLM_MAX_HISTORY", "20"))
        self.ENABLE_CACHE_POINTS = os.getenv("ENABLE_CACHE_POINTS", "true").lower() in ["1", "true", "yes", "on"]
        self.ENABLE_BEDROCK_CACHING = True
        self.ENABLE_MCP_TOOL_CACHING = False
        self.ENABLE_RESPONSE_CACHING = False
        self.CACHE_TTL = 300
        self.CACHE_NAMESPACE = os.getenv("CACHE_NAMESPACE", "local")
        self.AI_TOOL_ROUTING = False
        self.TOOL_ROUTING = False
        self.mongo_mcp_root = os.getenv("MONGO_MCP_ROOT", "http://localhost:8000")
        self.BEDROCK_SYSTEM_PROMPT_TEXTS = []
        self.AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "your-static-jwt-token-here")
        self.VOYAGE_AI_KEY = os.getenv("VOYAGE_AI_KEY", "your-voyage-api-key-here")
        self._credentials: Dict[str, str] = {
            "username": "your-mongo-username",
            "password": "your-mongo-password",
            "mongoUrl": "your-cluster.mongodb.net",
        }

    def get_mongo_credentials(self) -> Dict[str, str]:
        return self._credentials

    def get_auth_token(self) -> str:
        return self.AUTH_TOKEN

    def mongo_url(self) -> str:
        return self._credentials["mongoUrl"]

    def mongo_timeout(self) -> int:
        return 5000

    def mongo_voyage_apikey(self) -> str:
        return self.VOYAGE_AI_KEY


settings = LocalSettings()


def __getattr__(name: str):
    if hasattr(settings, name):
        return getattr(settings, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
