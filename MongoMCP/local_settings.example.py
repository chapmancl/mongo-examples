from typing import Dict, Optional
import os
import json


def _load_instructions() -> str:
    """Load agent instructions from agent_instructions.md, searching this dir then parent."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    for directory in [this_dir, os.path.dirname(this_dir)]:
        path = os.path.join(directory, "agent_instructions.md")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as infile:
                    return infile.read().strip()
            except Exception:
                pass
    return ""


_MEMORY_AGENT_INSTRUCTIONS = _load_instructions()


class LocalSettings:
    def __init__(self):
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.TOOL_NAME = os.getenv("MCP_TOOL_NAME", "YourToolName")
        self.IS_LOCAL = json.loads(os.getenv("IS_LOCAL", "true").lower())

        self.EMBEDDING_MODEL_ID = "voyage-4"
        self.QUERY_EMBEDDING_MODEL_ID = os.getenv(
            "QUERY_EMBEDDING_MODEL_ID",
            "voyage-4-lite" if self.EMBEDDING_MODEL_ID.startswith("voyage-") else self.EMBEDDING_MODEL_ID,
        )
        self.VOYAGE_AI_KEY = os.getenv("VOYAGE_AI_KEY", "your-voyage-api-key-here")

        self.mcp_config_db = "mcp_config"
        self.mcp_config_col = "mcp_tools"
        self.memory_db = os.getenv("MEMORY_DB", "mcp_config")

        self.LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
        self.LLM_MAX_ITERATIONS = int(os.getenv("LLM_MAX_ITERATIONS", "15"))
        self.ENABLE_CACHE_POINTS = os.getenv("ENABLE_CACHE_POINTS", "true").lower() in ["1", "true", "yes", "on"]
        self.ENABLE_BEDROCK_CACHING = True
        self.agent_instructions = _MEMORY_AGENT_INSTRUCTIONS

        self._credentials: Dict[str, str] = {
            "username": "your-mongo-username",
            "password": "your-mongo-password",
            "mongoUrl": "your-cluster.mongodb.net",
        }

    def get_mongo_credentials(self) -> Dict[str, str]:
        return self._credentials

    def mongo_url(self) -> str:
        return self._credentials["mongoUrl"]

    def mongo_timeout(self) -> int:
        return 5000

    def mongo_voyage_apikey(self) -> Optional[str]:
        return self.VOYAGE_AI_KEY


settings = LocalSettings()
