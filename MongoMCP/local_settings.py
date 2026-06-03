from typing import Dict, Optional
import os
import json

# local_settings.py — MCP server (mongo_mcp.py) local development settings.
# Copy this to local_settings.py and fill in your values.
# This is a drop-in replacement for AWS_settings.py that uses hardcoded credentials
# instead of AWS Secrets Manager — for local runs only, never commit real credentials.

def _load_instructions() -> str:
    """Load agent instructions from agent_instructions.md, searching this dir then parent."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    for directory in [this_dir, os.path.dirname(this_dir)]:
        path = os.path.join(directory, 'agent_instructions.md')
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                pass
    return ""

_MEMORY_AGENT_INSTRUCTIONS = _load_instructions()


class LocalSettings:
    def __init__(self):
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')

        # Name of the MCP tool group served by this instance (matches mcp_tools collection key)
        self.TOOL_NAME = os.getenv('MCP_TOOL_NAME', 'YourToolName')

        self.IS_LOCAL = json.loads(os.getenv('IS_LOCAL', 'true').lower())

        # Embedding model
        self.EMBEDDING_MODEL_ID = "voyage-4"
        self.QUERY_EMBEDDING_MODEL_ID = os.getenv(
            'QUERY_EMBEDDING_MODEL_ID',
            'voyage-4-lite' if self.EMBEDDING_MODEL_ID.startswith('voyage-') else self.EMBEDDING_MODEL_ID
        )

        # Voyage AI API key (only needed if embedding model starts with "voyage-")
        self.VOYAGE_AI_KEY = os.getenv('VOYAGE_AI_KEY', 'your-voyage-api-key-here')

        # MongoDB config collection location
        self.mcp_config_db = "mcp_config"
        self.mcp_config_col = "mcp_tools"
        self.memory_db = os.getenv('MEMORY_DB', 'mcp_config')

        # LLM model — Bedrock cross-region inference profile ID
        self.LLM_MODEL_ID = os.getenv('LLM_MODEL_ID', 'global.anthropic.claude-sonnet-4-6')
        self.LLM_MAX_ITERATIONS = int(os.getenv('LLM_MAX_ITERATIONS', '15'))

        self.ENABLE_CACHE_POINTS = os.getenv('ENABLE_CACHE_POINTS', 'true').lower() in ['1', 'true', 'yes', 'on']
        self.ENABLE_BEDROCK_CACHING = True

        # Static auth token for the MCP server (generate via the MCP server's token endpoint)
        self.AUTH_TOKEN = os.getenv('MCP_AUTH_TOKEN', 'your-static-jwt-token-here')

        self.agent_instructions = _MEMORY_AGENT_INSTRUCTIONS

        # Hardcoded MongoDB credentials for local development — replace with your Atlas cluster details
        self._credentials: Dict[str, str] = {
            "username": "your-mongo-username",
            "password": "your-mongo-password",
            "mongoUrl": "your-cluster.mongodb.net"
        }

    def get_mongo_credentials(self) -> Dict[str, str]:
        return self._credentials

    def mongo_url(self) -> str:
        return self._credentials["mongoUrl"]

    def mongo_timeout(self) -> int:
        return 5000

    def mongo_voyage_apikey(self) -> str:
        return self.VOYAGE_AI_KEY


# Create a singleton instance
settings = LocalSettings()
