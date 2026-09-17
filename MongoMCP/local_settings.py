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
        self.transport = os.getenv('MCP_TRANSPORT', 'http')
        self.host = os.getenv('SERVER_HOST', '0.0.0.0')
        self.port = int(os.getenv('SERVER_PORT', '8000'))
        self.webui_port = int(os.getenv('WEBUI_PORT', '8001'))
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')

        # Name of the MCP tool group served by this instance (matches mcp_tools collection key)
        self.mcp_tool_name = os.getenv('MCP_TOOL_NAME', 'AirbnbSearch')
        self.TOOL_NAME = os.getenv('MCP_TOOL_NAME', 'YourToolName')

        self.IS_LOCAL = json.loads(os.getenv('IS_LOCAL', 'true').lower())

        # LLM model — Bedrock cross-region inference profile ID
        self.LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'bedrock').lower()
        self.LLM_ENDPOINT = os.getenv('LLM_ENDPOINT', 'https://bedrock-runtime.us-east-1.amazonaws.com')
        self.LLM_PROVIDER_API_KEY = os.getenv('LLM_PROVIDER_API_KEY', '')
        self.LLM_MODEL_ID = os.getenv('LLM_MODEL_ID', 'global.anthropic.claude-sonnet-4-6')
        self.LLM_MAX_ITERATIONS = int(os.getenv('LLM_MAX_ITERATIONS', '15'))
        self.ENABLE_CACHE_POINTS = os.getenv('ENABLE_CACHE_POINTS', 'true').lower() in ['1', 'true', 'yes', 'on']
        self.ENABLE_BEDROCK_CACHING = True
        # Embedding model
        self.EMBEDDING_MODEL_ID = "voyage-4"
        self.QUERY_EMBEDDING_MODEL_ID = os.getenv(
            'QUERY_EMBEDDING_MODEL_ID',
            'voyage-4-lite' if self.EMBEDDING_MODEL_ID.startswith('voyage-') else self.EMBEDDING_MODEL_ID
        )

        # Voyage AI API key (only needed if embedding model starts with "voyage-")
        self.VOYAGE_AI_KEY = os.getenv('VOYAGE_AI_KEY', 'your-voyage-api-key-here')

        # MongoDB config collection location
        self.mcp_config_db = "ai_config"
        self.mcp_config_col = "mcp_tools"
        self.memory_db = os.getenv('MEMORY_DB', 'mcp_config')


        # Static auth token for the MCP server (generate via the MCP server's token endpoint)
        self.AUTH_TOKEN = os.getenv('MCP_AUTH_TOKEN', 'your-static-jwt-token-here')

        self.agent_instructions = _MEMORY_AGENT_INSTRUCTIONS

        # -------------- For WEBUI ---------------------- #
        self._cognito = None
        self.ENABLE_BEDROCK_CACHING = True
        self.ENABLE_MCP_TOOL_CACHING = False
        self.ENABLE_RESPONSE_CACHING = False
        self.CACHE_TTL = 300
        self.CACHE_NAMESPACE = os.getenv('CACHE_NAMESPACE', 'local')  # Isolates cache from AWS builds
        self.AI_TOOL_ROUTING = False
        self.TOOL_ROUTING = False
        self.mongo_mcp_root = os.getenv('MONGO_MCP_ROOT', 'http://localhost:8000')

        # Save LLM conversation history to MongoDB llm_history collection
        self.SAVE_LLM_HISTORY = os.getenv('SAVE_LLM_HISTORY', 'true').lower() in ['1', 'true', 'yes', 'on']

        self.BEDROCK_SYSTEM_PROMPT_TEXTS = [
            "***IMPORTANT: DO NOT recall sessions by username until you have confirmed the username with the user. DO NOT ASSUME you know the Username. Default username is demo-user",
            "***IMPORTANT: STRATEGY FIRST: Before any tool call execute memory_strategy_recall to find applicable patterns THEN EXECUTE the found pattern. Validated and high scoring patterns CANNOT be ignored.***",
            "***IMPORTANT: All output should be Markdown formatted for display within a div in an existing webpage. Do not include html, head, or body tags. Only include the inner content. Always use Markdown formatting.",
        ]

        # Static auth token for the MCP server
        self.AUTH_TOKEN = os.getenv('MCP_AUTH_TOKEN', 'your-static-jwt-token-here')

        # ----------- END WEBUI Specific --------------- #

        # Hardcoded MongoDB credentials for local development — replace with your Atlas cluster details
        self._credentials: Dict[str, str] = {
            "username": os.getenv('MONGO_USERNAME', 'mongo_username'),
            "password": os.getenv('MONGO_PASSWORD', 'password'),
            "mongoUrl": os.getenv('MONGO_URI', 'uri')
        }
    
    def get_auth_token(self) -> str:
        """Return a Cognito JWT if configured, otherwise fall back to the static AUTH_TOKEN."""
        if self._cognito is not None:
            return self._cognito.get_token()
        return self.AUTH_TOKEN
    
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
