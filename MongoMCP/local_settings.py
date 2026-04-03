from typing import Dict, Optional
import os
import json


class LocalSettings:
    def __init__(self):
        # Keep these defaults aligned with AWS_settings.py so this file is a drop-in local replacement.
        self.aws_region = os.getenv('AWS_REGION', 'us-east-2')
        self.TOOL_NAME = os.getenv('MCP_TOOL_NAME', 'AirbnbSearch')
        self.IS_LOCAL = json.loads(os.getenv('IS_LOCAL', 'true').lower())     
        self.EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
        self.mcp_config_db = os.getenv('MCP_CONFIG_DB', 'mcp_config')
        self.mcp_config_col = os.getenv('MCP_CONFIG_COL', 'mcp_tools')
        self.LLM_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        self.LLM_MAX_ITERATIONS = int(os.getenv('LLM_MAX_ITERATIONS', '15'))
        self.ENABLE_CACHE_POINTS = os.getenv('ENABLE_CACHE_POINTS', 'true').lower() in ['1', 'true', 'yes', 'on']
        self.ENABLE_BEDROCK_CACHING = True       
        
        # Hardcoded credentials for local development only.
        self._credentials: Dict[str, str] = {
            'username': os.getenv('MONGO_USERNAME', 'bbmcp_user'),
            'password': os.getenv('MONGO_PASSWORD', '<secret>'),
            'mongoUri': os.getenv('MONGO_URI', 'mongodb+srv://demo1.sf56l.mongodb.net')
        }

    def get_mongo_credentials(self) -> Dict[str, str]:
        """
        Return MongoDB credentials from local hardcoded values.

        Returns:
            Dict containing username, password, and mongoUrl
        """
        return self._credentials

    def mongo_url(self) -> str:
        """Get MongoDB connection URL."""
        return self._credentials["mongoUri"]

    def mongo_timeout(self) -> int:
        """Get MongoDB timeout in milliseconds."""
        return 5000


# Create a singleton instance
settings = LocalSettings()
