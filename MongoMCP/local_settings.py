import os
import json
from typing import Dict, Optional


class LocalSettings:
    """
     Local settings for running MongoMCP standalone without AWS dependencies.
    Uses environment variables for MongoDB credentials instead of AWS Secrets Manager.
    """
    def __init__(self):
        self.aws_region = os.getenv('AWS_REGION', 'us-east-2')
        self.EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
        self.mcp_config_db = os.getenv('MCP_CONFIG_DB', 'mcp_config')
        self.mcp_config_col = os.getenv('MCP_CONFIG_COL', 'mcp_tools')
        self.mcp_tool_name = os.getenv('MCP_TOOL_NAME', 'AirbnbSearch')
        self.transport = os.getenv('MCP_TRANSPORT', 'http')
        self.host = os.getenv('SERVER_HOST', '0.0.0.0')
        self.port = int(os.getenv('SERVER_PORT', '8001'))
        
        # LLM Model selection - can be overridden via environment variable
        self.LLM_MODEL_ID = os.getenv(
            'LLM_MODEL_ID', 
            'global.anthropic.claude-haiku-4-5-20251001-v1:0'
        )
        
        # Cache for credentials to avoid repeated parsing
        self._credentials_cache: Optional[Dict[str, str]] = None
        self._credentials_cache = {
            'username': os.getenv('MONGO_USERNAME', 'bbmcp_user'),
            'password': os.getenv('MONGO_PASSWORD', '<secret>'),
            'mongoUri': os.getenv('MONGO_URI', 'mongodb+srv://demo1.sf56l.mongodb.net')
        }
    
    def get_mongo_credentials(self, creds: Dict[str, str] = None) -> Dict[str, str]:
        """
        Update Credentials Cache
        
        Returns:
            Dict containing username, password, and mongoUri
            
        Raises:
            ValueError: If required environment variables are not set
        """
        if creds:
            self._credentials_cache = creds
            return self._credentials_cache           
        else:
            return self._credentials_cache
      
    def mongo_url(self) -> str:
        """Get MongoDB connection URL."""
        return self._credentials_cache['mongoUri']

    def mongo_timeout(self) -> int:
        """Get MongoDB timeout in milliseconds."""
        return int(os.getenv('MONGO_TIMEOUT', '5000'))


# Create a singleton instance
local_settings = LocalSettings()

