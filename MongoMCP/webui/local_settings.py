import os
from typing import Dict

class LocalSettings:
    def __init__(self):
        self.aws_region = os.getenv('AWS_REGION', 'us-east-2')        
        self.EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
        self.mcp_config_db = "mcp_config"
        self.mcp_config_col = "mcp_tools"
        self.LLM_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        self.LLM_MAX_ITERATIONS = int(os.getenv('LLM_MAX_ITERATIONS', '15'))
        self.ENABLE_CACHE_POINTS = os.getenv('ENABLE_CACHE_POINTS', 'true').lower() in ['1', 'true', 'yes', 'on']
        self.ENABLE_BEDROCK_CACHING = True
        self.ENABLE_MCP_TOOL_CACHING = True
        self.ENABLE_RESPONSE_CACHING = True
        self.CACHE_TTL = 300
        self.mongo_mcp_root = os.getenv('MONGO_MCP_ROOT', "https://mongomcproot.example.com")
        self.BEDROCK_SYSTEM_PROMPT_TEXTS = [
            "***IMPORTANT: if a vector_search tool is available, always use vector_search before aggregate_query",
            "***IMPORTANT: All output should be Markdown formatted for display within a div in an existing webpage. Do not include html, head, or body tags. Only include the inner content. Always use Markdown formatting.",
            "Only use vector_search with collections that have a search_indexes.type=vectorSearch.",
        ]
        self.AUTH_TOKEN = ""
        
        
        # Hardcoded credentials for local development only.
        self._credentials: Dict[str, str] = {
            "username": "mymongousername",
            "password": "mymongopassword",
            "mongoUrl": "mymongo.mongodb.net"
        }
    
    def get_mongo_credentials(self) -> Dict[str, str]:
        """
        Fetch MongoDB credentials from AWS Secrets Manager.
        
        Returns:
            Dict containing username, password, and mongoUrl
            
        Raises:
            Exception: If failed to fetch credentials
        """
        return self._credentials

      
    def mongo_url(self) -> str:
        """Get MongoDB connection URL."""
        return self._credentials['mongoUrl']

    def mongo_timeout(self) -> int:
        """Get MongoDB timeout in milliseconds."""
        return 5000


# Create a singleton instance
settings = LocalSettings()


def __getattr__(name: str):
    """Backward-compatible module attribute access via singleton settings."""
    if hasattr(settings, name):
        return getattr(settings, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

