import os
import json
from typing import Dict, Optional
import boto3
from botocore.exceptions import ClientError

# lmig_settings.py — MCP server (mongo_mcp.py) production settings using AWS Secrets Manager.
# Copy this to lmig_settings.py and configure the environment variables below.

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


class LMIGSettings:
    def __init__(self):
        # Keep these defaults aligned with lmig_settings.py so this file is a drop-in local replacement.
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')

        # AWS Secrets Manager secret name/ARN containing MongoDB credentials
        # Secret must contain: {"username": "...", "password": "...", "uri": "...", "voyageapikey": "..."}
        self.mongo_creds = os.getenv('MONGO_CREDS', 'your-secret-name/your-mongo-creds')

        # Name of the MCP tool group served by this instance (matches mcp_tools collection key)
        self.TOOL_NAME = os.getenv('MCP_TOOL_NAME', 'YourToolName')

        # Set to true when running locally (skips some AWS-specific paths)
        self.IS_LOCAL = json.loads(os.getenv('IS_LOCAL', 'false').lower())

        # Embedding model
        self.QUERY_EMBEDDING_MODEL_ID = "none"
        #self.EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'voyage-4')
        self.EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', "openai-text-embedding-3-large-v1")
        #self.EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v2:0')
        self.QUERY_EMBEDDING_MODEL_ID = os.getenv(
            'QUERY_EMBEDDING_MODEL_ID',
            'voyage-4-lite' if self.EMBEDDING_MODEL_ID.startswith('voyage-') else self.EMBEDDING_MODEL_ID
        )
        # MongoDB config collection location (stores MCP tool definitions)
        self.mcp_config_db = "ai_config"
        self.mcp_config_col = "mcp_tools"
        self.memory_db = os.getenv('MEMORY_DB', 'ai_config')

        # LLM provider: 'libertyair' routes inference through the LibertyAIR
        # gateway (EA-sanctioned path); 'bedrock' calls AWS Bedrock directly
        # (retired per EA Frameworks & SDKs Decision Record — dev cutover only).
        self.LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'libertyair')

        # LLM model — for LibertyAIR this is the Bedrock deployment name from
        # https://test-libertyair.lmig.com/docs/model-details (not a boto3 ARN).
        self.LLM_MODEL_ID = os.getenv('LLM_MODEL_ID', 'us.anthropic.claude-sonnet-4-6')
        self.LLM_MAX_ITERATIONS = int(os.getenv('LLM_MAX_ITERATIONS', '15'))
        self.ENABLE_CACHE_POINTS = os.getenv('ENABLE_CACHE_POINTS', 'true').lower() in ['1', 'true', 'yes', 'on']
        self.ENABLE_BEDROCK_CACHING = True
        self.BEDROCK_KEY = os.getenv('SECRET__BB_NEW_SECRET__AIENDPOINT', 'NO_KEY')
        # Liberty AIR
        # Azure AD tenant for the client-credentials token grant.
        self.AZURE_TENANT_ID = os.getenv('AZURE_TENANT_ID', '08a83339-90e7-49bf-9075-957ccd561bf1')
        # Liberty Internal Application ID
        self.LIBERTY_TROUX_ID = "B7135EEA-F02B-44BC-B19E-70D195A9C6E1"
        self.CERTIFICATE_PATH = "certs/lm-ca-bundle.crt" # avoid SSL verify err, local issuer certificate
        self.LIBERTY_AIR_URL = "https://test-libertyair.lmig.com"
        self.LIBERTY_AIR_CLIENT = os.getenv('SECRET__GEM_LIBERTYAIR_CLIENT__ID', 'air_client_id')
        self.LIBERTY_AIR_SECRET = os.getenv('SECRET__GEM_LIBERTYAIR_CLIENT__SECRET','air_secret')
        self.LIBERTY_CLIENT_SCOPE = "87d1c382-6128-4150-aacf-bb624a9f2748/.default"
        
        self.mongo_mcp_root = os.getenv('MONGO_MCP_ROOT', 'http://localhost:8000')
        self.agent_instructions = _MEMORY_AGENT_INSTRUCTIONS

        # Hardcoded credentials for local development only.
        
        self._credentials_cache: Dict[str, str] = {
            'username': os.getenv('SECRET__GEM_EXPOSURE_REPOSITORY_MONGO_DB_TEST__READ_WRITE_SHORT__USERNAME', 'bbmcp_user'),
            'password': os.getenv('SECRET__GEM_EXPOSURE_REPOSITORY_MONGO_DB_TEST__READ_WRITE_SHORT__PASSWORD', '<secret>'),
            'mongoUrl': os.getenv('MONGO_URI', 'mongodb+srv://demo1.sf56l.mongodb.net'),
            'voyageapikey': os.getenv('SECRET__VOYAGE__VOYAGEAPIKEY', '<secret>')
        }

        # Capture temporary tokens
        # BJB Use AIR - os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.BEDROCK_KEY
        os.environ["VOYAGE_API_KEY"] = self._credentials_cache["voyageapikey"]

    def get_mongo_credentials(self) -> Dict[str, str]:
        """
        Return MongoDB credentials from local hardcoded values.

        Returns:
            Dict containing username, password, and mongoUrl
        """
        return self._credentials_cache

    def mongo_url(self) -> str:
        """Get MongoDB connection URL."""
        return self._credentials_cache["mongoUrl"]

    def mongo_timeout(self) -> int:
        """Get MongoDB timeout in milliseconds."""
        return 5000
    
    def mongo_voyage_apikey(self) -> str:
        return self._credentials_cache.get('voyageapikey', None)


# Create a singleton instance
settings = LMIGSettings()
