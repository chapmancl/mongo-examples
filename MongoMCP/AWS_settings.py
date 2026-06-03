import os
import json
from typing import Dict, Optional
import boto3
from botocore.exceptions import ClientError

# AWS_settings.py — MCP server (mongo_mcp.py) production settings using AWS Secrets Manager.
# Copy this to AWS_settings.py and configure the environment variables below.

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


class AWSSettings:
    def __init__(self):
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')

        # AWS Secrets Manager secret name/ARN containing MongoDB credentials
        # Secret must contain: {"username": "...", "password": "...", "uri": "...", "voyageapikey": "..."}
        self.mongo_creds = os.getenv('MONGO_CREDS', 'your-secret-name/your-mongo-creds')

        # Name of the MCP tool group served by this instance (matches mcp_tools collection key)
        self.TOOL_NAME = os.getenv('MCP_TOOL_NAME', 'YourToolName')

        # Set to true when running locally (skips some AWS-specific paths)
        self.IS_LOCAL = json.loads(os.getenv('IS_LOCAL', 'false').lower())

        # Embedding model
        self.EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'voyage-4')
        self.QUERY_EMBEDDING_MODEL_ID = os.getenv(
            'QUERY_EMBEDDING_MODEL_ID',
            'voyage-4-lite' if self.EMBEDDING_MODEL_ID.startswith('voyage-') else self.EMBEDDING_MODEL_ID
        )

        # MongoDB config collection location (stores MCP tool definitions)
        self.mcp_config_db = "mcp_config"
        self.mcp_config_col = "mcp_tools"
        self.memory_db = os.getenv('MEMORY_DB', 'mcp_config')

        # LLM model — Bedrock cross-region inference profile ID
        self.LLM_MODEL_ID = os.getenv('LLM_MODEL_ID', 'global.anthropic.claude-sonnet-4-6')
        self.LLM_MAX_ITERATIONS = int(os.getenv('LLM_MAX_ITERATIONS', '15'))

        self.ENABLE_CACHE_POINTS = os.getenv('ENABLE_CACHE_POINTS', 'true').lower() in ['1', 'true', 'yes', 'on']
        self.ENABLE_BEDROCK_CACHING = True

        self.agent_instructions = _MEMORY_AGENT_INSTRUCTIONS

        self._secrets_client = boto3.client('secretsmanager', region_name=self.aws_region)
        self._credentials_cache: Optional[Dict[str, str]] = None
        self.get_mongo_credentials()

    def get_mongo_credentials(self) -> Dict[str, str]:
        """Fetch MongoDB credentials from AWS Secrets Manager."""
        if self._credentials_cache:
            return self._credentials_cache
        try:
            response = self._secrets_client.get_secret_value(SecretId=self.mongo_creds)
            secret = json.loads(response['SecretString'])
            self._credentials_cache = {
                'username': secret['username'],
                'password': secret['password'],
                'mongoUrl': secret['uri'],
                'voyageapikey': secret.get('voyageapikey', None)
            }
            return self._credentials_cache
        except ClientError as error:
            print(f'Failed to fetch MongoDB credentials: {error}')
            raise error
        except json.JSONDecodeError as error:
            print(f'Failed to parse secret JSON: {error}')
            raise error

    def mongo_url(self) -> str:
        return self._credentials_cache['mongoUrl']

    def mongo_timeout(self) -> int:
        return 5000

    def mongo_voyage_apikey(self) -> str:
        return self._credentials_cache.get('voyageapikey', None)


# Create a singleton instance
settings = AWSSettings()
