import os
import json
from typing import Dict, Optional
import boto3
from botocore.exceptions import ClientError


class AWSSettings:
    def __init__(self):
        self.aws_region = os.getenv('AWS_REGION', 'us-east-2')
        self.mongo_creds = os.getenv('MONGO_CREDS')
        self.TOOL_NAME = os.getenv('MCP_TOOL_NAME', 'AirbnbSearch')
        self.IS_LOCAL = json.loads(os.getenv('IS_LOCAL', 'true').lower())
        self.EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
        self.mcp_config_db = "mcp_config"
        self.mcp_config_col = "mcp_tools"
        self.LLM_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        self.LLM_MAX_ITERATIONS = int(os.getenv('LLM_MAX_ITERATIONS', '15'))
        self.ENABLE_CACHE_POINTS = os.getenv('ENABLE_CACHE_POINTS', 'true').lower() in ['1', 'true', 'yes', 'on']
        self.ENABLE_BEDROCK_CACHING = True

        # Initialize AWS Secrets Manager client
        self._secrets_client = boto3.client(
            'secretsmanager',
            region_name=self.aws_region
        )

        # Cache for credentials to avoid repeated API calls
        self._credentials_cache: Optional[Dict[str, str]] = None

    def get_mongo_credentials(self) -> Dict[str, str]:
        """
        Fetch MongoDB credentials from AWS Secrets Manager.

        Returns:
            Dict containing username, password, and mongoUrl

        Raises:
            Exception: If failed to fetch credentials
        """
        if self._credentials_cache:
            return self._credentials_cache

        try:
            response = self._secrets_client.get_secret_value(SecretId=self.mongo_creds)
            secret = json.loads(response['SecretString'])

            self._credentials_cache = {
                'username': secret['username'],
                'password': secret['password'],
                'mongoUrl': secret['uri']
            }

            return self._credentials_cache

        except ClientError as error:
            print(f'Failed to fetch MongoDB credentials: {error}')
            raise error
        except json.JSONDecodeError as error:
            print(f'Failed to parse secret JSON: {error}')
            raise error

    def mongo_url(self) -> str:
        """Get MongoDB connection URL."""
        return self._credentials_cache['mongoUrl']

    def mongo_timeout(self) -> int:
        """Get MongoDB timeout in milliseconds."""
        return 5000


# Create a singleton instance
settings = AWSSettings()
