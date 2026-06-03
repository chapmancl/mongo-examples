import os
import json
import logging
from typing import Dict, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class AWSSettings:
    def __init__(self):
        # AWS region where Secrets Manager and Bedrock are deployed
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')

        # AWS Secrets Manager secret name/ARN containing MongoDB credentials
        # Secret must contain: {"username": "...", "password": "...", "uri": "...", "voyageapikey": "..."}
        self.mongo_creds = os.getenv('MONGO_CREDS', 'your-secret-name/your-mongo-creds')

        # MongoDB config collection location (stores MCP tool definitions)
        self.mcp_config_db = "mcp_config"
        self.mcp_config_col = "mcp_tools"

        # LLM model — Bedrock cross-region inference profile ID
        self.LLM_MODEL_ID = os.getenv('LLM_MODEL_ID', 'global.anthropic.claude-sonnet-4-6')

        # Embedding model — "voyage-4" uses Voyage AI via Atlas; "amazon.titan-embed-text-v2:0" uses Bedrock
        self.EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'voyage-4')
        self.QUERY_EMBEDDING_MODEL_ID = os.getenv(
            'QUERY_EMBEDDING_MODEL_ID',
            'voyage-4-lite' if self.EMBEDDING_MODEL_ID.startswith('voyage-') else self.EMBEDDING_MODEL_ID
        )

        # Agent loop limits
        self.LLM_MAX_ITERATIONS = int(os.getenv('LLM_MAX_ITERATIONS', '15'))
        self.LLM_MAX_HISTORY = int(os.getenv('LLM_MAX_HISTORY', '20'))

        # Bedrock prompt caching (reduces cost on repeated system prompts)
        self.ENABLE_CACHE_POINTS = os.getenv('ENABLE_CACHE_POINTS', 'true').lower() in ['1', 'true', 'yes', 'on']
        self.ENABLE_BEDROCK_CACHING = True
        self.ENABLE_MCP_TOOL_CACHING = False  # Not implemented in webui path (APIQueryProcessor)
        self.ENABLE_RESPONSE_CACHING = False
        self.CACHE_TTL = 300
        self.CACHE_NAMESPACE = os.getenv('CACHE_NAMESPACE', 'aws')  # Isolates cache from local builds

        # Experimental routing features (leave False unless you know what these do)
        self.AI_TOOL_ROUTING = False
        self.TOOL_ROUTING = False

        # URL of the MCP server (mongo_mcp.py or deployed service)
        self.mongo_mcp_root = os.getenv('MONGO_MCP_ROOT', 'http://localhost:8000')

        # System prompt injected into every Bedrock conversation
        self.BEDROCK_SYSTEM_PROMPT_TEXTS = [
            "***IMPORTANT: DO NOT recall sessions by username until you have confirmed the username with the user. DO NOT ASSUME you know the Username. Default username is demo-user",
            "***IMPORTANT: STRATEGY FIRST: Before any tool call execute memory_strategy_recall to find applicable patterns THEN EXECUTE the found pattern. Validated and high scoring patterns CANNOT be ignored.***",
            "***IMPORTANT: All output should be Markdown formatted for display within a div in an existing webpage. Do not include html, head, or body tags. Only include the inner content. Always use Markdown formatting.",
        ]

        # Static fallback auth token for the MCP server (generate via the MCP server's token endpoint)
        # Overridden by Cognito if COGNITO_CLIENT_ID / COGNITO_USERNAME / COGNITO_PASSWORD are set.
        self.AUTH_TOKEN = os.getenv('MCP_AUTH_TOKEN', 'your-static-jwt-token-here')

        # Optional Cognito auth — takes precedence over AUTH_TOKEN when all three vars are set.
        _cognito_client_id = os.getenv('COGNITO_CLIENT_ID')
        _cognito_username  = os.getenv('COGNITO_USERNAME')
        _cognito_password  = os.getenv('COGNITO_PASSWORD')
        if _cognito_client_id and _cognito_username and _cognito_password:
            from cognito_auth import CognitoTokenProvider
            self._cognito: object = CognitoTokenProvider(
                region=self.aws_region,
                client_id=_cognito_client_id,
                username=_cognito_username,
                password=_cognito_password,
            )
            logger.info("Cognito auth configured for user %s", _cognito_username)
        else:
            self._cognito = None
            logger.warning("Cognito env vars not set — falling back to static AUTH_TOKEN")

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

    def get_auth_token(self) -> str:
        """Return a Cognito JWT if configured, otherwise fall back to the static AUTH_TOKEN."""
        if self._cognito is not None:
            return self._cognito.get_token()
        return self.AUTH_TOKEN

    def mongo_url(self) -> str:
        return self._credentials_cache['mongoUrl']

    def mongo_timeout(self) -> int:
        return 5000

    def mongo_voyage_apikey(self) -> str:
        return self._credentials_cache.get('voyageapikey', None)


# Create a singleton instance
settings = AWSSettings()


def __getattr__(name: str):
    """Backward-compatible module attribute access via singleton settings."""
    if hasattr(settings, name):
        return getattr(settings, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
