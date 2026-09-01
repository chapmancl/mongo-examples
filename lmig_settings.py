import os
import json
import logging
from typing import Dict, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class LMIGSettings:
    def __init__(self):
        # Keep these defaults aligned with lmig_settings.py so this file is a drop-in local replacement.
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.TOOL_NAME = os.getenv('MCP_TOOL_NAME', 'AirbnbSearch')
        self.IS_LOCAL = json.loads(os.getenv('USE_LOCAL_MODE', 'true').lower())     
        # Embedding model — "voyage-4" uses Voyage AI via Atlas; "amazon.titan-embed-text-v2:0" uses Bedrock
        #self.EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'voyage-4')
        self.EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v2:0')
        self.QUERY_EMBEDDING_MODEL_ID = os.getenv(
            'QUERY_EMBEDDING_MODEL_ID',
            'voyage-4-lite' if self.EMBEDDING_MODEL_ID.startswith('voyage-') else self.EMBEDDING_MODEL_ID
        )
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
        self.LLM_MAX_HISTORY = int(os.getenv('LLM_MAX_HISTORY', '20'))  # max messages kept in history before trimming
        self.ENABLE_CACHE_POINTS = os.getenv('ENABLE_CACHE_POINTS', 'true').lower() in ['1', 'true', 'yes', 'on']
        self.ENABLE_BEDROCK_CACHING = True
        self.ENABLE_MCP_TOOL_CACHING = False
        self.ENABLE_RESPONSE_CACHING = False
        self.CACHE_TTL = 300
        self.CACHE_NAMESPACE = os.getenv('CACHE_NAMESPACE', 'aws')  # Isolates cache from local builds

        # Experimental routing features (leave False unless you know what these do)
        self.AI_TOOL_ROUTING = False
        self.TOOL_ROUTING = False

        #self.BEDROCK_KEY = os.getenv('SECRET__BB_NEW_SECRET__AIENDPOINT', 'NO_KEY') 

        # Azure AD tenant for the client-credentials token grant.
        self.AZURE_TENANT_ID = os.getenv('AZURE_TENANT_ID', '08a83339-90e7-49bf-9075-957ccd561bf1')
        self.LIBERTY_TROUX_ID = "B7135EEA-F02B-44BC-B19E-70D195A9C6E1"
        self.CERTIFICATE_PATH = "certs/lm-ca-bundle.crt" # avoid SSL verify err, local issuer certificate
        self.LIBERTY_AIR_URL = "https://test-libertyair.lmig.com"
        self.LIBERTY_AIR_CLIENT = os.getenv('SECRET__GEM_LIBERTYAIR_CLIENT__ID', 'air_client_id')
        self.LIBERTY_AIR_SECRET = os.getenv('SECRET__GEM_LIBERTYAIR_CLIENT__SECRET','air_secret')
        self.LIBERTY_CLIENT_SCOPE = "87d1c382-6128-4150-aacf-bb624a9f2748/.default"
        self.LIBERTY_TROUX_UUID = "B7135EEA-F02B-44BC-B19E-70D195A9C6E1"
        
        self.mongo_mcp_root = os.getenv('MONGO_MCP_ROOT', 'http://localhost:8000')
        self.AUTH_TOKEN = os.getenv('SECRET__MCP_AUTH_TOKEN__AUTH_TOKEN', 'jwt-for-mcp-access')
        # System prompt injected into every Bedrock conversation
        self.BEDROCK_SYSTEM_PROMPT_TEXTS = [
            "***IMPORTANT: DO NOT recall sessions by username until you have confirmed the username with the user. DO NOT ASSUME you know the Username. Default username is demo-user",
            "***IMPORTANT: STRATEGY FIRST: Before any tool call execute memory_strategy_recall to find applicable patterns THEN EXECUTE the found pattern. Validated and high scoring patterns CANNOT be ignored.***",
            "***IMPORTANT: All output should be Markdown formatted for display within a div in an existing webpage. Do not include html, head, or body tags. Only include the inner content. Always use Markdown formatting.",
        ]

        # Hardcoded credentials for local development only.
        self._credentials_cache: Dict[str, str] = {
            'username': os.getenv('SECRET__GEM_EXPOSURE_REPOSITORY_MONGO_DB_TEST__READ_WRITE_SHORT__USERNAME', 'bbmcp_user'),
            'password': os.getenv('SECRET__GEM_EXPOSURE_REPOSITORY_MONGO_DB_TEST__READ_WRITE_SHORT__PASSWORD', '<secret>'),
            'mongoUrl': os.getenv('MONGO_URI', 'mongodb+srv://demo1.sf56l.mongodb.net'),
            'voyageapikey': os.getenv('SECRET__VOYAGE__VOYAGEAPIKEY', '<secret>')
        }
        # Pull in API Key secret
        #os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.BEDROCK_KEY
        
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

        # Capture temproary tokens
        # BJB - use AIR only os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.BEDROCK_KEY
        os.environ["VOYAGE_API_KEY"] = self._credentials_cache["voyageapikey"]
        
    def get_mongo_credentials(self) -> Dict[str, str]:
        """
        Return MongoDB credentials from local hardcoded values.

        Returns:
            Dict containing username, password, and mongoUrl
        """
        return self._credentials_cache

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
settings = LMIGSettings()
