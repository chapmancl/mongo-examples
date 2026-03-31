"""
MongoMCP Package

MongoDB MCP (Model Context Protocol) server package providing:
- MongoDB search capabilities
- Authentication and middleware
- AWS Bedrock LLM integration
- Configuration management

Main Classes:
- MongoDBQueryServer: Core Mongo Query functionality
- MongoMCPMiddleware: Request middleware, config interactions to/from MongoDB, MCP tool management
- BedrockClient: Base AWS Bedrock LLM client
- ServerBedrockClient: Server-specific Bedrock implementation
- MongoTokenVerifier: JWT token authentication
- MongoDBClient: MongoDB connection management

The ``mongomcp.agent`` subpackage contains the Web UI agent classes
(CachedQueryProcessor, ToolRouter, WebUiBedrockClient).  Install with
``pip install mongomcp[agent]`` to pull in agent-only dependencies.
"""

# Import all main classes for easy access
from .mongodb_query_provider import MongoDBQueryServer
from .mongo_mcp_middleware import MongoMCPMiddleware
from .bedrock_client import BedrockClient, ServerBedrockClient
from .mongo_token_verifier import MongoTokenVerifier
from .mongodb_client import MongoDBClient

# Package version
__version__ = "2.0.2"

# Expose main classes at package level
__all__ = [
   "MongoDBQueryServer",
   "MongoMCPMiddleware",
   "BedrockClient",
   "ServerBedrockClient",
   "MongoTokenVerifier",
   "MongoDBClient",
]
