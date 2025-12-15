"""
MongoMCP Package

A comprehensive MongoDB MCP (Model Context Protocol) server package providing:
- MongoDB vector search capabilities
- Authentication and middleware
- AWS Bedrock LLM integration
- Configuration management

Main Classes:
- MongoDBVectorServer: Core vector search functionality
- MongoMCPMiddleware: Request middleware and tool management
- BedrockClient: AWS Bedrock LLM client
- MongoTokenVerifier: JWT token authentication
- MongoDBClient: MongoDB connection management
"""

# Import all main classes for easy access
from .MongoDBVectorServer import MongoDBVectorServer
from .MongoMCPMiddleware import MongoMCPMiddleware
from .BedrockClient import BedrockClient
from .MongoTokenVerifier import MongoTokenVerifier
from .MongoDBClient import MongoDBClient

# Package version
__version__ = "1.0.0"

# Expose main classes at package level
__all__ = [
    "MongoDBVectorServer",
    "MongoMCPMiddleware", 
    "BedrockClient",
    "MongoTokenVerifier",
    "MongoDBClient"
]
