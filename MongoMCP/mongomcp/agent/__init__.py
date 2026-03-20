"""
mongomcp.agent — Web UI agent subpackage.

Contains the query processor, tool router, and WebUI Bedrock client.
These classes depend on additional packages (flask, pydantic, etc.)
that the MCP server does not need. Install with:

    pip install mongomcp[agent]
"""

from .cached_query_processor import CachedQueryProcessor
from .tool_router import ToolRouter
from .webui_bedrock_client import WebUiBedrockClient

__all__ = [
    "CachedQueryProcessor",
    "ToolRouter",
    "WebUiBedrockClient",
]
