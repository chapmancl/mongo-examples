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
from .prompt_agent import PromptAgent
from .mcp_tools import register_agent_tools, get_agent_bedrock_toolspecs
from .function_builder import register_function_builder_tools, get_function_builder_toolspecs
from .external_api import register_external_api_tools, get_external_api_toolspecs

__all__ = [
    "CachedQueryProcessor",
    "ToolRouter",
    "WebUiBedrockClient",
    "PromptAgent",
    "register_agent_tools",
    "get_agent_bedrock_toolspecs",
    "register_function_builder_tools",
    "get_function_builder_toolspecs",
    "register_external_api_tools",
    "get_external_api_toolspecs",
]
