#!/usr/bin/env python3
import asyncio
import logging
import os
from middle_tool_response import ListToolsLoggingMiddleware
from mongodb_vector_server import MongoDBVectorServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MongoDB server
mongo_server = MongoDBVectorServer()
TOOL_NAME = os.getenv('MCP_TOOL_NAME')
list_tools_middleware = ListToolsLoggingMiddleware(TOOL_NAME)
mongo_server.set_config(list_tools_middleware.ANNOTATIONS)


async def http_health_check():
    failed, server_info = await mongo_server.get_mongo_info(True)
    logger.info(f"Health check status: {server_info}")
    if failed:
        raise ConnectionError("MongoDB connection failed")
    return failed
    
if __name__ == "__main__":
    if asyncio.run(http_health_check()):
        # If the health check fails, exit with code 1 
        exit(1)
    exit(0)
