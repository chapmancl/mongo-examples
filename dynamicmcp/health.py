#!/usr/bin/env python3
import asyncio
import logging
from mongodb_vector_server import MongoDBVectorServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MongoDB vector server
mongo_server = MongoDBVectorServer()


async def http_health_check():
    failed, server_info = await mongo_server.get_mongo_info()
    logger.info(f"Health check status: {server_info}")
    if failed:
        raise ConnectionError("MongoDB connection failed")
    return failed
    
if __name__ == "__main__":
    if asyncio.run(http_health_check()):
        # If the health check fails, exit with code 1
        exit(0)
    exit(0)
