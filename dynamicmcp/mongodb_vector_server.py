"""
MongoDB Vector Search MCP Server
A fastMCP MCP server that provides vector search capabilities using MongoDB's $search aggregation pipeline.
"""

import datetime
import json
import asyncio
from typing import Any, Dict, List, Tuple
import logging
import boto3
import requests
from pymongo.errors import PyMongoError
from mongodb_client import MongoDBClient

# Import settings
from settings_aws import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBVectorServer(MongoDBClient):
    def __init__(self):
        super().__init__()
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=settings.aws_region)
        self.tool_config = None
    
    def set_config(self, config: Dict):
        """Set the tool configuration from a dictionary"""
        tool_name = config.get("Name", "UnknownTool")
        print(f"Using settings from tool config {tool_name}")
        self.tool_config = config
        self._db_name = self.tool_config["module_info"]['database']
        self._collection_name = self.tool_config["module_info"]['collection']
        
    def get_current_ip(self) -> str:
        """
        Get the current public IP address using AWS's checkip service.
        useful for logging network issues
        """
        try:
            # Make request to AWS checkip service with timeout
            response = requests.get('https://checkip.amazonaws.com', timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            
            ip_address = response.text.strip()
            
            return ip_address
        except Exception as e:
            logger.error(f"Error fetching current IP: {e}")
            return f"Error fetching current IP: {e}"
            
    
    async def get_mongo_info(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrieve MongoDB connection and collection health information.
        
        This function performs a health check on the MongoDB connection and gathers
        essential database statistics including connection status, collection metrics,
        and server information. It's designed to be used for monitoring, debugging,
        and health check endpoints.
                
        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple containing:
                - bool: Failed flag (True if any operation failed, False if all succeeded)
                - Dict: Health status dictionary containing:
                    - status: "healthy" or "unhealthy"
                    - version: MongoDB server version
                    - mongodb: Nested dict with connection details
                    - error: Error message (only present if operation failed)        
        """
        failed = True
        health_status = {
            "status": "unhealthy",            
            "mongodb": {                
                "database": self._db_name,
                "collection": self._collection_name
            }
        }
        ping_result = None
        try:
            # Ensure connection is established            
            ping_result = await self.ensure_connection()
            
            if ping_result and ping_result.get("ok", 0) == 1.0:
                is_connected = True                
            else:
                is_connected = False
                health_status["error"] = "MongoDB Connection failed"
                failed = True
            
            health_status["mongodb"]["connected"] = is_connected
            if is_connected:
                # Wait for all tasks to complete concurrently
                server_info, collection_stats = await asyncio.gather(                
                    self.client.server_info(),
                    self.db.command("collStats", self._collection_name)
                )            
                
                health_status["version"] = server_info.get("version", "unknown")
                health_status["status"] = "healthy"                
                health_status["mongodb"]["document_count"] = collection_stats.get("count", 0)
                health_status["mongodb"]["size_bytes"] = collection_stats.get("size", 0)        
                # Convert MongoDB Timestamp object to datetime
                cluster_time = ping_result["$clusterTime"]["clusterTime"]
                health_status["mongodb"]["timestamp"] = datetime.datetime.fromtimestamp(cluster_time.time).isoformat()
                failed = False
        except Exception as e:                        
            health_status["error"]= str(e)
            logger.error(f"Health check failed: {e}")

        return (failed, health_status)

    async def generate_embedding(self, text: str) -> list:
        """Generates an embedding for the input text using the given model.
        
        Args:
            text: Input text to embed.
        
        Returns:
            list: Embedding vector (list of floats) produced by the model.
        """
        body = json.dumps({"inputText": text})        
        # Invoke the Bedrock embedding model (e.g., Titan Embeddings) specified in config
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.bedrock_client.invoke_model(
                modelId= settings.EMBEDDING_MODEL_ID, #"amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="application/json",
                body=body
            )
        )
        # Parse the response and extract the embedding vector
        return json.loads(response["body"].read())["embedding"]
    
    async def vector_search(self, query_string: str, filters: list = None, limit: int = 10, num_candidates: int = 100) -> List[Dict[str, Any]]:
        """
        Perform vector search using MongoDB's $search aggregation pipeline
        
        Args:
            query_string: The query string for similarity search
            limit: Maximum number of results to return
            num_candidates: Number of candidates to consider during search
            
        Returns:
            List of search results with similarity scores
        """
        try:
            con_task = self.ensure_connection()
            query_task = self.generate_embedding(query_string)
            con_result, query_vector = await asyncio.gather(con_task, query_task)

            # MongoDB Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.tool_config['tools']['vector_search']['index'],
                        "path": "embedding", 
                        "queryVector": query_vector,
                        "numCandidates": num_candidates,
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": self.tool_config['tools']['vector_search']['projection']
                },
                {
                    "$sort": {
                        "score": -1
                    }
                }
            ]
            
            # Apply filters to narrow the search if provided        
            if filters:
                match_filter = {}
                if len(filters) > 1:
                    # Use $and for multiple filters
                    match_filter = {"$and": []}
                    for key, value in filters:
                        match_filter["$and"].append({key: value})
                        
                else:
                    # Single filter case
                    key, value = filters[0]
                    match_filter[key] = value
                
                # Inject the filter into the pipeline
                pipeline[0]["$vectorSearch"]["filter"] = match_filter
            
            results = []
            async for doc in self.collection.aggregate(pipeline):
                results.append(doc)
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except PyMongoError as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    async def text_search(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform text search using MongoDB's $search aggregation pipeline
        
        Args:
            query_text: The text query for search
            limit: Maximum number of results to return
            
        Returns:
            List of search results with relevance scores
        """
        try:
            await self.ensure_connection()
            # MongoDB Atlas Text Search aggregation pipeline
            pipeline = [
                {
                    "$search": {
                        "index": self.tool_config['tools']['text_search']['index'],
                        "text": {
                            "query": query_text,
                            "path": self.tool_config['tools']['text_search']['fields_searched']
                        }
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "searchScore"}
                    }
                },
                {
                    "$limit": limit
                },
                {
                    "$project": self.tool_config['tools']['text_search']['projection']
                },
                {
                    "$sort": {
                        "score": -1
                    }
                }
            ]
            
            results = []
            async for doc in self.collection.aggregate(pipeline):
                results.append(doc)
            logger.info(f"Text search returned {len(results)} results")
            return results
            
        except PyMongoError as e:
            logger.error(f"Text search failed: {e}")
            raise

    async def agg_pipeline(self, pipeline: List[Dict]) -> List[Dict[str, Any]]:
        """
        Perform MongoDB's aggregation pipeline
        
        Args:
            pipeline: The pipeline to execute (list of aggregation stages)            
            
        Returns:
            List of results
        """
        try:
            await self.ensure_connection()
            # MongoDB Atlas aggregation pipeline                        
            results = []
            async for doc in self.collection.aggregate(pipeline):
                results.append(doc)
            logger.info(f"pipeline returned {len(results)} results")
            return results
            
        except PyMongoError as e:
            logger.error(f"pipeline query failed: {e}")
            raise
