"""
MongoDB Vector Search MCP Server
A fastMCP MCP server that provides vector search capabilities using MongoDB's $search aggregation pipeline.
"""

import datetime
import asyncio
import traceback
from typing import Any, Dict, List, Tuple
import logging
from pymongo.errors import PyMongoError
from .MongoDBClient import MongoDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBVectorServer(MongoDBClient):
    """
    MongoDB wrapper class translates MCP Server functions to MongoDB operations.
    This started as just vector, now has some additional search capabilities.
    """
    def __init__(self, settings):
        super().__init__(settings=settings)
        self.tool_config = None
        self.tool_name = None
        self.description = "MongoDB Vector Search MCP Server"
        self.available_collections = [self._collection_name]
    
    def set_config(self, config: Dict) -> None:
        """Set the tool configuration from a dictionary. this overrides the default settings"""
        if config is None:
            raise ValueError("Config cannot be None. Check env variables and AWS secrets.")
        
        self.tool_config = config
        self.tool_name  = config["Name"]
        self.description = config["module_info"]["description"]
        print(f"Using settings from tool config {self.tool_name}")
        super().set_config(config["module_info"])
            
    async def get_mongo_info(self, shortResponse=False) -> Tuple[bool, Dict[str, Any]]:
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
            "toolname": self.tool_name,
            "description": self.description,
            "mongodb": {                
                "database": self._db_name,
                "collections": {}
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
            #health_status["mongodb"]["collections"] = self.available_collections
            
            if is_connected:
                # get all the user collections ignoring system collections
                all_collection_names = await self.db.list_collection_names()
                self.available_collections = [
                    name for name in all_collection_names
                    if not (name.startswith('system.') or name.startswith('_'))
                ]

                # Convert MongoDB Timestamp object to datetime
                cluster_time = ping_result["$clusterTime"]["clusterTime"]
                health_status["mongodb"]["timestamp"] = str(datetime.datetime.fromtimestamp(cluster_time.time).isoformat())
                health_status["status"] = "healthy"                
                failed = False
                
                if not shortResponse:
                    # Get server info and collection stats
                    collection_stats = await asyncio.gather(
                        *[self.db.command("collStats", collection_name) for collection_name in self.available_collections]                        
                    )   
                    # List all available collections in the database
                    for stats, collection_name in zip(collection_stats, self.available_collections):
                        health_status["mongodb"]["collections"][collection_name] = {
                            "document_count" : stats.get("count", 0),
                            "size_bytes" : stats.get("size", 0)
                        }
                    
                
        except Exception as e:                        
            health_status["error"]= str(e)
            logger.error(f"Health check failed: {e}")
            traceback.print_exc()

        return (failed, health_status)

    async def get_collection_info(self) -> Dict[str, Any]:
        """Retrieve information about the current MongoDB collection."""
        failed, info = await self.get_mongo_info(shortResponse=False)
        if failed:
            raise ConnectionError(f"Failed to retrieve MongoDB info: {info.get('error','Unknown error')}")
        
        for coll in self.available_collections:
            indexes = []
            async for idx in self.get_collection(coll).list_indexes():
                indexes.append(idx)
            
            search_indexes = []
            async for sidx in self.get_collection(coll).list_search_indexes():
                search_indexes.append(sidx)
            
            coll_info = {                                
                "indexes": [
                    {
                        "name": idx.get("name"),
                        "key": idx.get("key"),
                        "type": idx.get("type", "standard")
                    } for idx in indexes
                ],
                "search_indexes": [
                    sidx for sidx in search_indexes
                ]
            }        
            #print(info)
            if "collections" not in info["mongodb"]:
                info["mongodb"]["collections"] = {}
            if coll not in info["mongodb"]["collections"]:
                info["mongodb"]["collections"][coll] = {}
            info["mongodb"]["collections"][coll].update(coll_info)        
        return info
    
    async def vector_search(self, collection: str, vector_qry: str, filters: list = None, limit: int = 10, num_candidates: int = 100) -> List[Dict[str, Any]]:
        """
        Perform vector search using MongoDB's $search aggregation pipeline
        
        Args:
            vector_qry: The vectorized embeddings of the query string for similarity search
            limit: Maximum number of results to return
            num_candidates: Number of candidates to consider during search
            
        Returns:
            List of search results with similarity scores
        """
        try:
            await self.ensure_connection()

            # MongoDB Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.tool_config['tools']['vector_search']['index'],
                        "path": "embedding", 
                        "queryVector": vector_qry,
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
            async for doc in self.get_collection(collection).aggregate(pipeline):
                results.append(doc)
            #logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except PyMongoError as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    async def text_search(self, collection: str, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
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
            async for doc in self.get_collection(collection).aggregate(pipeline):
                results.append(doc)
            logger.info(f"Text search returned {len(results)} results")
            return results
            
        except PyMongoError as e:
            logger.error(f"Text search failed: {e}")
            raise

    async def agg_pipeline(self, collection: str, pipeline: List[Dict]) -> List[Dict[str, Any]]:
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
            async for doc in self.get_collection(collection).aggregate(pipeline):
                results.append(doc)
            return results
            
        except PyMongoError as e:
            logger.error(f"pipeline query failed: {e}")
            raise
