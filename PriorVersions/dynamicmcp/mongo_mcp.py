#!/usr/bin/env python3

import json
from typing import Any, Dict, List, Optional, Annotated
import logging
from pydantic import Field
from pymongo.errors import PyMongoError
from fastmcp import FastMCP
from fastapi import FastAPI
from starlette.responses import JSONResponse
from mongodb_vector_server import MongoDBVectorServer
from middle_tool_response import ListToolsLoggingMiddleware
import traceback
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOOL_NAME = os.getenv('MCP_TOOL_NAME')
list_tools_middleware = ListToolsLoggingMiddleware(TOOL_NAME)

# Initialize the MongoDB vector server
mongo_server = MongoDBVectorServer()
mongo_server.set_config(list_tools_middleware.ANNOTATIONS)

# Create FastMCP server instance
mcp = FastMCP("mongodb-vector-server")
# Add the middleware
mcp.add_middleware(list_tools_middleware)

@mcp.tool()
async def vector_search(
    query_text: Annotated[str, Field(description= "Natural language query describing desired property characteristics.")],    
    limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=50)] = 10,
    num_candidates: Annotated[int, Field(default=100, description="Number of candidates to consider during vector search.", ge=10, le=1000)] = 100,
    filters: Annotated[Optional[List], Field(
        default=None, 
        description= "Optional list of filters to narrow search results."
    )] = None
) -> Dict[str, Any]:
    """Dynamic docstring loaded from JSON configuration"""
    try:
        if not query_text or not isinstance(query_text, str):
            return {"error": "query_vector must be a non-empty array of numbers"}
        
        results = await mongo_server.vector_search(query_text, filters, limit, num_candidates)
        jobj = json.dumps(results, default=str)  # serialize results to JSON string... sometime results don't auto-serialize well so do it now
        return {
            "results": jobj,
            "count": len(results),
            "query_info": {                
                "limit": limit,
                "num_candidates": num_candidates
            }
        }
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        traceback.print_exc()
        return {"error":f"Error executing vector_search: {str(e)}" }

@mcp.tool()
async def text_search(
    query_text: Annotated[str, Field(description="Keywords or phrases to search for across property fields.")],
    limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=100)] = 10
) -> Dict[str, Any]:
    """Dynamic docstring loaded from JSON configuration"""
    try:
        if not query_text:
            return {"error": "query_text is required"}
        
        results = await mongo_server.text_search(query_text, limit)
        jobj = json.dumps(results, default=str) 
        return {
            "results": jobj,
            "count": len(results),
            "query_info": {
                "query_text": query_text,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        return {"error":f"Error executing text_search: {str(e)}"}

@mcp.tool()
async def get_unique_values(
    field: Annotated[str, Field(description="Field name to get unique values for.")]
) -> Dict[str, Any]:
    """Dynamic docstring loaded from JSON configuration"""
    try:
        
        # Use MongoDB aggregation to get unique values
        pipeline = [
            {
                "$group": {
                    "_id": f"${field}",
                    "count": {"$sum": 1}
                }
            },
            {
                "$match": {
                    "_id": {"$ne": None}  # Exclude null values
                }
            },
            {
                "$sort": {
                    "count": -1  # Sort by frequency, most common first
                }
            }
        ]
        
        results = await mongo_server.agg_pipeline(pipeline)
        # Also get total document count for percentage calculation
        total_docs = await mongo_server.collection.count_documents({})
        
        # Add percentage to each result
        for result in results:
            result["percentage"] = round((result["count"] / total_docs) * 100, 2)
        jobj = json.dumps(results, default=str)
        return {
            "field": field,
            "unique_values": jobj,
            "total_unique_count": len(results),
            "total_documents": total_docs
        }
        
    except Exception as e:
        logger.error(f"Get unique values failed: {e}")
        return {"error":f"Error executing get_unique_values: {str(e)}"}

@mcp.tool()
async def get_collection_info() -> Dict[str, Any]:
    """Dynamic docstring loaded from JSON configuration"""
    
    try:
        # Get collection stats and index information
        failed, mongo_info = await mongo_server.get_mongo_info()
        if failed:
            logger.error("Error: Unable to connect to MongoDB")
            return "Error: Unable to connect to MongoDB"

        indexes = []
        async for idx in mongo_server.collection.list_indexes():
            indexes.append(idx)
        
        search_indexes = []
        async for sidx in mongo_server.collection.list_search_indexes():
            search_indexes.append(sidx)
        
        info = {
            "name": mongo_server.tool_name,
            "database": mongo_info["mongodb"]["database"],
            "collection": mongo_info["mongodb"]["collection"],
            "description": mongo_server.description,
            "document_count": mongo_info["mongodb"]["document_count"],
            "size_bytes": mongo_info["mongodb"]["size_bytes"],
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
        return info
        
    except Exception as e:
        logger.error(f"Get collection info failed: {e}")
        return {"error":f"Error executing get_collection_info: {str(e)}"}

@mcp.tool()
async def aggregate_query(
    pipeline: Annotated[List[Dict[str, Any]], Field(description="MongoDB aggregation pipeline as a list of stage objects.")],
    limit: Annotated[Optional[int], Field(default=None, description="Optional limit to apply to the results.", ge=1, le=1000)] = None
) -> Dict[str, Any]:
    """Dynamic docstring loaded from JSON configuration"""
    try:
        # Validate pipeline parameter
        if not pipeline or not isinstance(pipeline, list):
            return {"error":"pipeline must be a non-empty list of aggregation stages"}
        
        # Validate each stage in the pipeline
        for i, stage in enumerate(pipeline):
            if not isinstance(stage, dict):
                return {"error":f"pipeline stage {i} must be a dictionary, got {type(stage)}"}
            if not stage:
                return {"error":f"pipeline stage {i} cannot be empty"}
        
        # Add limit stage if specified and not already present in pipeline
        final_pipeline = pipeline.copy()
        if limit is not None:
            # Check if pipeline already has a $limit stage
            has_limit = any("$limit" in stage for stage in pipeline)
            if not has_limit:
                final_pipeline.append({"$limit": limit})
        
        # Execute the aggregation pipeline
        results = await mongo_server.agg_pipeline(final_pipeline)
        jobj = json.dumps(results,default=str)
        logger.info(f"Aggregation query returned {len(results)} results")
        
        return {
            "results": jobj,
            "count": len(results),
            "query_info": {
                "pipeline": final_pipeline,
                "stages_count": len(final_pipeline),
                "limit_applied": limit
            }
        }
    except PyMongoError as e:
        logger.error(f"Aggregation query failed: {e}")
        return {"error":f"Error executing aggregation pipeline: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON serialization failed: {e}")
        return {"error":f"Error serializing results: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in aggregate_query: {e}")
        return {"error":f"Unexpected error executing aggregate_query: {str(e)}"}


mcp_app = mcp.http_app(path=f"/mcp")
#mcp_app = mcp.http_app(path=f"/{TOOL_NAME}/mcp")
# Key: Pass lifespan to FastAPI
app = FastAPI(title=TOOL_NAME, lifespan=mcp_app.lifespan)

# Root route
@app.get("/")
async def root_endpoint() -> Dict[str, Any]:
    """Root endpoint"""
    list_tools_middleware.load_annotations()  # Ensure annotations are loaded
    return {
        "message": "MongoDB Vector Server MCP",
        "status": "running",
        "available_tools": list_tools_middleware.ALLTOOLS,
        "available_endpoints": [
            f"/{TOOL_NAME}/health" if TOOL_NAME else None,
            f"/{TOOL_NAME}/mcp" if TOOL_NAME else "/mcp",
            "/tools_config",
            "/vectorize"
        ]
    }

# this is for the AWS load balancer health check
@app.get("/{TOOL_NAME}/health")
@app.get("/health")
async def http_health_check() -> Dict[str, Any]:
    """Regular HTTP GET endpoint for health checks"""
    # always return something or else the load balancer will mark it unhealthy and continue to reload the container
    failed, server_info = await mongo_server.get_mongo_info()
    status_code = 200
    #if failed:
    #    status_code = 500        
    return server_info

@app.get("/tools_config")
async def http_get_tools_config() -> Dict[str, Any]:
    """Regular HTTP GET endpoint for health checks"""
    # always return something or else the load balancer will mark it unhealthy and continue to reload the container
    list_tools_middleware.load_annotations()
    results = list_tools_middleware.ALLTOOLS
    return {"available_tools": results}


@app.post("/vectorize")
async def vectorize_text(body: Dict[str, Any]) -> Dict[str, Any]:
    """Dynamic docstring loaded from JSON configuration"""
    try:
        # Extract textChunk from the request body
        text_chunk = body.get("textChunk")
        
        if not text_chunk or not isinstance(text_chunk, str):
            return {
                "error": "textChunk must be a non-empty string in the request body"
            }
        
        vector = await mongo_server.generate_embedding(text_chunk)
        logger.info(f"Vectorization successful for input text of length {len(text_chunk)}")
        return {
            "input_text": text_chunk,
            "vector": vector
        }
        
    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        traceback.print_exc()
        return {
            "error": f"Error executing vectorize_text: {str(e)}"
        }

# Mount the MCP server
app.mount(f"/{TOOL_NAME}", mcp_app)

def main():
    """
    Main entry point for the FastMCP server
    python mongo_mcp.py
    for the container call fastapi directly
    fastapi run mongo_mcp.py 
    fastmcp mongo_mcp.py --transport sse --port 8001
    
    """   
    #mcp.run(transport="sse", host="0.0.0.0", port=8001)
    #mcp.run(transport="sse",  port=8001) # this is for local IDE/Cline integration
    mcp.run(transport="http", host="0.0.0.0", port=8000) # this is for AWS containers  


if __name__ == "__main__":
    main()
