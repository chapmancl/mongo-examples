#!/usr/bin/env python3
"""
MongoDB Vector Search MCP Server
A fastMCP MCP server that provides vector search capabilities using MongoDB's $search aggregation pipeline.
"""

import json
from typing import Any, Dict, List, Optional, Annotated
import logging
from pydantic import Field
from pymongo.errors import PyMongoError
from fastmcp import FastMCP
from starlette.responses import JSONResponse
from MongoDBVectorServer import MongoDBVectorServer
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MongoDB vector server
mongo_server = MongoDBVectorServer()

# Create FastMCP server instance
mcp = FastMCP("mongodb-vector-server")

# this is for the AWS load balancer health check
@mcp.custom_route("/health", methods=["GET"])
async def http_health_check(request):
    """Regular HTTP GET endpoint for health checks"""
    # always return something or else the load balancer will mark it unhealthy and continue to reload the container
    Failed,server_info = await mongo_server.get_mongo_info()
    staus_code = 200
    #if Failed:
    #    staus_code = 500        
    return JSONResponse(server_info, status_code=staus_code)


@mcp.tool()
async def vector_search(
    query_text: Annotated[str, Field(description="Natural language query describing desired property characteristics.")],    
    limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=50)] = 10,
    num_candidates: Annotated[int, Field(default=100, description="Number of candidates to consider during vector search.", ge=10, le=1000)] = 100,
    filters: Annotated[Optional[List], Field(
        default=None, 
        description="""Optional list of filters to narrow search results. Each filter should be a list with [field, value] format. 
        Use filters whenever possible to improve search relevance and performance. Use the get_unique_values tool to discover all available filter values for any field.
        This should be the primary search tool for most use cases.
        Supported fields include: bedrooms, beds, address.country_code, address.suburb, address.market        
        Available filter fields with example values from the collection:
        
        **bedrooms** (integer)
        **beds** (integer)
        **address.country_code** (string 2 char country abbreviation)
        **address.suburb** (neighborhood)        
        **address.market** (city or region) 
        **Example usage:**
        ["beds", 2], ["address.country_code", "US"]
        ["address.market", "New York"]
        ["address.country_code", "CA"]
        ["beds", 3], ["address.country_code", "AU"]
        """
    )] = None
) -> str:
    """
    Perform semantic vector similarity search on MongoDB collection using AI embeddings.
    
    This tool converts natural language queries into vector embeddings using Amazon Titan 
    Embeddings v2 and finds semantically similar property listings. It's ideal for finding 
    properties based on meaning and context rather than exact keyword matches.
    
    Use cases:
    - "Find a cozy apartment near Central Park" (semantic understanding of 'cozy')
    - "Spacious family home with modern amenities" (understands family needs)
    - "Romantic getaway with city views" (contextual search)
    
    Args:
        query_text: Natural language query describing desired property characteristics.
                   The text will be automatically converted to a 1024-dimensional vector.
        limit: Maximum number of results to return (default: 10, max recommended: 50) Higher values may impact performance.
        num_candidates: Number of candidates to consider during search (default: 100, 
                       higher values improve recall but increase latency)
        filters: Optional list of pre filters to narrow vector space. Each filter should be a list
                 with [field, value] format. Use filters whenever possible to improve relevance
    
    Returns:
        JSON with results array containing matching properties ranked by semantic similarity,
        each with a similarity score (0.0-1.0, higher is more similar).
    """
    try:
        if not query_text or not isinstance(query_text, str):
            return "Error: query_vector must be a non-empty array of numbers"
        
        results = await mongo_server.vector_search(query_text, filters, limit, num_candidates)
        
        return json.dumps({
            "results": results,
            "count": len(results),
            "query_info": {                
                "limit": limit,
                "num_candidates": num_candidates
            }
        }, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        traceback.print_exc()
        return f"Error executing vector_search: {str(e)}"

@mcp.tool()
async def text_search(
    query_text: Annotated[str, Field(description="Keywords or phrases to search for across property fields.")],
    limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=100)] = 10
) -> str:
    """
    Perform traditional keyword-based text search on MongoDB collection using Atlas Search.
    Do not use this tool for semantic searches - use vector_search instead.

    This tool searches for exact keyword matches and full-text indexing with stemming and fuzzy matching. 
    fields include: name, description, summary, amenities, beds, and property_type. Use only for finding
    properties with specific features or characteristics using precise terms.
    
    Use cases:
    - "2 bedroom apartment WiFi kitchen" (exact feature matching)
    - "Manhattan studio elevator" (location and amenity search)
    - "house parking garage 3 beds" (specific requirements)
    - "Airbnb Brooklyn" (location-based search)
    
    Search behavior:
    - Searches across: name, description, summary, amenities, beds, property_type fields
    - Uses MongoDB Atlas Search for full-text indexing and relevance scoring
    - Returns results ranked by text relevance score
    - Case-insensitive matching with stemming and fuzzy matching
    
    Args:
        query_text: Keywords or phrases to search for. Can include property features,
                   locations, amenities, or any descriptive terms.
        limit: Maximum number of results to return (default: 10, max recommended: 100) 
    
    Returns:
        JSON with results array containing matching properties ranked by text relevance,
        each with a relevance score indicating how well it matches the search terms.
    """
    try:
        if not query_text:
            return "Error: query_text is required"
        
        results = await mongo_server.text_search(query_text, limit)
        
        return json.dumps({
            "results": results,
            "count": len(results),
            "query_info": {
                "query_text": query_text,
                "limit": limit
            }
        }, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        return f"Error executing text_search: {str(e)}"

@mcp.tool()
async def get_unique_values(
    field: Annotated[str, Field(description="Field name to get unique values for.")]
) -> str:
    """
    Get unique values for a specific field in the MongoDB collection.
    
    This tool retrieves all unique values for a specified field to determine what filter 
    options are available when using vector_search.
    Use cases:
    - Discover available property types: get_unique_values("property_type")
    - Find all markets/cities: get_unique_values("address.market") 
    - See available room types: get_unique_values("room_type")
    - Check bed count options: get_unique_values("beds")
    - Explore any field values: get_unique_values("amenities")
    
    Args:
        field: The field name to get unique values for. Can be nested fields using dot notation
               (e.g., "address.market", "address.country_code").
    
    Returns:
        JSON with unique values array for the specified field, along with count information.
    """
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
        
        return json.dumps({
            "field": field,
            "unique_values": results,
            "total_unique_count": len(results),
            "total_documents": total_docs
        }, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Get unique values failed: {e}")
        return f"Error executing get_unique_values: {str(e)}"

@mcp.tool()
async def get_collection_info() -> str:
    """
    Get comprehensive information about the MongoDB collection, database statistics, and search capabilities.
    
    This tool provides essential metadata about the current database connection and collection status,
    including document counts, storage size, available indexes, and search configuration. Use this
    tool to understand the current state of the data and search capabilities.
    
    Information provided:
    - Database and collection names currently connected to
    - Total number of documents in the collection
    - Storage size in bytes
    - List of all available indexes with their configurations
    - Vector search index name and status
    - Text search index name and status
    
    Use cases:
    - Check if the collection has data before searching
    - Verify search indexes are properly configured
    - Monitor collection size and growth
    - Troubleshoot search performance issues
    - Understand available search capabilities
    
    Returns:
        JSON containing complete collection metadata including document count, size,
        indexes, and search configuration details.
    """
    try:
        # Get collection stats and index information
        mfail, mongo_info = await mongo_server.get_mongo_info()
        if mfail:
            logger.error("Error: Unable to connect to MongoDB")
            return "Error: Unable to connect to MongoDB"

        indexes = []
        async for idx in mongo_server.collection.list_indexes():
            indexes.append(idx)
        
        search_indexes = []
        async for sidx in mongo_server.collection.list_search_indexes():
            search_indexes.append(sidx)
        
        info = {
            "database": mongo_info["mongodb"]["database"],
            "collection": mongo_info["mongodb"]["collection"],
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
        return json.dumps(info, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Get collection info failed: {e}")
        return f"Error executing get_collection_info: {str(e)}"

@mcp.tool()
async def aggregate_query(
    pipeline: Annotated[List[Dict[str, Any]], Field(description="MongoDB aggregation pipeline as a list of stage objects. Each stage should be a dictionary representing a MongoDB aggregation stage like $match, $group, $project, $sort, etc.")],
    limit: Annotated[Optional[int], Field(default=None, description="Optional limit to apply to the results. If not specified, no limit will be applied. Recommended for large result sets to avoid memory issues.", ge=1, le=1000)] = None
) -> str:
    """
    Execute a custom MongoDB aggregation pipeline query on the collection.
    
    This tool allows you to run complex MongoDB aggregation queries with full flexibility.
    You can use any MongoDB aggregation stage including $match, $group, $project, $sort,
    $lookup, $unwind, $addFields, and many others. This is the most powerful tool for
    complex data analysis and custom queries.
    see here for details:
    https://www.mongodb.com/docs/manual/reference/mql/aggregation-stages/#std-label-aggregation-pipeline-operator-reference
    
    Common aggregation stages:
    - $match: Filter documents (like WHERE in SQL)
    - $group: Group documents and perform aggregations
    - $project: Include/exclude fields or create computed fields
    - $sort: Sort documents by specified fields
    - $limit: Limit the number of results
    - $skip: Skip a number of documents
    - $addFields: Add new fields to documents
    - $unwind: Deconstruct array fields
    - $lookup: Join with other collections
    
    Example pipelines:
    1. Count properties by type:
    [{"$group": {"_id": "$property_type", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]    
    2. Find average beds by country:
    [{"$group": {"_id": "$address.country_code", "avg_beds": {"$avg": "$beds"}}}, {"$sort": {"avg_beds": -1}}]
    3. Properties with 2+ beds in specific markets:
    [{"$match": {"beds": {"$gte": 2}, "address.market": {"$in": ["New York", "Sydney"]}}}, {"$project": {"name": 1, "beds": 1, "address.market": 1, "price": 1}}]    
    4. Top 10 most expensive properties:
    [{"$match": {"price": {"$exists": true, "$ne": null}}}, {"$sort": {"price": -1}}, {"$limit": 10}, {"$project": {"name": 1, "price": 1, "address.market": 1}}]
    
    Args:
        pipeline: List of aggregation stage dictionaries. Each stage performs a specific
                 operation on the data as it flows through the pipeline.
        limit: Optional limit for results. Use this for large result sets to prevent
               memory issues. If your pipeline already includes $limit, this parameter
               will add an additional limit at the end.
    
    Returns:
        JSON with results array containing the aggregation results, along with metadata
        about the query execution including result count and pipeline information.
    """
    try:
        # Validate pipeline parameter
        if not pipeline or not isinstance(pipeline, list):
            return "Error: pipeline must be a non-empty list of aggregation stages"
        
        # Validate each stage in the pipeline
        for i, stage in enumerate(pipeline):
            if not isinstance(stage, dict):
                return f"Error: pipeline stage {i} must be a dictionary, got {type(stage)}"
            if not stage:
                return f"Error: pipeline stage {i} cannot be empty"
        
        # Add limit stage if specified and not already present in pipeline
        final_pipeline = pipeline.copy()
        if limit is not None:
            # Check if pipeline already has a $limit stage
            has_limit = any("$limit" in stage for stage in pipeline)
            if not has_limit:
                final_pipeline.append({"$limit": limit})
        
        # Execute the aggregation pipeline
        results = await mongo_server.agg_pipeline(final_pipeline)
        
        logger.info(f"Aggregation query returned {len(results)} results")
        
        return json.dumps({
            "results": results,
            "count": len(results),
            "query_info": {
                "pipeline": final_pipeline,
                "stages_count": len(final_pipeline),
                "limit_applied": limit
            }
        }, indent=2, default=str)
        
    except PyMongoError as e:
        logger.error(f"Aggregation query failed: {e}")
        return f"Error executing aggregation pipeline: {str(e)}"
    except json.JSONDecodeError as e:
        logger.error(f"JSON serialization failed: {e}")
        return f"Error serializing results: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in aggregate_query: {e}")
        return f"Unexpected error executing aggregate_query: {str(e)}"



def main():
    """
    Main entry point for the FastMCP server
    python mongo_mcp.py

    For local testing or to bypass this function use fastmcp:
    fastmcp run mongo_mcp.py --transport sse --port 8001
    
    """    
    #mcp.run(transport="sse", host="0.0.0.0", port=8001)
    #mcp.run(transport="sse",  port=8001) # this is for local IDE/Cline integration
    mcp.run(transport="http", host="0.0.0.0", port=8000) # this is for AWS containers  

if __name__ == "__main__":
    main()
