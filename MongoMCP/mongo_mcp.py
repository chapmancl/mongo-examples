import json
import time
from typing import Any, Dict, List, Optional, Annotated
import logging
from pydantic import Field
from pymongo.errors import PyMongoError
from fastmcp import FastMCP
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.responses import JSONResponse
from AWS_settings import settings
from MongoMCP import MongoDBVectorServer, MongoMCPMiddleware, BedrockClient, MongoTokenVerifier
import traceback
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
main component flow:
1. MongoMCPMiddleware: Connects to MongoDB config database, loads tool configurations, and prep token authorization.
2. MongoDBVectorServer: Implements core MongoDB query functionalities for the specific tool_name from env
3. BedrockClient: Handles AWS Bedrock LLM interactions and tool integrations.
4. instantiate FastMCP server with MongoMCPMiddleware, MongoDBVectorServer, MongoTokenVerifier
5. instantiate FastAPI app, mounts FastMCP app
6. define additional endpoints for health checks, tool configuration retrieval, settings reset, LLM invocation, and text vectorization.

"""

TOOL_NAME = os.getenv('MCP_TOOL_NAME')
mongo_middleware: MongoMCPMiddleware
mongo_server: MongoDBVectorServer
auth_provider = None

def setup_from_mongo():
    """
     setup the list tools middleware to load the tool configuration from mongo
     this will also verify we can connect to mongo before starting the server
     the middleware will be added to the MCP server instance below to intercept tool calls
    """
    global mongo_middleware
    global mongo_server
    global auth_provider
    mongo_middleware = None
    mongo_server = None
    auth_provider = None
    failed = False
    error = None

    # load or reload the mongo middleware and server config
    # we do this to get fresh settings from mongo if reset_settings is called
    try:
        mongo_middleware = MongoMCPMiddleware(TOOL_NAME, settings)
        if mongo_middleware.ANNOTATIONS:
            # Initialize the MongoDB vector server
            mongo_server = MongoDBVectorServer(settings)
            mongo_server.set_config(mongo_middleware.ANNOTATIONS)
            auth_provider = MongoTokenVerifier(mongo_middleware)
        else:
            failed = True
    except ConnectionError as e:
        failed = True
        error = e
    if failed:
        logger.error(f"Failed to get configuration from MongoDB. Will wait for 10s before retry.\r\n {KeyError}")
        time.sleep(10)
        sys.exit(1)

setup_from_mongo()
# Create FastMCP server instance with bearer token authentication
mcp = FastMCP("mongodb-vector-server", include_fastmcp_meta=False, auth=auth_provider)
mcp.add_middleware(mongo_middleware)
llm_client = BedrockClient(settings)

@mcp.tool()
async def upsert_document(
    collection: Annotated[str, Field(description="Name of the MongoDB collection to upsert into.")],
    filter: Annotated[Dict, Field(description="Filter to find the document to update.")],
    update: Annotated[Dict, Field(description="Update data for the document.")],
) -> Dict[str, Any]:
    """Upsert a document in the specified MongoDB collection."""
    # TODO: validate write permissions from the token scopes
    try:
        doc_id = await mongo_server.upsert_document(collection, filter, update)
        return {            
            "message": f"Document {doc_id} upserted successfully in collection '{collection}'."
        }
    except (ValueError,PyMongoError) as e:
        logger.error(f"Upsert document failed: {e}")
        return {"error": f"Error executing upsert_document: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in upsert_document: {e}")
        traceback.print_exc()
        return {"error": f"Unexpected error executing upsert_document: {str(e)}"}

@mcp.tool()
async def vector_search(
    collection: Annotated[str, Field(description="Name of the MongoDB collection to search in.")],
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
        
        #TODO: validate collection exists and matches tool config, validate vector index exists on collection

        # incoming input is text, we need a vector for search. Use the LLM client to generate the embedding
        vector_qry = await llm_client.generate_embedding(query_text)  
        results = await mongo_server.vector_search(collection, vector_qry, filters, limit, num_candidates)
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
    collection: Annotated[str, Field(description="Name of the MongoDB collection to search in.")],
    query_text: Annotated[str, Field(description="Keywords or phrases to search for across property fields.")],
    limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=100)] = 10
) -> Dict[str, Any]:
    """Dynamic docstring loaded from JSON configuration"""
    try:
        if not query_text:
            return {"error": "query_text is required"}
        
        #TODO: validate collection exists, validate text search index exists on collection

        results = await mongo_server.text_search(collection, query_text, limit)
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
    collection: Annotated[str, Field(description="Name of the MongoDB collection to search in.")],
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
        
        results = await mongo_server.agg_pipeline(collection, pipeline)
        # Also get total document count for percentage calculation
        total_docs = await mongo_server.get_collection(collection).count_documents({})
        
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
        info = await mongo_server.get_collection_info()       
        return info
        
    except Exception as e:
        logger.error(f"Get collection info failed: {e}")
        traceback.print_exc()
        return {"error":f"Error executing get_collection_info: {str(e)}"}

@mcp.tool()
async def aggregate_query(
    collection: Annotated[str, Field(description="Name of the MongoDB collection to search in.")],
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
        results = await mongo_server.agg_pipeline(collection, final_pipeline)
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


#***********  BEGIN FASTAPI SECTION  ***************

# We have our tools, mount the mcp to fastapi and setup our fastapi authentication
# everything after this should be FastAPI endpoints.
mcp_app = mcp.http_app(path=f"/mcp")
app = FastAPI(title=TOOL_NAME, lifespan=mcp_app.lifespan)
security_token = HTTPBearer()

# Tool handler function for mapping tool names to functions
async def tool_handler(toolname: str, tool_input: dict) -> dict:
    """
    Wrapper function that maps tool names to the appropriate MCP tool functions
    LLM tool use requests will flow through here
    
    Args:
        toolname: Name of the tool to execute
        tool_input: Input parameters for the tool
        
    Returns:
        dict: Result from the tool execution
    """
    
    try:
        if toolname == "upsert_document":
            return await upsert_document.fn(
                collection=tool_input.get("collection"),
                filter=tool_input.get("filter"),
                update=tool_input.get("update")
            )
        elif toolname == "vector_search":
            return await vector_search.fn(
                collection=tool_input.get("collection"),
                query_text=tool_input.get("query_text"),
                limit=tool_input.get("limit", 10),
                num_candidates=tool_input.get("num_candidates", 100),
                filters=tool_input.get("filters")
            )
        elif toolname == "text_search":
            return await text_search.fn(
                collection=tool_input.get("collection"),
                query_text=tool_input.get("query_text"),
                limit=tool_input.get("limit", 10)
            )
        elif toolname == "get_unique_values":
            return await get_unique_values.fn(
                collection=tool_input.get("collection"),
                field=tool_input.get("field")
            )
        elif toolname == "get_collection_info":
            return await get_collection_info.fn()
        elif toolname == "aggregate_query":
            return await aggregate_query.fn(
                collection=tool_input.get("collection"),
                pipeline=tool_input.get("pipeline"),
                limit=tool_input.get("limit")
            )
        else:
            return {"error": f"Unknown tool: {toolname}"}
    except Exception as e:
        logger.error(f"Tool handler error for {toolname}: {e}")
        traceback.print_exc()  # Prints full stack trace
        return {"error": f"Error executing {toolname}: {str(e)}"}

# Root route
@app.get("/")
async def root_endpoint() -> Dict[str, Any]:
    """Root endpoint"""
    mongo_middleware.load_annotations()  # Ensure annotations are loaded
    return {
        "message": "MongoDB Vector Server MCP",
        "status": "running",
        "available_tools": mongo_middleware.ALLTOOLS,
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
    failed, server_info = await mongo_server.get_mongo_info(False)
    status_code = 200
    #if failed:
    #    status_code = 500        
    return server_info

@app.get("/tools_config")
async def http_get_tools_config() -> Dict[str, Any]:
    """Regular HTTP GET endpoint for health checks"""
    # always return something or else the load balancer will mark it unhealthy and continue to reload the container
    mongo_middleware.load_annotations()
    results = mongo_middleware.ALLTOOLS
    return {"available_tools": results}

@app.get("/{TOOL_NAME}/reset")
async def reset_settings(credentials: HTTPAuthorizationCredentials = Depends(security_token)
                     ) -> Dict[str, Any]:    
    """
    reload the settings. this will pull new configs from mongo and update the server with any changes
    """ 
    (allowed, agent_rec) = mongo_middleware.check_authorization(credentials.credentials)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info(f"Begin settings reset for {TOOL_NAME}")
    global llm_client
    llm_client = None
    output = {"action":"reset settings"}
    try:
        setup_from_mongo()
        llm_client = BedrockClient(settings)
        mcp_tools = await mcp.get_tools()
        tools_config = mongo_middleware.get_llm_tools(mcp_tools)
        llm_client.configure_tools(tools_config, tool_handler)        

        output["result"] = "success"
        logger.info(f"Finished settings reset for {TOOL_NAME}: Success")
    except Exception as e:
        logger.error(f"reset_settings failed: {e}")
        traceback.print_exc()        
        output["error"] = f"Error executing invoke_llm: {str(e)}"
        output["result"] = "failed"
        logger.info(f"Finished settings reset for {TOOL_NAME}: Failed")
        return JSONResponse(output, 500)

    return output

@app.post("/{TOOL_NAME}/prompt/{prompt_name}")
async def invoke_llm(prompt_name: str, body: Dict[str, Any], 
                     credentials: HTTPAuthorizationCredentials = Depends(security_token)
                     ) -> Dict[str, Any]:    
    """
    Invoke LLM with specified prompt and incoming context. 
    The prompt is looked from and must exist in the MongoDB tool configuration prompts section.
    
    """ 
    (allowed, agent_rec) = mongo_middleware.check_authorization(credentials.credentials)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    context = body.get("context")
    output = {        
        "prompt_name": prompt_name,
        "input_context": context
    }
    try:
        if not context:
            raise ValueError("context must be a non-empty json object in the request body")
        
        # don't load llm client with tools unless there are prompts available.         
        # if the prompt changes (on the mongo side), then reset_settings must be called to reload the tool annotations.
        # this will finish the setup next time an invoke is called.
        global llm_client
        if not llm_client.llm_setup:
            # need the mongo middleware annotations first to get the prompts
            mongo_middleware.load_annotations()
            # this feels circular, but we need to ensure the LLM client is configured with tools, and I didn't want to 
            # reproduce the tool loading code.
            mcp_tools = await mcp.get_tools()
            tools_config = mongo_middleware.get_llm_tools(mcp_tools)
            llm_client.configure_tools(tools_config, tool_handler)
                        
        # Lookup prompt from mongo_server.tool_config["prompts"] if it exists        
        if ("prompts" in mongo_server.tool_config and 
            prompt_name in mongo_server.tool_config["prompts"]):
            #We have a prompt!
            prompt = mongo_server.tool_config["prompts"][prompt_name]
            output["prompt"] = prompt
                        
            resp_obj = await llm_client.invoke_bedrock_with_tools(prompt, json.dumps(context), 15)
            output.update(resp_obj)  # merge the response object into output
            
            # lots of potential errors and exceptions here, so catch them all. 
            # tried to pass most through the return, but some may still raise
            # I could not handle all the exceptions by name either, some would raise a runtime exception 
            # instead of passing the exception directly
            return_json = {} 
            if resp_obj.get("error"):
                return_json = JSONResponse(output, 500)                    
            else:
                logger.info(f"invoke successful for prompt {prompt_name}")
                return_json = JSONResponse(output, 201)
            
            # We want to save the full conversation including LLM output regardless of success or failure
            # Try to handle the exceptions and bubble them up to the output so we don't hit the catches below.
            mongo_middleware.save_llm_conversation(output, agent_rec["agent_key"], TOOL_NAME, prompt_name )
            return return_json

        else:            
            output["error"] = f"Prompt '{prompt_name}' not found in configuration."
            return JSONResponse(output, 404)        
        
    except HTTPException as he:
        logger.error(f"Authorization failed: {he.detail}")
        output["error"] = he.detail
        return JSONResponse(output,he.status_code)
    except Exception as e:
        logger.error(f"invoke_llm failed: {e}")
        traceback.print_exc()        
        output["error"] = f"Error executing invoke_llm: {str(e)}"
        return JSONResponse(output, 500)

@app.post("/vectorize")
async def vectorize_text(body: Dict[str, Any], 
                     credentials: HTTPAuthorizationCredentials = Depends(security_token)
                     )  -> Dict[str, Any]:
    """
    API endpoint to vectorize input text using the LLM embedding model.
    this is not an MCP tool
    """
    try:
        (allowed, agent_rec) = mongo_middleware.check_authorization(credentials.credentials)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Extract textChunk from the request body
        text_chunk = body.get("textChunk")
        
        if not text_chunk or not isinstance(text_chunk, str):
            raise Exception("textChunk must be a non-empty string in the request body")
        
        vector = await llm_client.generate_embedding(text_chunk)
        logger.info(f"Vectorization successful for input text of length {len(text_chunk)}")
        return {
            "input_text": text_chunk,
            "vector": vector
        }
        
    except HTTPException as he:
        logger.error(f"Authorization failed: {he.detail}")
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        traceback.print_exc()
        input = json.dumps(body)
        return {
            "error": f"Error executing vectorize_text: {str(e)}",
            "body" : input
        }

# we now have all the components and routes. mount the MCP server to FastAPI
# Mount the MCP server
app.mount(f"/{TOOL_NAME}", mcp_app)


# These are not really used, left them in just in case.
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
