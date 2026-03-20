import json
from typing import Any, Dict, List, Optional, Annotated
import logging
from pydantic import Field
from pymongo.errors import PyMongoError
from fastmcp import FastMCP
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastmcp.server.dependencies import AccessToken, get_access_token
from starlette.responses import JSONResponse
from AWS_settings import settings 
#from local_settings import settings # change this to use AWS_settings
from mongomcp import MongoDBQueryServer, MongoMCPMiddleware, ServerBedrockClient, MongoTokenVerifier, __version__ as MCP_VERSION
from mongomcp.agent.tool_router import ToolRouter
import traceback
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
main component flow:
1. MongoMCPMiddleware: Connects to MongoDB config database, loads tool configurations, and prep token authorization.
2. MongoDBQueryServer: Implements core MongoDB query functionalities for the specific tool_name from env
3. BedrockClient: Handles AWS Bedrock LLM interactions and tool integrations.
4. instantiate FastMCP server with MongoMCPMiddleware, MongoDBQueryServer, MongoTokenVerifier
5. instantiate FastAPI app, mounts FastMCP app
6. define additional endpoints for health checks, tool configuration retrieval, settings reset, LLM invocation, and text vectorization.

"""

mongo_middleware: MongoMCPMiddleware
mongo_server: MongoDBQueryServer
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
    error = None  # captured on ConnectionError, used in failure log

    # load or reload the mongo middleware and server config
    # we do this to get fresh settings from mongo if reset_settings is called
    try:
        mongo_middleware = MongoMCPMiddleware(settings)
        if mongo_middleware.ANNOTATIONS:
            mongo_server = MongoDBQueryServer(settings)
            mongo_server.set_config(mongo_middleware.ANNOTATIONS)
            auth_provider = MongoTokenVerifier(mongo_middleware)
        else:
            failed = True
    except ConnectionError as e:
        failed = True
        error = e
    if failed:
        logger.error(f"Failed to get configuration from MongoDB. Will wait for 10s before retry.\r\n {error}")
        #time.sleep(10)
        sys.exit(1)

setup_from_mongo()
# Create FastMCP server instance with bearer token authentication
mcp = FastMCP("mongodb-vector-server", auth=auth_provider)
mcp.add_middleware(mongo_middleware)
llm_client = ServerBedrockClient(settings)

@mcp.tool()
async def upsert_document(
    collection: Annotated[str, Field(description="Name of the MongoDB collection to upsert into.")],
    filter: Annotated[Dict, Field(description="Filter to find the document to update.")],
    update: Annotated[Dict, Field(description="Update data for the document.")],
    token: Annotated[AccessToken, Depends(get_access_token)] = None
) -> Dict[str, Any]:
    """Upsert a document in the specified MongoDB collection."""
    
    # if it comes in from the mcp tool directly then we have a token object 
    # otherwise it is a dict from the http endpoint llm_invoke
    # fastapi and fastmcp handle the dependency injection differently
    scopes = set()
    client_id = ""
    if token is None:
        token = get_access_token()
    if isinstance(token, dict):
        scopes = set(token.get("scope", []))
        client_id = token.get("agent_key","")            
    elif token is not None:
        scopes = set(token.scopes)  
        client_id = token.client_id        

    # validate write permissions from the token scopes
    if "write" not in scopes:
        logger.error(f"Insufficient scope for upsert_document: write permission required for agent {client_id}")
        return {"error": "Insufficient scope: this agent does not have write permission."}

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
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
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
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
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
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
        return {"error":f"Error executing text_search: {str(e)}"}

@mcp.tool()
async def geospatial_search(
    collection: Annotated[str, Field(description="Name of the MongoDB collection to search in.")],
    longitude: Annotated[float, Field(description="Longitude for the center point in WGS84.", ge=-180, le=180)],
    latitude: Annotated[float, Field(description="Latitude for the center point in WGS84.", ge=-90, le=90)],
    limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=100)] = 10,
    max_distance_meters: Annotated[Optional[float], Field(default=None, description="Optional maximum distance from the center point in meters.", ge=0)] = None,
    min_distance_meters: Annotated[Optional[float], Field(default=None, description="Optional minimum distance from the center point in meters.", ge=0)] = None,
    filters: Annotated[Optional[List], Field(default=None, description="Optional list of filters in [field, value] format.")] = None,
    geo_field: Annotated[Optional[str], Field(default=None, description="GeoJSON point field path with a 2dsphere index. Defaults to the location_field defined in the tool config.")] = None
) -> Dict[str, Any]:
    """Dynamic docstring loaded from JSON configuration"""
    try:
        results = await mongo_server.geospatial_search(
            collection=collection,
            longitude=longitude,
            latitude=latitude,
            max_distance_meters=max_distance_meters,
            min_distance_meters=min_distance_meters,
            filters=filters,
            limit=limit,
            geo_field=geo_field,
        )
        jobj = json.dumps(results, default=str)
        return {
            "results": jobj,
            "count": len(results),
            "query_info": {
                "longitude": longitude,
                "latitude": latitude,
                "limit": limit,
                "max_distance_meters": max_distance_meters,
                "min_distance_meters": min_distance_meters,
                "geo_field": geo_field,
            }
        }
    except Exception as e:
        logger.error(f"Geospatial search failed: {e}")
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
        return {"error": f"Error executing geospatial_search: {str(e)}"}

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
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
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
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
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
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
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
app = FastAPI(title=settings.TOOL_NAME, lifespan=mcp_app.lifespan)
security_token = HTTPBearer()
optional_token = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials) -> Any:
    (allowed, agent_rec) = mongo_middleware.check_authorization(credentials.credentials)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return agent_rec

async def get_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security_token)]
):
    return verify_token(credentials)

def verify_optional_token(credentials: Optional[HTTPAuthorizationCredentials]) -> Any:
    if not credentials:
        return None
    return verify_token(credentials)

async def get_optional_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(optional_token)]
):
    return verify_optional_token(credentials)

# Dispatch table: tool name → .fn reference. Add new tools here when registered with @mcp.tool().
_TOOL_DISPATCH = {
    "upsert_document":    upsert_document.fn,
    "vector_search":      vector_search.fn,
    "text_search":        text_search.fn,
    "geospatial_search":  geospatial_search.fn,
    "get_unique_values":  get_unique_values.fn,
    "get_collection_info": get_collection_info.fn,
    "aggregate_query":    aggregate_query.fn,
}

async def tool_handler(token: AccessToken, toolname: str, tool_input: dict) -> dict:
    """Map toolname to the appropriate MCP tool function and execute it."""
    fn = _TOOL_DISPATCH.get(toolname)
    if fn is None:
        return {"error": f"Unknown tool: {toolname}"}
    try:
        kwargs = dict(tool_input)
        if toolname == "upsert_document":
            kwargs["token"] = token
        return await fn(**kwargs)
    except Exception as e:
        logger.error(f"Tool handler error for {toolname}: {e}")
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
        return {"error": f"Error executing {toolname}: {str(e)}"}

# Root route
@app.get("/")
async def root_endpoint(token: Annotated[str | None, Depends(get_optional_token)]) -> Dict[str, Any]:
    """Root endpoint"""
    if token:
        return {
            "message": "MongoDB Vector Server MCP",
            "status": "running",
            "version": MCP_VERSION,
            "available_tools": mongo_middleware.active_endpoints,
            "available_endpoints": [
                f"GET  /{settings.TOOL_NAME}/health",                
                f"GET  /{settings.TOOL_NAME}/collection_info",
                f"GET  /{settings.TOOL_NAME}/llm_tools",
                f"POST /{settings.TOOL_NAME}/route",
                f"GET  /{settings.TOOL_NAME}/reset",
                f"POST /{settings.TOOL_NAME}/prompt/{{prompt_name}}",
                "GET  /tools_config",
                "POST /vectorize",
            ]
        }
    else:
        return {
            "message": "OK",
            "status": "running"
        }

# this is for the AWS load balancer health check
@app.get(f"/{settings.TOOL_NAME}/health")
@app.get("/health")
async def http_health_check(token: Annotated[str | None, Depends(get_optional_token)]) -> Dict[str, Any]:
    """Regular HTTP GET endpoint for health checks"""
    # always return something or else the load balancer will mark it unhealthy and continue to reload the container        
    failed, server_info = await mongo_server.get_mongo_info(False)
    output = server_info.copy()
    output["version"] = MCP_VERSION
    if not token: 
        # no token, remove sensitive info
        output.pop("mongodb")
        output.pop("description")
        output["connected"] = server_info["mongodb"].get("connected", False)
        output["timestamp"] = server_info["mongodb"].get("timestamp", "")
        
    status_code = 200
    #if failed:
    #    status_code = 500        
    return output

@app.get("/tools_config")
async def http_get_tools_config(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Regular HTTP GET endpoint for tools config"""
    return {"available_tools": mongo_middleware.active_endpoints, "tool_name": settings.TOOL_NAME}

@app.get(f"/{settings.TOOL_NAME}/collection_info")
async def http_get_collection_info(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Regular HTTP GET endpoint for collection info"""        
    results = await get_collection_info.fn()
    return {"collection_info": results}


@app.get(f"/{settings.TOOL_NAME}/llm_tools")
async def http_get_llm_tools(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Returns preformatted Bedrock toolSpec JSON for all active tools with MongoDB annotations applied."""
    tools = mongo_middleware.build_tools_from_annotations()
    return {"tools": tools, "count": len(tools)}


@app.post(f"/{settings.TOOL_NAME}/route")
async def route_tools(body: Dict[str, Any], token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:
    """Select a subset of tools relevant to a question or explicit tool list.

    Body options (mutually exclusive):
      {"question": "..."}         — LLM routing: asks the model which tools are needed
      {"tools": ["ep.tool", ...]} — Static routing: deterministic filter by name

    The routing prompt is read from mongo config at prompts.tool_router if it exists.
    """
    all_tools = mongo_middleware.build_tools_from_annotations()
    question = body.get("question")
    explicit_tools = body.get("tools")

    if explicit_tools and isinstance(explicit_tools, list):
        # Static routing — no LLM call
        router = ToolRouter(tool_catalog=all_tools)
        filtered = router.select_tools(explicit_tools)
        return {"tools": filtered, "count": len(filtered), "routing": "static"}

    if question:
        # LLM routing
        routing_prompt = (
            mongo_server.tool_config.get("prompts", {}).get("tool_router")
            if hasattr(mongo_server, "tool_config") else None
        )
        router = ToolRouter(tool_catalog=all_tools, llm_client=llm_client)
        filtered = await router.route_for_question(question, routing_prompt)
        return {"tools": filtered, "count": len(filtered), "routing": "llm"}

    return JSONResponse({"error": "Request body must contain 'question' (string) or 'tools' (list)"}, 400)


@app.get(f"/{settings.TOOL_NAME}/reset")
async def reset_settings(token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:    
    """Reload config from MongoDB and reconfigure the LLM client with the latest tool annotations."""
    logger.info(f"Begin settings reset for {settings.TOOL_NAME}")
    output = {"action": "reset settings"}
    try:
        setup_from_mongo()
        new_client = ServerBedrockClient(settings)
        new_client.configure_tools(mongo_middleware.build_tools_from_annotations())
        global llm_client
        llm_client = new_client
        output["result"] = "success"
        logger.info(f"Finished settings reset for {settings.TOOL_NAME}: Success")
    except Exception as e:
        logger.error(f"reset_settings failed: {e}")
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
        output["error"] = f"Error executing reset_settings: {str(e)}"
        output["result"] = "failed"
        logger.info(f"Finished settings reset for {settings.TOOL_NAME}: Failed")
        return JSONResponse(output, 500)

    return output

@app.post(f"/{settings.TOOL_NAME}/prompt/{{prompt_name}}")
async def invoke_llm(prompt_name: str, body: Dict[str, Any], 
                     token: Annotated[str, Depends(get_token)]) -> Dict[str, Any]:    
    """
    Invoke LLM with specified prompt and incoming context. 
    The prompt is looked from and must exist in the MongoDB tool configuration prompts section.
    
    """     
    if not "llm:invoke" in token.get("scope", []):
        logger.error(f"Insufficient scope for invoke_llm: llm:invoke permission required for agent {token["agent_key"]}")
        raise HTTPException(status_code=403, detail="Insufficient scope")

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
            tools_config = mongo_middleware.build_tools_from_annotations()
            llm_client.configure_tools(tools_config)
                        
        # Lookup prompt from mongo_server.tool_config["prompts"] if it exists        
        if ("prompts" in mongo_server.tool_config and 
            prompt_name in mongo_server.tool_config["prompts"]):
            #We have a prompt!
            prompt = mongo_server.tool_config["prompts"][prompt_name]
            output["prompt"] = prompt

            async def scoped_mcp_call(toolname: str, tool_input: dict) -> dict:
                # Keep token handling in this top-level request scope.
                return await tool_handler(token, toolname, tool_input)

            # Bind request-scoped callback on the client instance; BedrockClient no longer
            # accepts a per-call tool callback parameter.
            llm_client.mcp_call = scoped_mcp_call
                        
            resp_obj = await llm_client.invoke_bedrock_with_tools(
                prompt=prompt,
                context=json.dumps(context),
            )
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
            mongo_middleware.save_llm_conversation(output, token["agent_key"], settings.TOOL_NAME, prompt_name)
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
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))        
        output["error"] = f"Error executing invoke_llm: {str(e)}"
        return JSONResponse(output, 500)

@app.post("/vectorize")
async def vectorize_text(body: Dict[str, Any], 
                     token: Annotated[str, Depends(get_token)]
                     )  -> Dict[str, Any]:
    """
    API endpoint to vectorize input text using the LLM embedding model.
    this is not an MCP tool
    """
    try:
        if not "llm:invoke" in token.get("scope", []):
            raise HTTPException(status_code=403, detail="Insufficient scope")

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
        logger.debug("".join(traceback.format_exception(None, e, e.__traceback__)))
        input = json.dumps(body)
        return {
            "error": f"Error executing vectorize_text: {str(e)}",
            "body" : input
        }

# we now have all the components and routes. mount the MCP server to FastAPI
# Mount the MCP server
app.mount(f"/{settings.TOOL_NAME}", mcp_app)


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
