from fastmcp.server.middleware.middleware import Middleware, MiddlewareContext, CallNext
from typing import List, Dict, Any
import mcp.types as mt    
import jwt
import jwt.exceptions
import datetime
import logging
import os
import json
import traceback
from .mongodb_client import MongoDBClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SHOW_ONCE = 0
class MongoMCPMiddleware(Middleware):
    """
    FastMCP Middleware is the central point for connecting to MongoDB config database.
    handles intercept and print on_list_tools output
    middleware is always connected to the mongo MCP config collection.
    use this to make core config requests and send logging info to the central MCP config collections
    """
    def __init__(self, settings):
        super().__init__()
        self.endpoint_name = settings.TOOL_NAME
        self._is_local = settings.IS_LOCAL
        logger.info("MongoMCPMiddleware initialized")
        self.mongo_client = MongoDBClient(settings)
        self.ANNOTATIONS = None
        self.active_endpoints = [self.endpoint_name]
        self.endpoint_tools = {}
        self.load_annotations()
        
    def load_annotations(self):
        """Load tool annotations from the JSON out of mongo"""        
        global SHOW_ONCE
        try:
            if self.mongo_client.sync_connect_to_mongodb():
                if SHOW_ONCE < 1:
                    logger.info(f"loading dynamic config for endpoint {self.endpoint_name}")
                # load the config for this specific tool, then we load it for everything so we can return all tools on the shared endpoint 
                # make 2 calls because we need this config regardless of active state 
                doc = self.mongo_client.get_collection().find_one({"Name": self.endpoint_name})    
                self.ANNOTATIONS = doc
                self.endpoint_tools = self.ANNOTATIONS.get('tools', {})
                #### load all active endpoints to return configs
                if self._is_local:
                    if SHOW_ONCE < 1:                    
                        logger.info(f"Running in local mode, loading only the current endpoint config for {self.endpoint_name}")
                        SHOW_ONCE += 1
                else:
                    self.active_endpoints = list(self.mongo_client.get_collection().distinct("Name",{ "active": True}))
                    if SHOW_ONCE < 1:                    
                        logger.info(f"Running in dynamic mode, loading all available endpoint configs for endpoints: {self.active_endpoints}")
                        SHOW_ONCE += 1

                return doc
        except ConnectionError as ce:
            logger.error(f"MongoDB connection error while loading annotations for endpoint {self.endpoint_name}. check IP whitelist, networking etc.:\r\n {ce}")
            return None
        except Exception as e:
            logger.error(f"Failed to load annotations for endpoint {self.endpoint_name}:\r\n {e}")
            return None

    def refresh_active_endpoints(self) -> list:
        """Re-query active endpoints from MongoDB so newly activated entries are picked up."""
        if self._is_local:
            return self.active_endpoints
        try:
            if self.mongo_client.sync_connect_to_mongodb():
                self.active_endpoints = list(
                    self.mongo_client.get_collection().distinct("Name", {"active": True})
                )
        except Exception as e:
            logger.error(f"Failed to refresh active endpoints: {e}")
        return self.active_endpoints

    def check_authorization(self, token: str):
        """Check if the provided token is valid"""
        allowed = False
        agent_rec = None        
        try:
            header = jwt.get_unverified_header(token)
            api_key = header.get("api_key")
            
            self.mongo_client.sync_connect_to_mongodb()
            agent_coll = self.mongo_client.get_collection("agent_identities")
            agent_rec = agent_coll.find_one({"agent_key": api_key})
            if agent_rec:
                # you should hash.... do as I say not as I do.
                # store the hash private key in secrets manager, then implement hash. 
                # I think most will come in through a token service which makes this moot.
                # trying to keep the demo simple, so just verifying the token directly here.
                # in order to hash I would need a token generator service and I don't want to build that here.
                # https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/#oauth2passwordrequestform
                pvk = agent_rec.get("pvk")
                decoded_payload = jwt.decode(token,pvk, algorithms=["HS256"])
                agent_name = decoded_payload.get("agent_name")
                if decoded_payload.get("revoked", False):
                    logger.warning(f"Token for agent {agent_name}:{api_key} is revoked.")
                    return (False, None)
                agent_rec.pop("pvk")  # remove sensitive info
                
                if agent_name == agent_rec.get("agent_name"):
                    logger.info(f"Authorization successful for agent: {agent_name}")
                    allowed = True
        
        except jwt.exceptions.InvalidTokenError  as je:
            logger.error(f"JWT decoding error: {je}")
            allowed = False
        except Exception as e:
            logger.error(f"Unexpected error during authorization check: {e}")
            allowed = False

        return (allowed, agent_rec)

    _PYTHON_TO_JSON_SCHEMA_TYPE = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "List": "array",
        "dict": "object",
        "Dict": "object",
    }

    def _python_type_to_json_schema(self, type_str: str) -> str:
        """Map a Python type annotation string (from MongoDB annotations) to a JSON Schema type."""
        if not type_str:
            return "string"
        t = type_str.strip()
        if t.startswith("Optional["):
            t = t[9:-1].strip()
        if t.startswith("List"):
            return "array"
        if t.startswith("Dict"):
            return "object"
        return self._PYTHON_TO_JSON_SCHEMA_TYPE.get(t, "string")

    def build_tools_from_annotations(self) -> List[Dict]:
        """Build Bedrock toolSpec JSON entirely from MongoDB annotations.

        Requires no FastMCP introspection — all tool names, descriptions, parameter
        types, defaults, and required lists come directly from the MongoDB config
        collection. Call this instead of get_llm_tools()/get_formatted_llm_tools()
        wherever you previously needed to first call mcp.get_tools().

        Returns:
            List of dicts in Bedrock toolSpec format, ready for LLM consumption.
        """
        self.load_annotations()
        tools_dict = []
        try:
            for tool_name, anot in self.endpoint_tools.items():
                description = anot.get("description", f"Tool: {tool_name}")
                returns = anot.get("returns")
                if returns:
                    description += f"\n\nReturns:\n\t{returns}"

                properties = {}
                for p_name, p_info in anot.get("parameters", {}).items():
                    json_type = self._python_type_to_json_schema(p_info.get("type", "str"))
                    prop = {
                        "type": json_type,
                        "description": p_info.get("description", ""),
                    }
                    if "default" in p_info and p_info["default"] is not None:
                        prop["default"] = p_info["default"]
                    properties[p_name] = prop

                tools_dict.append({
                    "toolSpec": {
                        "name": tool_name,
                        "description": description,
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": properties,
                                "required": anot.get("required", []),
                            }
                        }
                    }
                })
        except Exception as e:
            logger.error(f"Error building tools from annotations: {e}")
        return tools_dict

    def save_llm_conversation(self, conversation_data: Dict[str, Any], agent_id: str, tool_name: str, prompt_name: str) -> bool:
        """Save LLM conversation data to MongoDB"""
        try:
            if self.mongo_client.sync_connect_to_mongodb():
                collection = self.mongo_client.get_collection("llm_history")
                data = {
                    "agent_id": agent_id,                    
                    "tool_name": tool_name,
                    "prompt_name": prompt_name,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                data.update(conversation_data)

                result = collection.insert_one(data)
                logger.info(f"LLM conversation saved with id: {result.inserted_id}")
                return True
            else:
                logger.error("MongoDB connection not established. Cannot save LLM conversation.")
                return False
        except Exception as e:
            logger.error(f"Failed to save LLM conversation: {e}")
            return False

    async def on_list_prompts(self, context, call_next):        
        return await super().on_list_prompts(context, call_next)

    async def on_list_tools(
        self, 
        context: MiddlewareContext[mt.ListToolsRequest], 
        call_next: CallNext[mt.ListToolsRequest, List[mt.Tool]]
    ) -> List[mt.Tool]:
        """Intercept list_tools and apply MongoDB annotation config: descriptions, parameter info, and tool filtering."""
        try:
            result = await call_next(context)

            if not result:
                logger.info("on_list_tools: no tools returned from handler")
                return result

            self.load_annotations()
            remove_tools = []
            for tool in result:
                anot = self.endpoint_tools.get(tool.name)
                if not anot:
                    logger.info(f"No annotation found for tool '{tool.name}', removing from list")
                    remove_tools.append(tool)
                    continue

                description = anot.get("description", f"Tool: {tool.name}")
                returns = anot.get("returns")
                if returns:
                    description += f"\n\nReturns:\n    {returns}"
                tool.description = description

                if tool.parameters and "properties" in tool.parameters:
                    new_props = {}
                    for prop_name, prop_val in tool.parameters["properties"].items():
                        if prop_name == "token":
                            continue
                        new_props[prop_name] = prop_val
                        param_info = anot.get("parameters", {}).get(prop_name, {})
                        if param_info.get("description"):
                            new_props[prop_name]["description"] = param_info["description"]
                    tool.parameters["properties"] = new_props

            for rt in remove_tools:
                result.remove(rt)

            return result

        except Exception as e:
            logger.error(f"ERROR in on_list_tools: {e}")
            traceback.print_exc()
            raise
