from fastmcp.tools import Tool
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
from .MongoDBClient import MongoDBClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# flag to load only 1 tool when local
IS_LOCAL = json.loads(os.getenv('IS_LOCAL', 'false').lower())

class MongoMCPMiddleware(Middleware):
    """
    FastMCP Middleware is the central point for connecting to MongoDB config database.
    handles intercept and print on_list_tools output
    middleware is always connected to the mongo MCP config collection.
    use this to make core config requests and send logging info to the central MCP config collections
    """
    def __init__(self, tool_name: str, settings):
        super().__init__()
        self.tool_name = tool_name
        logger.info("MongoMCPMiddleware initialized")
        self.mongo_client = MongoDBClient(settings)
        self.ANNOTATIONS = None
        self.ALLTOOLS = ["claimsSearch"] #AirbnbSearch
        self.ActiveTools = []
        self.load_annotations()  
        
    def load_annotations(self):
        """Load tool annotations from the JSON out of mongo"""        
        try:
            if self.mongo_client.sync_connect_to_mongodb():
                #print(f"loading dynamic config for tool {self.tool_name}")      
                # load the config for this specific tool, then we load it for everything so we can return all tools on the shared endpoint 
                # make 2 calls because we need this config regardless of active state 
                doc = self.mongo_client.get_collection().find_one({"Name": self.tool_name})    
                self.ANNOTATIONS = doc
                self.ActiveTools = self.ANNOTATIONS.get('tools', [])
                #### load all active tools to return configs
                if IS_LOCAL:
                    logger.info(f"Running in local mode, loading only the current tool config for {self.tool_name}")
                else:
                    self.ALLTOOLS = list(self.mongo_client.get_collection().distinct("Name",{ "active": True}))                

                return doc
        except ConnectionError as ce:
            logger.error(f"MongoDB connection error while loading annotations for tool {self.tool_name}. check IP whitelist, networking etc.:\r\n {ce}")
            return None
        except Exception as e:
            logger.error(f"Failed to load annotations for tool {self.tool_name}:\r\n {e}")
            return None

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
                agent_rec.pop("pvk")  # remove sensitive info
                agent_name = decoded_payload.get("agent_name")
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

    def get_llm_tools(self, mcp_tools: Dict[str, Tool]) -> Any:
        """local function which outputs JSON from get_tools for internal LLM tool configuration"""    
        try:            
            tools_dict = []
            for tool_name, tool in mcp_tools.items():                
                # mcp_tools is important because FastMCP has a lot of helper functions that automate the tools response
                # I just want the output.
                if not tool_name in self.ActiveTools:
                    # the mcp_tools contains all tools, we only want the active ones from our annotations
                    continue                              
                
                # Still need to rebuild to match the expected output format
                # output from this is the expected intput for the LLM tool config
                props = {}
                required = []
                ret_type = "object"
                for prop_name, prop in tool.parameters.items():
                    if prop_name == "properties":
                        props = prop
                    elif prop_name == "required":
                        required = prop
                    elif prop_name == "type":  
                        ret_type = prop

                tool_obj = {
                    "name": tool_name,
                    "description": tool.description,
                    "inputSchema": {
                    "json": {"properties": props,
                                "required": required,
                                "type": ret_type                           
                            }
                    }
                }
                
                tooslspec = {"toolSpec": tool_obj}
                tools_dict.append(tooslspec)        
            return tools_dict
        except Exception as e:
            logger.error(f"Error outputting tools JSON: {e}")
            return {"error": f"Failed to serialize tools: {str(e)}"}

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

    # Get tool annotation by name
    def get_tool_annotation(self, tool_name: str) -> Dict:
        """Get annotation data for a specific tool"""
        # load it fresh every time? 
        self.load_annotations()  
        if tool_name in self.ActiveTools:
            tool = self.ActiveTools[tool_name]
            return tool
        return {}

    def generate_docstring(self, tool_name: str) -> str:
        """Generate docstring for a tool from JSON annotation"""
        tool_info = self.get_tool_annotation(tool_name)
        if not tool_info:
            return None
        
        docstring = tool_info.get("description", f"Tool: {tool_name}")
        
        # Add returns information if available
        returns = tool_info.get("returns")
        if returns:
            docstring += f"\n\nReturns:\n    {returns}"
        
        return docstring

    async def on_list_prompts(self, context, call_next):        
        return await super().on_list_prompts(context, call_next)

    async def on_list_tools(
        self, 
        context: MiddlewareContext[mt.ListToolsRequest], 
        call_next: CallNext[mt.ListToolsRequest, List[mt.Tool]]
    ) -> List[mt.Tool]:
        """Intercept the list_tools call and alter output to match JSON config from mongo mcp_config collection"""        
        try:
            # Call the next middleware or the actual handler
            result = await call_next(context)
            
            if result:
                remove_tools = []
                for tool in result:
                    tool_description =  self.generate_docstring(tool.name)
                    if tool_description:
                        tool.description = tool_description
                    else:
                        #print(f"No annotation found for tool '{tool.name}'")
                        remove_tools.append(tool)
                        continue

                    
                    anot = self.get_tool_annotation(tool.name)
                    req = anot.get("required", [])
                    
                    if tool.parameters:
                        keys = list(tool.parameters.keys())                         
                        for param_name in keys:                            
                            param = tool.parameters[param_name]                            
                            if param_name == "required":                                
                                if param_name in req:                                    
                                    tool.parameters[param_name]["required"] = True                                
                            elif param_name == "properties":
                                new_props = {}
                                for prop in param:
                                    if prop == "token":
                                        # token is for internal passing only, ignore it
                                        continue
                                    try:
                                        new_props[prop] = param[prop]
                                        param_info = anot["parameters"].get(prop, {})
                                        new_props[prop]["description"] =  param_info["description"]
                                        #new_props[prop]["type"] =  param_info["type"]
                                    except KeyError as ke:
                                        logger.error(f"No parameter info found for {prop} in tool {tool.name}: {ke}")
                                tool.parameters["properties"] = new_props                                                
            else:
                print("   No tools found")
                        
            if len(remove_tools) > 0:
                for rt in remove_tools:
                    result.remove(rt)

            return result
            
        except Exception as e:
            print(f"ERROR in middleware: {e}")
            print("Full stack trace:")
            traceback.print_exc()  # Prints full stack trace
            print("=" * 60 + "\n")
            raise
