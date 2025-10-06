from fastmcp.server.middleware.middleware import Middleware, MiddlewareContext, CallNext
from typing import List, Dict
import mcp.types as mt    
import logging
import traceback
from mongodb_client import MongoDBClient
from settings_aws import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ListToolsLoggingMiddleware(Middleware):
    """FastMCP Middleware to intercept and print on_list_tools output"""
    def __init__(self, tool_name: str):
        super().__init__()
        self.tool_name = tool_name
        logger.info("ListToolsLoggingMiddleware initialized")
        self.mongo_client = MongoDBClient()
        self.mongo_client._db_name = settings.mcp_config_db
        self.mongo_client._collection_name = settings.mcp_config_col
        self.ANNOTATIONS = None
        self.ALLTOOLS = []
        self.load_annotations()  
        
    def load_annotations(self):
        """Load tool annotations from the JSON configuration file"""        
        try:
            self.mongo_client.sync_connect_to_mongodb()
            doc = self.mongo_client.collection.find_one({"Name": self.tool_name})
            print(f"Loaded dynamic config for tool {self.tool_name}")       
            self.ANNOTATIONS = doc

            # load all tools to return configs
            self.ALLTOOLS = list(self.mongo_client.collection.distinct("Name",{ "active": True}))

            return doc
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            traceback.print_exc()
            return None

    # Get tool annotation by name
    def get_tool_annotation(self, tool_name: str) -> Dict:
        """Get annotation data for a specific tool"""
        # load it fresh every time? 
        self.load_annotations()  
        tools = self.ANNOTATIONS.get('tools', [])
        if tool_name in tools:
            tool = tools[tool_name]
            return tool
        return {}

    # Generate docstring from JSON annotation
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

    async def on_list_tools(
        self, 
        context: MiddlewareContext[mt.ListToolsRequest], 
        call_next: CallNext[mt.ListToolsRequest, List[mt.Tool]]
    ) -> List[mt.Tool]:
        """Intercept the list_tools call and alter output to match JSON config"""        
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
                                    new_props[prop] = param[prop]
                                    param_info = anot["parameters"].get(prop, {})
                                    new_props[prop]["description"] =  param_info["description"]
                                    #new_props[prop]["type"] =  param_info["type"]
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
