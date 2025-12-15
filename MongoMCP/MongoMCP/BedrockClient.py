"""
MongoDB AI/LLM client functions 

"""

import datetime
import json
import re
import asyncio
import traceback
from typing import Callable
import logging
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()  
        return json.JSONEncoder.default(self, obj)


class BedrockClient():
    def __init__(self, settings):
        self.settings = settings
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.settings.aws_region)
        self.mcp_tools = None
        self.mcp_call = None
        self.llm_setup = False
        
        
    def configure_tools(self, tools_config, tool_handler: Callable):
        """Configure MCP tools for Bedrock client"""        
        self.mcp_tools = tools_config
        self.mcp_call = tool_handler
        self.llm_setup = True

    def _try_parse_json(self, json_string):
        """ try to parse a string to json, return json obj or nothing """
        # I hate using try/catch as logic, but here we are. is there a better way?
        try:
            # Remove markdown code fences
            json_string = re.sub(r'^```json\s*', '', json_string)
            json_string = re.sub(r'^```\s*', '', json_string)
            json_string = re.sub(r'\s*```$', '', json_string)

            # Find the JSON object (between first { and last })
            start_idx = json_string.find('{')
            end_idx = json_string.rfind('}')

            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                logger.info("No valid JSON object found in response")
            else:
                json_string = json_string[start_idx:end_idx+1]
                json_string = json_string.strip()
                data = json.loads(json_string)
                return data
        except json.JSONDecodeError as e:
            return None
        except ValueError as e:
            return None
        return None

    async def invoke_bedrock_with_tools(self, prompt: str, context: str=None, max_iterations=10 ) -> list:
        """
        Invoke Bedrock with MCP tools support and caching enabled
        
        Args:
            prompt: User prompt to send to the LLM
            max_iterations: Maximum number of tool call iterations
            
        Returns:
            str: Final assistant response
        """        
        # Prepare the conversation messages
        if context:
            prompt = prompt + f"\nUse the following data for Context: {context}"
        messages = [{
            "role": "user",
            "content": [{                
                "text": prompt
            }]
        }]
        
        # Tool configuration for Bedrock
        tool_config = {
            "tools": self.mcp_tools
        }

        usage = None
        return_obj = {
            "history": messages,
            "usage": usage
        }

        # subtract 1 or else we would end on a tool response        
        for iteration in range(max_iterations):
            try:
                # Invoke Bedrock using the Converse API
                response = self.bedrock_client.converse(
                    modelId=self.settings.LLM_MODEL_ID,
                    messages=messages,
                    toolConfig=tool_config
                )

                # Aggregate usage statistics
                itt_used = response['usage']
                if usage is None: 
                    usage = itt_used
                else:
                    for k,v in itt_used.items(): usage[k] += v
                return_obj["usage"] = usage

                # Get the assistant's response
                assistant_message = response['output']['message']                
                messages.append(assistant_message)
                return_obj["history"] = messages

                # if this is the final itteration, return what we have, but don't do the tool call.
                # just think it makes sense to end after the last LLM response
                if iteration + 1 >= max_iterations:
                    break
                
                # Check if the assistant wants to use tools
                if 'content' in assistant_message:
                    tool_calls = []

                    for content in assistant_message['content']:
                        if 'toolUse' in content:
                            tool_calls.append(content['toolUse'])
                        # don't care about the text content right now its already recorded
                        #elif 'text' in content:
                        #    text_content.append(content['text'])
                    
                    # If there are tool calls, execute them
                    if tool_calls:
                        tool_results = []
                        
                        for tool_call in tool_calls:
                            tool_name = tool_call['name']
                            tool_input = tool_call['input']
                            tool_use_id = tool_call['toolUseId']
                            #print(f"Executing tool:{tool_use_id}-{tool_name} with input: {tool_input}")
                            # Execute the MCP tool call (with caching)
                            try:
                                tool_result = await self._call_mcp_tool(tool_name, tool_input)
                                tool_results.append({
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": str(tool_result)}]
                                    }
                                })
                            except Exception as e:
                                # don't fail here, the LLM can usually find a work around
                                # just log it and keep going
                                logger.error(f"Error executing MCP tool {tool_name}: {e}")
                                tool_results.append({
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": f"Error: {str(e)}"}],
                                        "status": "error"
                                    }
                                })
                                
                        
                        # Add tool results to the conversation
                        if tool_results:
                            tool_message = {"role": "user", "content": tool_results}
                            messages.append(tool_message)              
                            return_obj["history"] = messages  
                            continue  # Continue the conversation loop
                    
                    # If no more tool calls, then we're done and return the response
                    return_obj["stats"] = {"total_itterations": iteration + 1, "max_itterations": max_iterations}     
                    # put the last message content as the main response
                    if len(messages) > 0 and messages[-1]["role"] == "assistant":
                        msg = messages[-1]["content"][0]["text"]
                        jobj = self._try_parse_json(msg)
                        if jobj:
                            return_obj["response"] = jobj
                        else:
                            return_obj["response"] = msg
                    return return_obj
                
                # If we get here, there was no content to process
                return_obj["error"] = "No response generated"
                return return_obj
                
            except ClientError as error:
                error_code = error.response['Error']['Code']
                logger.error(f"Bedrock error: {error_code} - {error.response['Error']['Message']}")
                if error_code == 'ValidationException':
                    return_obj["error"] = f"Input validation failed {error.response['Error']['Message']}"
                elif error_code in ['ExpiredTokenException', 'ExpiredToken']:
                    raise Exception("credentials have expired", error)
                else:
                    return_obj["error"] = error.response['Error']['Message']
                return return_obj
            except Exception as e:
                logger.error(f"Unexpected error in invoke_bedrock_with_tools: {e}")
                return_obj["error"] = str(e)
                return return_obj

        # If max iterations reached without completion
        logger.error(f"invoke_bedrock_with_tools reached maximum iterations: {max_iterations}")
        return_obj["error"] = f"Maximum iterations ({max_iterations}) reached without completion"
        return return_obj
        

    async def _call_mcp_tool(self, toolname: str, tool_input: dict) -> str:
        """Initialize a stateless session for tool calls."""
        try:
            result = await self.mcp_call(toolname, tool_input)
            # Handle the dictionary result from tool_handler function
            if isinstance(result, dict):
                return json.dumps(result, cls=DateTimeEncoder, indent=2)
            else:
                return str(result)
        except Exception as e:
            print(f"Failed MCP {toolname} call: {e}")
            traceback.print_exc()
            raise

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
                modelId= self.settings.EMBEDDING_MODEL_ID, #"amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="application/json",
                body=body
            )
        )
        # Parse the response and extract the embedding vector
        return json.loads(response["body"].read())["embedding"]
