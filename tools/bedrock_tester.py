# openAi test
import sys
import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from pymongo import MongoClient


parent_dir = str(Path(__file__).resolve().parents[1])
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from mongomcp.bedrock_client import ServerBedrockClient
#from mongomcp.libertyair_embedding import LibertyAIREmbeddingClient
#from lmig_settings import settings
from ai_test_settings import settings

embedding_client = None
llm_client = None
#embedding_client = LibertyAIREmbeddingClient.from_settings(settings)
# this is the mongo tools from config load.
llm_client = ServerBedrockClient(settings)
db = None

def process_args(arglist):
    args = {}
    for arg in arglist:
        pair = arg.split("=")
        if len(pair) == 2:
            args[pair[0].strip()] = pair[1].strip()
        else:
            args[arg] = ""
    return args

def db_client():
    global db
    creds = settings.get_mongo_credentials()
    uri = f"mongodb+srv://{creds['username']}:{creds['password']}@{creds['mongoUrl']}"
    db = MongoClient(uri)["ai_config"]
    print(f"Database: {uri}")
    return db

async def generate_embeddings(text: str, model_id: Optional[str] = None, is_query: bool = True) -> list:
    # openai model should be initialized already if embedding model id is openai
    #model_id = model_id.replace("openai-","")
    print(f"GenerateEmbedding Using OpenAI, embedding model: {model_id}")
    result = await llm_client.generate_embedding(text, model_id)
    return result

def generate_openai_embeddings2(text: str, model_id: Optional[str] = None, is_query: bool = True) -> list:
    # openai model should be initialized already if embedding model id is openai
    #model_id = model_id.replace("openai-","")
    print(f"GenerateEmbedding Using OpenAI, embedding model: {model_id}")
    result = embedding_client.get_embedding(text, model_id)
    return {
                "embedding_model": model_id,
                "vector": result[0]
            }

async def abstract_embeddings(text: str, model_id: Optional[str] = None, is_query: bool = True) -> list:
        # openai model should be initialized already if embedding model id is openai
        #model_id = model_id.replace("openai-","")
        #logger.info(f"GenerateEmbedding Using OpenAI, embedding model: {model_id}")
        result = await asyncio.to_thread(generate_embeddings, text, model_id)
        return result
    
async def reprocess_embeddings():
    #Reprocess embeddings with new model
    verbose = False
    if "verbose" in ARGS:
        verbose = True
    colls = ["memory_episodic", "memory_semantic"]
    icnt = 0
    for coll in colls:
        results = db[coll].find({})
        print(f"# -------- Updating vectors in {coll} with {settings.EMBEDDING_MODEL_ID} ----- #")
        for doc in results:
            cont = doc["content"]
            result = await generate_embeddings(cont, settings.EMBEDDING_MODEL_ID, verbose)
            if len(result["vector"]) > 500:
                ans = db[coll].update_one({"_id": doc["_id"]}, {"$set": {"embedding": result["vector"]}})
            else:
                print("# ERROR: no vector")
            print(f"Count: {icnt}")
            icnt += 1

def get_tools():
    result = db["mcp_tools"].find_one({"Name": "gemSearch"})
    return(result["tools"])

def test():
    messages = ARGS["messages"]

async def test_embed():
    text = "Brady has a cool spotted dog name Comet"
    result = await generate_embeddings(text, settings.EMBEDDING_MODEL_ID, False)
    #result = await abstract_embeddings(text, settings.EMBEDDING_MODEL_ID, False)
    
    print(result["embedding_model"])
    print(result["vector"][:5])


if __name__ == "__main__":
    db_client()
    #tools_config = get_tools()

    ARGS = process_args(sys.argv)
    if "process" in ARGS:
        asyncio.run(reprocess_embeddings())
    else:
        asyncio.run(test_embed())
    
