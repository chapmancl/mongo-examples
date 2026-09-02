# Process Data through VoyageAI embeddings
import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from bson import ObjectId
from bson import json_util
import datetime
import random
import time
from pymongo import MongoClient
import multiprocessing
import voyageai
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError
import pprint
#import logging
parent_dir = str(Path(__file__).resolve().parents[1])
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ai_test_settings import settings

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

_context_model_id = settings.CONTEXT_EMBEDDING_MODEL_ID
_model_id = settings.EMBEDDING_MODEL_ID
_api_key = settings.mongo_voyage_apikey()
voyage_client = voyageai.Client()
#voyage_client = voyageai.Client(api_key=_api_key)

#_database = "gem-exposure-repository-mongo-db"
#_collection = "Exposure"
_database = "gem-exposure-repository-mongo-db"
_collection = "Exposure"
_output_database = "ai_config"
_output_collection = "exposure_search"

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
    creds = settings.get_mongo_credentials()
    uri = f"mongodb+srv://{creds['username']}:{creds['password']}@{creds['mongoUrl']}"
    #db = MongoClient(uri)["ai_config"]
    client = MongoClient(uri)
    db = client[_database]
    #print(f"Database: {uri}")
    return client, db

def build_narratives(docs):
    # Initialize MarkItDown and output folder 
    cur_process = multiprocessing.current_process()
    pid = cur_process.pid
    print(f"[{pid}] Starting batch conversion...")
    converted_count = 0

    # 1. Iterate through documents
    converted_docs = []
    for doc in docs:
        # Safely get a unique ID for the filename
        doc_id = str(doc.get('_id'))
        # Convert MongoDB objects (like ObjectIds/dates) to strings
        doc_json = json.loads(json_util.dumps(doc, default=str, indent=4))       
        try:
            md_doc = dict_to_md(doc_json)
            converted_docs.append(md_doc)
            converted_count += 1
            #print(f"Converted {converted_count} documents so far...")          
        except Exception as e:
            print(f"Failed to convert document {doc_id}: {e}")            
    print(f"[{pid}] Successfully converted {converted_count} documents into markdown")
    #print(f"First doc: {converted_docs[0]}")
    return converted_docs

def dict_to_md(data, depth=1):
    """Convert dictionary to markdown format with support for nested structures.

    Handles:
    - Nested dictionaries (recursive)
    - Lists/arrays (with bullet points)
    - Nested lists (recursive with indentation)
    - Primitive values (strings, numbers, etc.)

    Args:
        data: Dictionary to convert
        depth: Current heading depth level (for nested dicts)

    Returns:
        String containing markdown-formatted text
    """
    output = []

    def format_list(items, indent_level=0):
        """Recursively format list items with proper indentation."""
        list_output = []
        indent = "  " * indent_level  # 2 spaces per indent level

        for item in items:
            if isinstance(item, dict):
                # Nested dict in list - format as sub-section
                list_output.append(f"{indent}- Object:\n")
                for k, v in item.items():
                    if isinstance(v, dict):
                        list_output.append(f"{indent}  - **{k}**:\n")
                        list_output.append(dict_to_md(v, depth + 1))
                    elif isinstance(v, list):
                        list_output.append(f"{indent}  - **{k}**:\n")
                        list_output.append(format_list(v, indent_level + 2))
                    else:
                        list_output.append(f"{indent}  - **{k}**: {v}\n")
            elif isinstance(item, list):
                # Nested list - recurse with increased indentation
                list_output.append(f"{indent}- Nested list:\n")
                list_output.append(format_list(item, indent_level + 1))
            else:
                # Primitive value
                list_output.append(f"{indent}- {item}\n")

        return "".join(list_output)

    # Main loop through dictionary items
    for key, value in data.items():
        if isinstance(value, dict):
            # Nested dictionary - create heading and recurse
            output.append(f"{'#' * depth} {key}\n")
            output.append(dict_to_md(value, depth + 1))
        elif isinstance(value, list):
            # List/array - format with bullet points
            if len(value) == 0:
                output.append(f"**{key}**: (empty list)  \n")
            else:
                output.append(f"**{key}**:\n")
                output.append(format_list(value))
        else:
            # Primitive value
            output.append(f"**{key}**: {value}  \n")

    return "".join(output)

def create_embeddings(batch_docs, batch_ids):
    cur_process = multiprocessing.current_process()
    pid = cur_process.pid
    print(f"[{pid}] docs for embedding: {len(batch_docs)}")
    result = voyage_client.contextualized_embed(
        inputs=batch_docs, 		# one document; the model splits it into chunks
        model=_context_model_id,
        input_type="document",
        enable_auto_chunking=True,  # let voyage-context-4 chunk the document
        chunk_size=512, 	          # target chunk size in tokens (optional)
        chunk_overlap=32,       # overlap between adjacent chunks (optional)
        output_dimension=1024,  # Matryoshka dim: 2048 | 1024 | 512 | 256
    )
    # Python SDK exposes returned chunk text here
    bulk_docs = []
    # For embeddings collection format
    # {doc_id: "", chunk_id: 12, chunk_text: "blah blah...", embedding: [], updated_at: 12/23/2026"}
    for doc_idx, doc_result in enumerate(result.results):
        doc_meta = batch_ids[doc_idx]
        print(f"# [{pid}] ---- Document {doc_idx} - [{doc_meta['doc_id']}] ---- #")
        print(f"[{pid}] Chunk index: ", end="")
        for chunk_idx, embedding in enumerate(doc_result.embeddings):
            res_doc = {
                "doc_id": doc_meta["doc_id"],
                "id": doc_meta["id"],
                "type_code": doc_meta["type_code"],
                "category_code": doc_meta["category_code"],
                "chunk_id": chunk_idx,
                "chunk_text": result.chunk_texts[doc_idx][chunk_idx],
                "embedding": embedding,
                "updated_at": datetime.datetime.now()
            }
            print(f"{chunk_idx}", end=".")
            #print("Chunk text:")
            #print(result.chunk_texts[doc_idx][chunk_idx])
            #print(f"Embedding dimension: {len(embedding)}")
            #print("First 8 embedding values:")
            #print(embedding[:8])
            bulk_docs.append(res_doc)
        print(" ")
    #pprint.pprint.info(bulk_docs[0:2])
    return bulk_docs

def process_documents(ipos,passed_args):
    #  Working in parallel thread
    cur_process = multiprocessing.current_process()
    pid = cur_process.pid
    batches = passed_args["batches"]
    bulk_size = 5
    sep(pid)
    print(f"[{pid}] Loading Collection Batch Data [{ipos}]")
    print(f'[{pid}] Current process is {cur_process.name}')
    client, db = db_client()
    doc_ids = []
    tot_docs = 0
    batch_num = 0
    # Perform query for each batch (40M/1000 batches = 40K docs per batch)
    for batch in batches:
        docs = list(db[_collection].find({
            "_id": {
                "$gte": batch["start_id"],
                "$lt": batch["end_id"]
            }
        }))
        num_docs = len(docs)
        tot_docs += num_docs
        iterations = int(num_docs / bulk_size)
        iter_cnt = 0
        doc_ids = []
        # Loop through docs in batch, but get embeddings for each x
        for doc in docs:
            doc_ids.append({"doc_id": str(doc["_id"]), "id": doc["id"], "type_code": doc["type"]["code"], "category_code": doc["category"]["code"]})

        for iter in range(iterations):
            start_idx = iter * bulk_size
            end_idx = start_idx + bulk_size
            str_docs = build_narratives(docs[start_idx:end_idx])
            bulk_docs =create_embeddings(str_docs, doc_ids[start_idx:end_idx])
            if len(bulk_docs) > 0:
                client[_output_database][_output_collection].insert_many(bulk_docs)
                # Token usage is documented in the REST response as usage.total_tokens.
                # If your SDK version exposes it, this may work:
                print(f"# [{pid}] ----- {len(bulk_docs)} embeddings created. {len(bulk_docs)} documents processed ----- #")
                #print("\nPossible total_tokens attribute:", getattr(result, "total_tokens", None))
            
        batch_num += 1
        print(f"[{pid}] Batch {batch['batch_num']}, {num_docs} ({iterations}-iters) complete")
    sep(pid)
    print(f"[{pid}] Process complete - {tot_docs} documents processed")
    sep(pid)

    
def batch_data(num_chunks: int = 1000, partial_mode = 0) -> List[Dict[str, Any]]:
    """Divide collection data into batches using _id timestamp.
    
    MongoDB ObjectIds contain a timestamp in the first 4 bytes. This method:
    1. Finds the min and max ObjectId in the collection
    2. Calculates the time range between them
    3. Divides the time range into num_chunks equal intervals
    4. Returns start/end ObjectId pairs for each chunk
    
    Args:
        num_chunks: Number of batches to divide data into (default: 1000)
        
    Returns:
        List of dicts with 'batch_num', 'start_id', 'end_id', 'start_time', 'end_time'
        
    Example usage:
        batches = self.batch_data(num_chunks=1000)
        for batch in batches:
            # Query documents in this batch
            docs = self.collection.find({
                "_id": {
                    "$gte": batch["start_id"],
                    "$lt": batch["end_id"]
                }
            })
            # Process docs...
    """
    client, db = db_client()
    print(f"Analyzing collection: {_collection}")
    # Find min and max ObjectId
    min_id = 0
    max_id = 0
    ans = db[_collection].find({}, { "_id": 1 }).sort({ "_id": 1 }).limit(1)
    for a in ans:
        min_id = a["_id"]
    ans = db[_collection].find({}, { "_id": 1 }).sort({ "_id": -1 }).limit(1)
    for a in ans:
        max_id = a["_id"]
    
    
    if not min_id or not max_id:
        return []
   
    # Extract timestamps from ObjectIds
    min_timestamp = min_id.generation_time.timestamp()
    max_timestamp = max_id.generation_time.timestamp()
    
    # Handle edge case: all documents have same timestamp
    if min_timestamp == max_timestamp:
        return [{
            "batch_num": 1,
            "start_id": min_id,
            "end_id": max_id,
            "start_time": min_id.generation_time,
            "end_time": max_id.generation_time
        }]
    
    # Calculate time interval per chunk
    time_range = max_timestamp - min_timestamp
    interval = time_range / num_chunks
    """
    # Debug Session
    # min: _id: ObjectId('685b0998076cb66b7bc0fa78') - 2025-06-24T20:24:56.000Z
    # max: _id: ObjectId('685b0fd3a1671074dcc28117') - 2025-06-24T20:51:31.000Z
    # example _id=ObjectId('685b0998076cb66b7bc0fa79')
    # 2025-06-24T20:24:56.000Z
    # ObjectId('685b09e40000000000000000')
    # 2025-06-24T20:26:12.000Z
    """
    if partial_mode > 0:
        print("# ------ Full Collection Stats ------- #")
        print(f"min_timestamp: {datetime.datetime.fromtimestamp(min_timestamp)}, OBJ: {min_id}")
        print(f"max_timestamp: {datetime.datetime.fromtimestamp(max_timestamp)}, OBJ: {max_id}")
        print(f"time_range: {time_range}, interval: {interval}")
        min_obj = ObjectId.from_datetime(min_id.generation_time)
        print(f"reversibility test: idtime: {datetime.datetime.fromtimestamp(min_timestamp)}, GenTime: {datetime.datetime.fromtimestamp(min_obj.generation_time.timestamp())}")
        min_timestamp = min_timestamp + (time_range / 2)
        min_id = ObjectId.from_datetime(datetime.datetime.fromtimestamp(min_timestamp, tz=datetime.timezone.utc))
        # Use aggregation pipeline with $sample for efficient random sampling
        pipeline = [
            {"$match": {"_id": {"$gt": min_id}}},
            {"$limit": partial_mode},
            {"$project": {"_id": 1}},
            {"$sort": {"_id": 1}}  # Sort by _id to get proper ordering
        ]
        sampled_docs = list(db[_collection].aggregate(pipeline))
        if not sampled_docs or len(sampled_docs) == 0:
            print("❌ No documents found in collection!")
            return []
        actual_sample_size = len(sampled_docs)
        print(f"✅ Sampled {actual_sample_size} documents")
        # Get min and max ObjectId from sampled documents
        min_id = sampled_docs[0]["_id"]
        max_id = sampled_docs[-1]["_id"]
        print(f"Sample range:")
        print(f"  Min ObjectId: {min_id} ({min_id.generation_time})")
        print(f"  Max ObjectId: {max_id} ({max_id.generation_time})")
        # Extract timestamps from ObjectIds
        min_timestamp = min_id.generation_time.timestamp()
        max_timestamp = max_id.generation_time.timestamp()
        time_range = max_timestamp - min_timestamp
        interval = time_range / num_chunks
        
    # Generate batches
    batches = []
    for i in range(num_chunks):
        start_ts = min_timestamp + (i * interval)
        end_ts = min_timestamp + ((i + 1) * interval)
        
        # Last chunk: use actual max to avoid missing documents
        if i == num_chunks - 1:
            end_ts = max_timestamp
        
        # Create ObjectIds from timestamps (must use UTC timezone-aware datetimes)
        start_id = ObjectId.from_datetime(datetime.datetime.fromtimestamp(start_ts, tz=datetime.timezone.utc))
        end_id = ObjectId.from_datetime(datetime.datetime.fromtimestamp(end_ts, tz=datetime.timezone.utc))
        sort_key = random.randint(0, num_chunks + 100)
        
        batches.append({
            "sort_key" : sort_key,
            "batch_num": i + 1,
            "start_id": start_id,
            "end_id": end_id,
            "start_time": datetime.datetime.fromtimestamp(start_ts, tz=datetime.timezone.utc),
            "end_time": datetime.datetime.fromtimestamp(end_ts, tz=datetime.timezone.utc)
        })
    batches[num_chunks - 1]["end_id"] = max_id
    #print(f"Calculated {len(batches)} batches to complete embedding process")
    return batches

def vectorize_documents(partial_size = 0):
    num_procs = 4 #number of process threads to run
    num_chunks = 1000
    istart = 0
    if partial_size > 0 and partial_size < 10000:
        num_chunks = int(partial_size / 10)
    sep()
    print(f"Vectorizing Documents for {_database}.{_collection}")
    sep()
    batches = batch_data(num_chunks, partial_mode=partial_size)
    print(f"Processing {len(batches)} batches in {num_procs} processes")
    if partial_size > 0:
        print("Sample Batches")
        pprint.pprint(batches[0:9])  
        sep()
    # Prepare values for each thread
    batches.sort(key=lambda x: x['sort_key']) # Ensure random distribution of tranches for even process loading
    chunks_per_batch = int(num_chunks/num_procs)
    jobs = []
    inc = 0
    multiprocessing.set_start_method("spawn", force=True)
    for item in range(num_procs):
        params = {"batch_cnt": chunks_per_batch, "batches": batches[istart:(istart+chunks_per_batch-1)]}
        p = multiprocessing.Process(target=process_documents, args = (item,params))
        jobs.append(p)
        p.start()
        time.sleep(1)
        inc += 1
        istart += chunks_per_batch

    main_process = multiprocessing.current_process()
    print('Main process is %s %s' % (main_process.name, main_process.pid))
    for i in jobs:
        i.join()

def get_prompt_vector(prompt):
    result = voyage_client.embed([prompt], model=_model_id)
    return result.embeddings[0]

def test_search():
        # https://www.mongodb.com/docs/atlas/atlas-search/query-syntax/
    start_time = datetime.datetime.now()
    client, db = db_client()
    db = client[_output_database]
    collection = db[_output_collection]
    vector_index = "exposure_index"
    prompt = "Lovely day!"
    num_results = 4
    dedup = False
    full_sentence = False
    if "prompt" in ARGS:
        prompt = ARGS["prompt"]
    else:
        print("ERROR: enter a prompt= parameter")
        sys.exit(1)
    if "num" in ARGS:
        num_results = int(ARGS["num"])
    prompt_vector = get_prompt_vector(prompt)
    #pprint.pprint(prompt_vector)
    pipe = [
        {
        "$vectorSearch": {
            "index": vector_index,
            "queryVector": prompt_vector,
            "path": "embedding",
            "limit": num_results,
            "numCandidates": num_results}
        },
        {"$project": {
            "_id": 0,
            "id" : 1,
            "type_code": 1,
            "category_code" : 1,
            "chunk_text" : 1,
            "score": { "$meta": "vectorSearchScore" }}
        }
    ]
    #pprint.pprint(pipe)
    result = collection.aggregate(pipe)
    sep()
    print("#                     Search Results                    #")
    sep()
    print(f'Searching: {prompt}')
    for doc in result:
        #pprint.pprint(doc)
        print(f'# --------- Score: {doc["score"]} --------- #')
        print(f'Sentence: {doc["chunk_text"]}')
        print(f'Source: {doc["id"]}')
        
# ----------------------------------------------------------------- #
#     UTILITY METHODS
# ----------------------------------------------------------------- #

def bulk_writer(collection, bulk_arr, db, msg = ""):
    try:
        result = collection.bulk_write(bulk_arr, ordered=False)
        ## result = db.test.bulk_write(bulkArr, ordered=False)
        # Opt for above if you want to proceed on all dictionaries to be updated, even though an error occured in between for one dict
        #pprint.pprint(result.bulk_api_result)
        note = f'BulkWrite - mod: {result.bulk_api_result["nModified"]} {msg}'
        #file_log(note,locker,hfile)
        print(note)
    except BulkWriteError as bwe:
        print("An exception occurred ::", bwe.details)

def get_tools():
    result = db["mcp_tools"].find_one({"Name": "gemSearch"})
    return(result["tools"])

def sep(pid = ""):
    if pid != "":
        pid = f"[{pid}] "    
    print(f"# {pid}----------------------------------------------------------- #")
    
def test_embedding_data():
    extra_txt = """
We delivered better-than-expected first quarter results, as our go-to-market teams continue to execute well and capitalize on strong end-market demand for the MongoDB platform across enterprise use cases and emerging AI opportunities.\n
At the same time, we continue to show strong profitability, demonstrating we can drive durable revenue growth while simultaneously expanding margin. \n
Based on the momentum we are seeing in the business, we are raising our fiscal 2027 guidance," said CJ Desai, President and Chief Executive Officer of MongoDB.\n
With our recently expanded leadership across both product and sales, I’m confident that we have the right team in place to move with even greater velocity. \n
These changes sharpen our focus on delivering mission-critical innovation for our customers while scaling our global go-to-market engine, giving us high confidence in our ability to drive durable, long-term growth.\n
Net income was $4.4 million, or $0.05 per share, based on 81.6 million diluted weighted-average shares outstanding, for the first quarter of fiscal 2027. \n
This compares to a net loss of $37.6 million, or $0.46 per share, based on 81.1 million basic and diluted weighted-average shares outstanding in the year-ago period. \n
Non-GAAP net income was $112.3 million, or $1.32 per share, based on 85.3 million fully diluted weighted-average shares outstanding. This compares to a non-GAAP net income of $86.3 million, or $1.00, based on 86.3 million fully diluted weighted-average shares outstanding in the year-ago period.
"""
    str_docs = [
      "This is the SEC filing on Leafy Inc's Q2 2024 performance.\nThe company's revenue increased by 15% compared to the previous quarter." + extra_txt,
      "This is the SEC filing on Elephant Ltd's Q2 2024 performance.\nThe company's revenue decreased by 2% compared to the previous quarter." + extra_txt
    ]
    return str_docs

def test_create_embeddings():
    #uri = os.getenv('MONGO_URI', 'mongodb+srv://demo1.sf56l.mongodb.net')
    #db = MongoClient(uri)["claimxml"]
    client, db = db_client()
    #print(f"Database: {uri}")
    result = db[_collection].find({}).limit(200)
    process_documents(list(result))

# --------------------------------------------------------------- #
#             MAIN
# --------------------------------------------------------------- #

if __name__ == "__main__":
    ARGS = process_args(sys.argv)
    if "process" in ARGS:
        partial_size = 0
        if "size" in ARGS:
            partial_size = int(ARGS["size"])
        vectorize_documents(partial_size)  # 0 means do the entire collection
    elif "test" in ARGS:
        test_create_embeddings()
    elif "search" in ARGS:
        test_search()
    else:
        print("argument not found")