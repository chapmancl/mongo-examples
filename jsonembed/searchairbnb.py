import json
import boto3
from botocore.exceptions import ClientError
import re
import time
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.collection import Collection
from bson.json_util import dumps

import settings

class QueryProcessor:
    """A class to process user queries using vector search and Claude LLM via Bedrock.
    This is a simple RAG example for Airbnb data.
    Manages configuration, MongoDB Atlas vector search, and AWS Bedrock interactions
    to retrieve and aggregate facts based on user questions.
    """

    def __init__(self):
        """Initializes the QueryProcessor with configuration from settings.py.
        
        Sets up Bedrock and MongoDB clients, and connects to the vector collection.
        """
        # Conversation history (starts empty)
        self.history = None
        # AWS session objects
        self.bedrock_client = None
        self._create_bedrock_client()
        
        # Initialize MongoDB client for vector search
        self.mongo_client = self._create_mongo_client()
        # Connect to the MongoDB collection for vectorized data
        self.collection = self._get_vector_collection()
        
        self.mcp_client = fastmcp.Client(settings.mong_mcp)
        self.mcp_tools_config = None
        
    def _create_bedrock_client(self) -> None:
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws_region
        )
    
    def _create_mongo_client(self) -> MongoClient:
        # Connect to MongoDB Atlas using the URI and Server API version 1
        return MongoClient(settings.MongoURI, server_api=ServerApi('1'))

    def _get_vector_collection(self) -> Collection:
        database = self.mongo_client[settings.monogo_database]
        return database[settings.vector_collection]

    def generate_embedding(self, text: str) -> list:
        """Generates an embedding for the input text using the given model.
        
        Args:
            text: Input text to embed.
        
        Returns:
            list: Embedding vector (list of floats) produced by the model.
        """
        body = json.dumps({"inputText": text})        
        # Invoke the Bedrock embedding model (e.g., Titan Embeddings) specified in config
        response = self.bedrock_client.invoke_model(
            modelId=settings.EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        # Parse the response and extract the embedding vector
        return json.loads(response["body"].read())["embedding"]

    def search_similar_documents(self, query_vector: list, filters: list = None, limit: int = 100, candidates: int = 3000) -> list:
        """Performs a vector search in MongoDB Atlas to find similar documents.
        
        Args:
            query_vector: Query embedding vector to search with.
            filters: List of (key, value) tuples for metadata pre filtering (default: None).
            limit: Maximum number of results to return (default: 100).
            candidates: Number of candidate documents to consider (default: 1000).
        
        Returns:
            list: List of json strings combining chunk text and metadata for each result.
        """
        # Define the MongoDB Atlas vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": settings.vecotr_index,  # Name of the vector index
                    "path": "embedding",        # Field storing embeddings
                    "queryVector": query_vector, # Vector to compare against
                    #"exact":True,
                    "limit": limit,             # Max results to return
                    "numCandidates": candidates # Number of candidates to evaluate
                }
            },
            {
                "$project": {                   # Shape the output
                    "embedding":0,
                    "images":0,
                    #"reviews":0,
                    "host":0,
                    "neighborhood_overview":0,
                    "summary":0,
                    "space":0,
                    "transit":0,
                    "access":0,
                    "score": {"$meta": "vectorSearchScore"},  # Include similarity score                    
                }
            },
            {
                "$sort": {
                    "score": -1
                }
            }
        ]

        # Apply filters to narrow the search if provided        
        if filters:
            match_filter = {}
            if len(filters) > 1:
                # Use $and for multiple filters
                match_filter = {"$and": []}
                for key, value in filters:
                    match_filter["$and"].append({key: value})
                    
            else:
                # Single filter case
                key, value = filters[0]
                match_filter[key] = value
            pipeline[0]["$vectorSearch"]["filter"] = match_filter

        output = []
        # Execute the vector search aggregation
        results = self.collection.aggregate(pipeline)
        # Combine chunk text and metadata into a single string for each result
        bool_isfirst = True        
        for result in results:            
            if bool_isfirst: 
                #print(result)
                bool_isfirst =False
            if result["score"] > 0.50:
                output.append(dumps(result))        
        return output
        
    def _invoke_claude(self, prompt: str) -> tuple:
        """Sends a prompt to Claude via Bedrock and updates conversation history.
        
        Args:
            prompt: User prompt to send to the LLM.
        
        Returns:
           assistant message (str)
        """
        # Initialize history if not provided
        if self.history is None:
            self.history = []
        
        # Add the user prompt to the conversation history
        self.history.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        
        # Prepare the request body for Claude's Messages API
        body = json.dumps({
            "messages": self.history,       # Full conversation history
            "anthropic_version": "bedrock-2023-05-31",  # Bedrock-specific version
            "max_tokens": 1000,            # Max output tokens
            "top_k": 250,                  # Sampling diversity parameter
            "temperature": 1,              # Randomness in response generation
            "top_p": 0.999                 # Cumulative probability for token selection
        })

        # Invoke the Claude model specified in config
        response = self.bedrock_client.invoke_model(
            modelId=settings.LLM_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        # Decode and parse the response
        response_body = response["body"].read().decode('utf-8')
        assistant_message = json.loads(response_body)["content"][0]["text"]
        # Add the assistant's response to the history
        self.history.append({"role": "assistant", "content": [{"type": "text", "text": assistant_message}]})

        return assistant_message

    def extract_filters(self, question: str) -> list or None:
        """Extracts metadata filters from a user question using regex patterns.
            This will be specific to the data in the vector collection. For this dataset we have member_id and field in the metadata section
        
        Args:
            question: User question to analyze.
        
        Returns:
            list or None: List of (key, value) tuples for filtering, or None if no filters found.
        """
        filters = []  # List to store extracted filters

        # Check for listing id. really, if we're here we should just skip the vector bit and return the full record
        pattern = r'(?i)id=(\d{8})'
        match = self._split_filters(pattern, question)
        if match:
            filters.append(("_id", match))             
            print(f"found objectid, skipping vector search: {filters}")  
            return (filters, True)

        # Check for payment-related keywords (case-insensitive)
        pattern = r'(?i)country=([^"]+)'
        match = self._split_filters(pattern, question)
        if match:
            filters.append(("address.country_code", match))

        # Check for market filter
        pattern = r'(?i)market=([^"]+)'
        match = self._split_filters(pattern, question)
        if match:
            filters.append(("address.market", match))
        
        # Check for beds filter
        pattern = r'(?i)beds=(\d+)'
        match = self._split_filters(pattern, question)
        if match:
            filters.append(("beds", int(match)))
        
        # Check for bedrooms filter
        pattern = r'(?i)bedrooms=(\d+)'
        match = self._split_filters(pattern, question)
        if match:
            filters.append(("bedrooms", int(match)))


        # Return filters if any were found, otherwise None
        if filters:
            print(filters)  # Log the extracted filters for debugging
            return (filters, False)
        return None

    def _split_filters(self, pattern, question) -> str:
        """split the filters on a comma - this way we can list out multiple filters at the end. ie country=US,beds=2
        """
        match = re.search(pattern, question)
        filter_value = None
        if match:
            temp = match.group(1)
            filter_value = temp.split(",",1)[0]
        return filter_value

    def query_claude(self, question: str, history: list or None = None) -> tuple:
        """Formats a question with optional context and sends it to Claude.  This is stateless in case we're calling from an API.
            The history would be stored in browser perhaps?
        
        Args:
            question: User question or full prompt 
            history: optional list of historical questions and assistant answers. overwrites internal history      
        
        Returns:
            tuple: (assistant response (str), updated history (list))
        """
        # update history
        if history:
            self.history = history
        # Invoke Claude with the formatted prompt
        assistant_message = self._invoke_claude(question)
        return assistant_message, self.history
    
    def retrieve_aggregate_facts(self, question: str, history: list or None = None) -> tuple:
        """Processes a user question to retrieve and aggregate facts using vector search and LLM. This is stateless in case we're calling from an API.
            The history would be stored in browser perhaps?
        
        Args:
            question: User question to process.
            history: optional list of historical questions and assistant answers. overwrites internal history
        
        Returns:
            tuple: (LLM response (str), updated history (list))
        """
        mongo_results = None
        # Measure time for extracting filters
        start_time = time.time()
        filters = self.extract_filters(question)
        duration = time.time() - start_time
        print(f"extract_filters completed in {duration:.4f} seconds.")

        # do we have a document id??
        if filters and filters[1]:
            key, value = filters[0][0]
            print(f"found objectId filter. pulling single document: {value}")
            mongo_results = self.collection.find_one({'_id': value})
            del(mongo_results["embedding"])
        else:
            filter_list = None
            if filters:
                filter_list = filters[0]
            # Generate embedding for the question
            start_time = time.time()
            query_vector = self.generate_embedding(question)
            duration = time.time() - start_time
            print(f"generate_embedding completed in {duration:.4f} seconds.")

            # Perform vector search with fixed limits
            start_time = time.time()
            # we can probably lower this based on filters... leaving it for now
            limit = 5      # Maximum results to return
            candidates = 400  # Number of candidates to evaluate
            mongo_results = self.search_similar_documents(query_vector, filter_list, limit, candidates)
            duration = time.time() - start_time
            print(f"search_similar_documents completed in {duration:.4f} seconds.")

        # Default response if no results are found
        response_message = "no context from vectors"
        if mongo_results:
            # Combine vector results into a single context string
            context = "\n".join(mongo_results)
            # Query Claude with the question and vector search results as context 
            start_time = time.time()
            try:
                # Format the prompt with question and context if provided, otherwise just the question
                prompt = question if context is None else f"Use the following context to answer the question. Context: {context}\n Question: {question}"
                response_message, htemp = self.query_claude(prompt, history) # don't need htemp here
                # this keeps overflowing claude... going to remove the top 2 every time
                if len(self.history) > 4:
                    self.history = self.history[2:]
            except ClientError as error:
                # Handle AWS Bedrock errors
                error_code = error.response['Error']['Code']                
                if error_code == 'ValidationException':
                    # if input exceeds token limit just drop it all
                    self.history = None 
                    print("too much history, clearing...", error)
                elif error_code in ['ExpiredTokenException', 'ExpiredToken']:
                    raise
                else:
                    # Log other errors without modifying history
                    print("Some other client error occurred:", error.response)
            duration = time.time() - start_time
            print(f"query_claude completed in {duration:.4f} seconds.")

        return response_message, self.history

    def run(self) -> None:
        """Runs an interactive loop on the command line to handle user questions.
        
        Supports direct Claude queries (prefixed with 'ask'), MCP tool queries (prefixed with 'mcp'),
        or vector-backed fact retrieval.
        """
        print("Enter questions (Press Ctrl+C to stop):")
        print("Commands:")
        print("  ask <question> - Direct Claude query without vector search")
        print("  <question> - Full query with vector search and Claude (clasic RAG)")
        print("  clear - Clear conversation history")
        try:
            while True:
                # Get user input and strip whitespace
                user_input = input("Question: ").strip()
                answer = "unknown"  # Default answer if no processing occurs
                if not user_input:
                    answer = "Not a valid question"
                elif user_input.startswith("ask"):
                    # Direct Claude query without vector search
                    user_input = user_input.removeprefix("ask").strip()
                    answer, history = self.query_claude(user_input)  
                elif user_input.startswith("clear"):
                    self.history = None
                    answer = "history cleared..."
                else:
                    answer, history = self.retrieve_aggregate_facts(user_input) # don't need histroy here, but we would need to return it for a stateless API                    
                if answer:
                    print(f"Answer: {answer}")
        except ClientError as error:
            error_code = error.response['Error']['Code']
            if error_code in ['ExpiredTokenException', 'ExpiredToken']:
                print("AWS Token has expired!", error)
            elif error_code == 'ValidationException':
                # if input exceeds token limit just drop it all
                self.history = None 
                print("too much history, clearing...", error)
                self.run()
            else:
                # Log other errors                 
                print("Some other AWS client error occurred:", error.response)
        except KeyboardInterrupt:
            # Handle user interruption
            print("\nKeyboard interrupt received, exiting...")

def main():    
    processor = QueryProcessor()
    processor.run()

if __name__ == "__main__":
    main()
