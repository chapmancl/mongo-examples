import boto3
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.collection import Collection
from bson.json_util import dumps
import json
import time
import re
import settings

class DocumentVectorizer:
    """A class to process JSON documents, chunk them, vectorize chunks, and store in MongoDB.
    
    This class integrates AWS Bedrock for vectorization and MongoDB Atlas for storage,
    handling document retrieval, chunking, embedding generation, and persistence.
    """

    def __init__(self):
        """Initializes the DocumentVectorizer with configuration from settings.py.        
        
        Sets up Bedrock and MongoDB clients, and connects to source and target collections.
        """
        # Initialize AWS Bedrock client for vectorization
        self.bedrock_client = self._create_bedrock_client()
        # Initialize MongoDB client for document retrieval and storage
        self.mongo_client = self._create_mongo_client()
        # Connect to the source collection for retrieving documents
        self.source_collection = self._get_source_collection()
        # Connect to the target collection for storing vectorized chunks
        self.vector_collection = self._get_vector_collection()
        # top level prefix to start object hierarchy. using the collection here
        self.initial_object_name = settings.source_collection


    def _create_bedrock_client(self) -> boto3.client:
        return boto3.client(
            'bedrock-runtime',
            region_name=settings.aws_region
        )

    def _create_mongo_client(self) -> MongoClient:        
        return MongoClient(settings.MongoURI, server_api=ServerApi('1'))

    def _get_source_collection(self) -> Collection:
        database = self.mongo_client[settings.monogo_database]
        return database[settings.source_collection]

    def _get_vector_collection(self) -> Collection:
        database = self.mongo_client[settings.monogo_database]
        return database[settings.vector_collection]

    def chunk_entire_doc(self, json_object: dict) -> str:
        """converts the whole document to a single chunk
        
        """
        return dumps(json_object)

    def extract_fields(self, json_object: dict, parent_field: str = None) -> list:
        """Recursively processes a JSON object to create text chunks from its fields.
        
        Args:
            json_object: JSON object (dict, list, or primitive) to process.
            parent_field: Parent field name for nested context (default: None).
        
        Returns:
            list: List of dictionaries with 'text' (chunk content) and optional 'metadata'.
        
        Chunks are created by flattening fields into strings, with nested objects processed recursively.
        """
        chunks = []  # Stores the final list of chunk dictionaries
        field_strings = []  # Temporarily collects strings for a single chunk
        
        if isinstance(json_object, dict):
            # Iterate over dictionary key-value pairs
            for key, value in json_object.items():
                if isinstance(value, (dict, list)):
                    # Recursively process nested dictionaries or lists, building the parent field path
                    chunks.extend(self.extract_fields(value, f"{parent_field}.{key}" if parent_field else key))
                elif str(value).strip():
                    # Add non-empty primitive values as 'key:value' strings
                    field_strings.append(f"{key}:{value}")
        
        elif isinstance(json_object, list):
            # Handle lists by concatenating items under a single list items string
            # this will not handle nested objects inside a list, for that you will need to recursively call this function 
            # and handle it after the isinstance call.
            list_items = parent_field.removeprefix(f"{self.initial_object_name}.") if parent_field else ""
            for item in json_object:
                if isinstance(item, (dict, list)):
                    # Append nested objects/lists as part of the list items string
                    list_items = list_items + " " + str(item).strip()
                elif str(item).strip():
                    # Add non-empty primitive items directly
                    field_strings.append(item)
            # if we have list items, add them to the field strings
            if list_items.strip():
                field_strings.append(list_items)
        
        else:
            # Log unexpected standalone values (not typically reached with proper JSON)
            print(f"found orphan string: {json_object}:{parent_field}")

        # If any strings were collected, create a chunk
        if field_strings:
            chunk = {"text": " ".join(field_strings)}  # Join strings into a single text field
            if parent_field:
                chunk["metadata"] = {"field": parent_field}  # Add metadata with field context
            chunks.append(chunk)

        return chunks


    def iterate_json(self, json_object, current_key: str = None, parent_field: str = None) -> list:
        """Recursively iterates over a JSON object to create serialized chunks.
        
        Args:
            json_object: JSON object (dict, list, or primitive) to process.
            current_key: Current key being processed (default: None).
            parent_field: Parent field name for context (default: None).
        
        Returns:
            list: List of JSON-serialized strings representing chunks.
        
        Alternative chunking method (currently unused), serializing each primitive value.
        """
        chunks = []
        if isinstance(json_object, dict):
            # Set parent field and recurse over dictionary items
            parent_field = current_key
            for key, value in json_object.items():
                chunks.extend(self.iterate_json(value, key, parent_field))
        elif isinstance(json_object, list):
            # Recurse over list items with the same parent field
            for item in json_object:
                chunks.extend(self.iterate_json(item, current_key, parent_field))
        else:
            # Create a chunk for primitive values with key-value pair
            chunk = {current_key: json_object}
            if parent_field:
                chunk["metadata"] = {"parent": parent_field}
            chunks.append(json.dumps(chunk))  # Serialize to JSON string
        return chunks

    def vectorize_chunk(self, chunk_text: str) -> list:
        """Generates an embedding for a text chunk using the supplied model ie. Bedrock's Titan Embeddings model.
        
        Args:
            chunk_text: Text chunk to vectorize.
        
        Returns:
            list: Embedding vector produced by the model.
        
        Uses the MODEL_ID from settings to invoke the Bedrock embedding model.
        """
        request = json.dumps({"inputText": chunk_text})
        response = self.bedrock_client.invoke_model(
            modelId=settings.EMBEDDING_MODEL_ID,
            body=request
        )
        model_response = json.loads(response["body"].read())
        return model_response['embedding']

    def process_documents(self, documents_limit: int = 10) -> None:
        """Processes documents, chunks them, vectorizes, and stores in MongoDB.
        
        Args:
            documents_limit: Maximum number of documents to process (default: 10).
        
        Retrieves documents from source collection, processes them, and stores results.
        Tracks and logs processing time for each document and the total run.
        """
        start_time_total = time.time()  # Start time for the entire process
        processed_count = 0  # Counter for processed documents
        
        try:
            # Fetch documents with the specified limit
            cursor = self.source_collection.find().limit(documents_limit)
            for document in cursor:
                start_time = time.time()  # Start time for this document
                object_id = document["_id"]
                for_vector = {}
                
                for_vector["name"] = document["name"]
                for_vector["summary"] = document["summary"]
                #for_vector["space"] = document["space"]
                for_vector["description"] = document["description"]
                #for_vector["notes"] = document["notes"]
                for_vector["property_type"] = document["property_type"]
                for_vector["room_type"] = document["room_type"]
                for_vector["bed_type"] = document["bed_type"]
                for_vector["address"] = document["address"]
                #for_vector["reviews"] = document["reviews"]
                                
                # Chunk the document into text segments. the top level parent name is the collection name
                # alternatively you can use iterate_json and each field is its own chunk
                chunk = self.chunk_entire_doc(for_vector)

                # clear out the json special characters
                pattern = r'[{}\[\]",]'
                clean_chunk = re.sub(pattern, '', chunk)

                try:
                    # Generate embedding for the chunk text
                    embedding = self.vectorize_chunk(clean_chunk)
                    # Store the vectorized chunks in MongoDB
                    self.vector_collection.update_one(
                        {"_id": object_id}, 
                        {"$set": {"embedding": embedding}}
                    )   
                except Exception as e:
                    print(clean_chunk)
                    print(e)            

                # Log processing time for this document
                end_time = time.time()
                duration = end_time - start_time
                print(f"{object_id} completed in {duration:.4f} seconds.")
                processed_count += 1

        except KeyboardInterrupt:
            # Handle user interruption
            print("   user canceled, stopping")
        finally:            
            self.mongo_client.close()
        
        # Log total processing time
        end_time_total = time.time()
        duration_total = end_time_total - start_time_total
        print(f"{processed_count} completed in {duration_total:.4f} seconds.")

def main():
    """Entry point to run the DocumentVectorizer.
    
    Instantiates the class and processes 1000 documents by default.
    """
    vectorizer = DocumentVectorizer()
    vectorizer.process_documents(6000)

if __name__ == "__main__":
    main()
