from bson.json_util import dumps
import json
import time
import re
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_settings import settings
from mongomcp.bedrock_client import BedrockClient
from mongomcp.mongodb_client import MongoDBClient

class DocumentVectorizer:
    """A class to process JSON documents, chunk them, vectorize chunks, and store in MongoDB.
    
    This class integrates AWS Bedrock for vectorization and MongoDB Atlas for storage,
    handling document retrieval, chunking, embedding generation, and persistence.
    """

    def __init__(self):
        """Initializes the DocumentVectorizer with configuration from settings.py.        
        
        Sets up Bedrock and MongoDB clients, and connects to source and target collections.
        """
        self.bedrock_client = BedrockClient(settings=settings)
        self.mongo_client = MongoDBClient(settings=settings)
        self.mongo_client.set_config(
            {
                "url": settings.mongo_url(),
                "database": "sample_airbnb",
                "collection": "listingsAndReviews",
            }
        )
        self.mongo_client.sync_connect_to_mongodb()

        self.source_collection = self.mongo_client.get_collection("listingsAndReviews")
        self.vector_collection = self.mongo_client.get_collection("listingsAndReviews")
        self.initial_object_name = "listingsAndReviews"


    def generate_embedding(self, chunk_text: str) -> list:
        """Run the async BedrockClient.generate_embedding method in this sync script."""
        return asyncio.run(self.bedrock_client.generate_embedding(chunk_text))

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
                                
                # use the whole document as a single Chunk
                chunk = dumps(for_vector)

                # clear out the json special characters
                pattern = r'[{}\[\]",]'
                clean_chunk = re.sub(pattern, '', chunk)

                try:
                    # Generate embedding for the chunk text
                    embedding = self.generate_embedding(clean_chunk)
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
            self.mongo_client.client.close()
        
        # Log total processing time
        end_time_total = time.time()
        duration_total = end_time_total - start_time_total
        print(f"{processed_count} completed in {duration_total:.4f} seconds.")

def main():
    """Entry point to run the DocumentVectorizer.
    
    Instantiates the class and processes 6000 documents by default.
    """
    vectorizer = DocumentVectorizer()
    vectorizer.process_documents(6000)

if __name__ == "__main__":
    main()
