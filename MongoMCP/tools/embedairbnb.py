from bson.json_util import dumps
import json
import time
import re
import asyncio
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import UpdateOne

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

    def __init__(self, target_database: str = "sample_airbnb", target_collection: str = "listingsAndReviews"):
        """Initializes the DocumentVectorizer with configuration from settings.py.        
        
        Sets up Bedrock and MongoDB clients, and connects to source and target collections.
        """
        self.bedrock_client = BedrockClient(settings=settings)
        self.mongo_client = MongoDBClient(settings=settings)
        self.mongo_client.set_config(
            {
                "url": settings.mongo_url(),
                "database": target_database,
                "collection": target_collection,
            }
        )
        self.mongo_client.sync_connect_to_mongodb()
        self.is_voyage_embedding = settings.EMBEDDING_MODEL_ID.startswith("voyage-")
        self.source_collection = self.mongo_client.get_collection(target_collection)
        self.vector_collection = self.mongo_client.get_collection(target_collection)
        self.initial_object_name = target_collection


    async def generate_embedding_async(self, chunk_text: str) -> list:
        """Async version of embedding generation for use within an event loop."""
        if self.is_voyage_embedding:
            return await self.bedrock_client.generate_voyage_embeddings(chunk_text, is_query=False)
        return await self.bedrock_client.generate_embedding(chunk_text)

    
    async def _process_single_document_async(self, document: dict, fn_chunk) -> tuple:
        """Async version: processes a single document for embedding.
        
        Args:
            document: The document to process.
            fn_chunk: The chunking function to apply.
        
        Returns:
            Tuple of (object_id, embedding, duration, success, error_message).
        """
        start_time = time.time()
        object_id = document["_id"]
        
        try:
            chunk = dumps(fn_chunk(document))
            
            # clear out the json special characters
            pattern = r'[{}\[\]",]'
            clean_chunk = re.sub(pattern, '', chunk)
            
            # Generate embedding for the chunk text asynchronously
            embedding = await self.generate_embedding_async(clean_chunk)
            
            duration = time.time() - start_time
            return (object_id, embedding, duration, True, None)
        
        except Exception as e:
            duration = time.time() - start_time
            return (object_id, None, duration, False, str(e))

    def _process_batch_of_documents(self, batch: list, fn_chunk, embedding_field: str) -> tuple[int, int]:
        """Batch thread: concurrently vectorize all documents, then write them to MongoDB.
        
        Each batch thread:
          1. Spawns an async coroutine per document and awaits all embeddings concurrently.
          2. Bulk-writes the successful results to MongoDB itself.
        
        Args:
            batch: List of documents to process.
            fn_chunk: The chunking function to apply to documents.
            embedding_field: The field name to store the embedding in MongoDB.
        
        Returns:
            Tuple of (success_count, failure_count).
        """
        # ── BATCH THREAD starts here ─────────────────────────────────────────
        # This method runs inside a worker thread from the ThreadPoolExecutor.
        # Each invocation is one independent batch thread.

        # Create a private event loop for this thread so async code can run
        # without interfering with other threads or the main thread.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # ── ASYNC TASKS spawned here (one per document in this batch) ────
            # Each coroutine calls the embedding API independently.
            # asyncio.gather() runs all of them concurrently on this thread's
            # event loop and waits until every one has returned.
            tasks = [
                self._process_single_document_async(doc, fn_chunk)
                for doc in batch
            ]
            results = loop.run_until_complete(asyncio.gather(*tasks))
            # ── ASYNC TASKS all done; event loop idle ────────────────────────
        finally:
            loop.close()  # release the event loop for this thread
        
        # Build updates and write directly from this thread.
        # The MongoDB bulk_write is the only point where a pool connection
        # is borrowed; it is returned as soon as bulk_write returns.
        batch_updates = []
        failures = 0
        for object_id, embedding, duration, success, error in results:
            if success:
                batch_updates.append(
                    UpdateOne(
                        {"_id": object_id},
                        {"$set": {embedding_field: embedding}}
                    )
                )
            else:
                print(f"{object_id} failed: {error}")
                failures += 1
        
        if batch_updates:
            self.vector_collection.bulk_write(batch_updates)
        
        return (len(batch_updates), failures)
        # ── BATCH THREAD ends here; worker slot returned to the pool ─────────

    @staticmethod
    def _batch_cursor(cursor, batch_size: int):
        """Lazily yield fixed-size batches from a MongoDB cursor without loading all docs into RAM."""
        batch = []
        for doc in cursor:
            batch.append(doc)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def process_documents(self, documents_limit: int = 10, fn_chunk=None) -> None:
        """Processes documents using num_threads batch threads running in parallel.
        
        Args:
            documents_limit: Maximum number of documents to process.
            fn_chunk: The chunking function to apply to documents.
        
        Architecture:
          - Documents are split into batches of docs_per_thread.
          - Up to num_threads batch threads run concurrently at all times.
          - Each batch thread: vectorizes all docs asynchronously, then bulk-writes
            its own results to MongoDB independently.
          - The main thread uses as_completed() to keep the pool saturated,
            starting a new batch thread whenever one finishes.
        """
        start_time_total = time.time()
        processed_count = 0
        num_threads = 20
        docs_per_thread = 10 # batches of 10 seem to work best. the vector API will be the slow point.
        embedding_field = "embedding"
        if self.is_voyage_embedding:
            embedding_field = "voyage_embedding"
        
        print(f"Begin processing up to {documents_limit} documents ({num_threads} threads × {docs_per_thread} docs each)")
        print(f"using embedding model: {settings.EMBEDDING_MODEL_ID}")
        print("This may take a while...")
        
        try:
            # Open a lazy MongoDB cursor — documents are fetched on demand,
            # not loaded into RAM all at once.
            cursor = self.source_collection.find().limit(documents_limit)
            batch_gen = self._batch_cursor(cursor, docs_per_thread)
            batch_idx = 0
            
            def submit_next(executor, pending: dict) -> bool:
                """Read the next batch from the cursor and submit it as a new batch thread.
                Returns False when the cursor is exhausted."""
                nonlocal batch_idx
                try:
                    batch = next(batch_gen)  # reads docs_per_thread docs from MongoDB
                    # ── BATCH THREAD submitted here ──────────────────────────
                    future = executor.submit(
                        self._process_batch_of_documents, batch, fn_chunk, embedding_field
                    )
                    pending[future] = (batch_idx, time.time())
                    batch_idx += 1
                    return True
                except StopIteration:
                    return False  # cursor exhausted; no more batches to submit
            
            # ── MAIN THREAD: ThreadPoolExecutor manages up to num_threads workers ──
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                pending = {}  # maps future → (batch_idx, start_time)
                
                # Prime the pool: submit the first num_threads batches so all
                # worker slots are busy from the start.
                for _ in range(num_threads):
                    if not submit_next(executor, pending):
                        break  # fewer total batches than num_threads
                
                # Main loop: block until any one batch thread finishes,
                # then immediately submit the next batch to keep the pool full.
                while pending:
                    # Blocks here until the next batch thread completes.
                    done_future = next(as_completed(pending))
                    idx, batch_start_time = pending.pop(done_future)
                    try:
                        success_count, failure_count = done_future.result()
                        processed_count += success_count
                        duration = time.time() - batch_start_time
                        print(
                            f"Batch {idx + 1} complete: "
                            f"{success_count} written, {failure_count} failed "
                            f"in {duration:.2f}s. Total so far: {processed_count}"
                        )
                    except Exception as e:
                        print(f"Batch {idx + 1} raised an exception: {e}")
                    
                    # Slot freed — submit the next batch immediately.
                    submit_next(executor, pending)
                # ── All batch threads done; executor shuts down here ─────────

        except KeyboardInterrupt:
            print("   user canceled, stopping")
        finally:
            self.mongo_client.client.close()
        
        end_time_total = time.time()
        elapsed = end_time_total - start_time_total
        mins, secs = divmod(elapsed, 60)
        duration_str = f"{int(mins)}m {secs:.1f}s" if mins else f"{secs:.1f}s"
        print(f"{processed_count} vectors completed in {duration_str}.")


def chunk_Airbnb_document(document: dict) -> dict:
        """Chunks an Airbnb document for embedding.
            write your own method specific to your dataset
        Args:
            document: The original Airbnb document to be chunked.
        
        Returns:
            specific fields extracted from the document.
        
        This method extracts relevant fields from the Airbnb document and prepares them for embedding.
        """        
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
        
        return for_vector



def main():
    """Entry point to run the DocumentVectorizer.
    
    Instantiates the class and processes 6000 documents by default.
    """
    vectorizer = DocumentVectorizer("sample_airbnb", "listingsAndReviews")
    vectorizer.process_documents(6000, fn_chunk=chunk_Airbnb_document)

if __name__ == "__main__":
    main()
