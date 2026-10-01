"""
MongoDB AI/LLM client functions 

"""

import datetime
import json
import re
import asyncio
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
import logging
import voyageai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()  
        return json.JSONEncoder.default(self, obj)


class VoyageClient:
    """
    Voyage Client for Embeddings.
    """
    def __init__(self, settings):
        self.settings = settings
        self.voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
        self.message_handler = None
        self.show_response_progress = True
        
    def _emit_progress(self, message_handler: Optional[Callable], message: str, status: str = "Processing") -> None:
        """Emit optional progress updates without impacting request flow."""
        if not message_handler:
            return
        try:
            message_handler(message, status=status)
        except Exception:
            # Progress updates should never fail the main LLM flow.
            return

    async def generate_embedding(self, input_data, input_type: str = "query") -> list:
        """Generates an embedding for the input text using the given model.

        Args:
            text: Input text or array of texts to embed.
            input_type: Type of input ("query" or "document"). Defaults to "query".

        Returns:
            list: Embedding vector (list of floats) produced by the model.
        """
        #body = json.dumps({"inputText": text})
        if isinstance(input_data, str):
            input_data = [input_data]
        # Invoke the Voyage embedding model (e.g., voyage-3.5) specified in config
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.voyage_client.embed(
                    texts=input_data,
                    model=self.settings.voyage_model,
                    input_type=input_type
                    )
        )
        # Parse the response and extract the embedding vector
        return response.embeddings

