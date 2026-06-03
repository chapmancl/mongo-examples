"""
Plain async handler functions for collection-backed query tools.

These are never registered directly with @mcp.tool(). register_query_tools()
wraps them under config-driven names so multiple tool entries can point at the
same underlying handler with different collection/index values.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Annotated

from pydantic import Field

logger = logging.getLogger(__name__)


def build_query_handler_fns(mongo_server, llm_client) -> dict:
    """Return {handler_name: async_fn} capturing mongo_server and llm_client via closure."""

    async def vector_search(
        query_text: Annotated[str, Field(description="Natural language query describing what to search for.")],
        limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=50)] = 10,
        num_candidates: Annotated[int, Field(default=100, description="Number of candidates for vector search.", ge=10, le=1000)] = 100,
        filters: Annotated[Optional[List], Field(default=None, description="Optional list of filters to narrow results.")] = None,
        collection: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        index: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        vector_path: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        projection: Annotated[Optional[Dict], Field(default=None, description="Injected from tool config by middleware.")] = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not collection or not str(collection).strip():
                raise ValueError("handlers.vector_search:collection must be a non-empty string")
            if not query_text or not isinstance(query_text, str):
                return {"error": "handlers.vector_search:query_text must be a non-empty string"}
            embedding_result = await llm_client.generate_embedding(query_text)
            vector_qry = embedding_result.get("vector") if isinstance(embedding_result, dict) else embedding_result
            if not vector_qry or not isinstance(vector_qry, list):
                return {"error": f"Embedding generation returned unexpected format: {type(embedding_result)}"}
            results = await mongo_server.vector_search(collection, vector_qry, filters, limit, num_candidates, index=index, vector_path=vector_path, projection=projection)
            return {
                "results": json.loads(json.dumps(results, default=str)),
                "count": len(results),
                "query_info": {
                    "embedding_model": embedding_result.get("embedding_model") if isinstance(embedding_result, dict) else None,
                    "limit": limit,
                    "num_candidates": num_candidates,
                },
            }
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {"error": f"Error executing vector_search: {str(e)}"}

    async def text_search(
        query_text: Annotated[str, Field(description="Keywords or phrases to search for.")],
        limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=100)] = 10,
        collection: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not query_text:
                return {"error": "query_text is required"}
            results = await mongo_server.text_search(collection, query_text, limit)
            return {
                "results": json.loads(json.dumps(results, default=str)),
                "count": len(results),
                "query_info": {"query_text": query_text, "limit": limit},
            }
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return {"error": f"Error executing text_search: {str(e)}"}

    async def geospatial_search(
        longitude: Annotated[float, Field(description="Longitude for the center point in WGS84.", ge=-180, le=180)],
        latitude: Annotated[float, Field(description="Latitude for the center point in WGS84.", ge=-90, le=90)],
        limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=100)] = 10,
        max_distance_meters: Annotated[Optional[float], Field(default=None, description="Optional maximum distance from center in meters.", ge=0)] = None,
        min_distance_meters: Annotated[Optional[float], Field(default=None, description="Optional minimum distance from center in meters.", ge=0)] = None,
        filters: Annotated[Optional[List], Field(default=None, description="Optional list of filters in [field, value] format.")] = None,
        collection: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        geo_field: Annotated[Optional[str], Field(default=None, description="Injected from tool config location_field by middleware.")] = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            results = await mongo_server.geospatial_search(
                collection=collection,
                longitude=longitude,
                latitude=latitude,
                max_distance_meters=max_distance_meters,
                min_distance_meters=min_distance_meters,
                filters=filters,
                limit=limit,
                geo_field=geo_field,
            )
            return {
                "results": json.loads(json.dumps(results, default=str)),
                "count": len(results),
                "query_info": {
                    "longitude": longitude,
                    "latitude": latitude,
                    "limit": limit,
                    "max_distance_meters": max_distance_meters,
                    "min_distance_meters": min_distance_meters,
                    "geo_field": geo_field,
                },
            }
        except Exception as e:
            logger.error(f"Geospatial search failed: {e}")
            return {"error": f"Error executing geospatial_search: {str(e)}"}

    async def hybrid_search(
        query_text: Annotated[str, Field(description="Natural language query — used for both semantic vector search and BM25 full-text scoring. $rankFusion combines both signals.")],
        limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=50)] = 10,
        num_candidates: Annotated[int, Field(default=100, description="Vector search candidate pool size.", ge=10, le=1000)] = 100,
        filters: Annotated[Optional[List], Field(default=None, description="Optional list of [field, value] filters. Applied as a $match stage after $rankFusion scoring — narrows fused results by exact field value.")] = None,
        vector_weight: Annotated[float, Field(default=0.6, description="Weight for vector similarity score in fusion (0.0–1.0).", ge=0.0, le=1.0)] = 0.6,
        text_weight: Annotated[float, Field(default=0.4, description="Weight for BM25 text score in fusion (0.0–1.0).", ge=0.0, le=1.0)] = 0.4,
        collection: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        vector_index: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        text_index: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        vector_path: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        text_fields: Annotated[Optional[List[str]], Field(default=None, description="Injected from tool config by middleware.")] = None,
        projection: Annotated[Optional[Dict], Field(default=None, description="Injected from tool config by middleware.")] = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not collection or not str(collection).strip():
                raise ValueError("handlers.hybrid_search: collection must be a non-empty string")
            if not query_text or not isinstance(query_text, str):
                return {"error": "handlers.hybrid_search: query_text must be a non-empty string"}
            embedding_result = await llm_client.generate_embedding(query_text)
            vector_qry = embedding_result.get("vector") if isinstance(embedding_result, dict) else embedding_result
            if not vector_qry or not isinstance(vector_qry, list):
                return {"error": f"Embedding generation returned unexpected format: {type(embedding_result)}"}
            results = await mongo_server.hybrid_search(
                collection=collection,
                vector_qry=vector_qry,
                query_text=query_text,
                limit=limit,
                num_candidates=num_candidates,
                filters=filters,
                vector_index=vector_index,
                text_index=text_index,
                vector_path=vector_path,
                text_fields=text_fields,
                vector_weight=vector_weight,
                text_weight=text_weight,
                projection=projection,
            )
            return {
                "results": json.loads(json.dumps(results, default=str)),
                "count": len(results),
                "query_info": {
                    "embedding_model": embedding_result.get("embedding_model") if isinstance(embedding_result, dict) else None,
                    "search_type": "hybrid ($rankFusion)",
                    "vector_weight": vector_weight,
                    "text_weight": text_weight,
                    "limit": limit,
                    "num_candidates": num_candidates,
                },
            }
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {"error": f"Error executing hybrid_search: {str(e)}"}

    return {
        "vector_search": vector_search,
        "text_search": text_search,
        "geospatial_search": geospatial_search,
        "hybrid_search": hybrid_search,
    }
