"""
MongoDB Vector Search MCP Server
A fastMCP MCP server that provides vector search capabilities using MongoDB's $search aggregation pipeline.
"""

import datetime
import asyncio
import traceback
from typing import Any, Dict, List, Optional, Tuple
import logging
from pymongo.errors import PyMongoError
from .mongodb_client import MongoDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBQueryServer(MongoDBClient):
    """
    MongoDB wrapper class translates MCP Server functions to MongoDB operations.
    """
    def __init__(self, settings):
        super().__init__(settings=settings)
        self.tool_config = None
        self.tool_name = None
        self.description = "MongoDB Vector Search MCP Server"
        self.available_collections = [self._collection_name]

    @staticmethod
    def _build_projection(projection: Optional[Dict], vector_path: Optional[str]) -> Optional[Dict]:
        """Return a projection dict that always excludes the vector_path field.

        - Inclusion projection (values=1): already excludes vector_path implicitly — returned as-is.
        - Exclusion projection (values=0) or None: the vector_path field is added with value 0.
        - Empty dict: treated as exclusion — returns {vector_path: 0}.
        """
        if not vector_path:
            return projection

        if projection:
            # Detect inclusion projection: at least one value is 1 (and it's not _id)
            non_id = {k: v for k, v in projection.items() if k != '_id'}
            if non_id and all(v == 1 for v in non_id.values()):
                # Inclusion — embedding not in list so already excluded
                return projection
            # Exclusion or mixed — add the vector path
            return {**projection, vector_path: 0}

        # No projection: exclude only the vector field
        return {vector_path: 0}

    def set_config(self, config: Dict) -> None:
        """Set the tool configuration from a dictionary. this overrides the default settings"""
        if config is None:
            raise ValueError("Config cannot be None. Check env variables and AWS secrets.")

        self.tool_config = config
        self.tool_name  = config["Name"]
        self.description = config["module_info"]["description"]
        print(f"Using settings from tool config {self.tool_name}")
        super().set_config(config["module_info"])

    async def get_mongo_info(self, shortResponse=False) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrieve MongoDB connection and collection health information.

        This function performs a health check on the MongoDB connection and gathers
        essential database statistics including connection status, collection metrics,
        and server information. It's designed to be used for monitoring, debugging,
        and health check endpoints.

        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple containing:
                - bool: Failed flag (True if any operation failed, False if all succeeded)
                - Dict: Health status dictionary containing:
                    - status: "healthy" or "unhealthy"
                    - version: MongoDB server version
                    - mongodb: Nested dict with connection details
                    - error: Error message (only present if operation failed)
        """
        failed = True
        health_status = {
            "status": "unhealthy",
            "toolname": self.tool_name,
            "description": self.description,
            "mongodb": {
                "database": self._db_name,
                "collections": {}
            }
        }
        ping_result = None
        try:
            # Ensure connection is established
            ping_result = await self.ensure_connection()

            if ping_result and ping_result.get("ok", 0) == 1.0:
                is_connected = True
            else:
                is_connected = False
                health_status["error"] = "MongoDB Connection failed"
                failed = True

            health_status["mongodb"]["connected"] = is_connected
            #health_status["mongodb"]["collections"] = self.available_collections

            if is_connected:
                # get all the user collections ignoring system collections
                all_collection_names = await self.db.list_collection_names()
                self.available_collections = [
                    name for name in all_collection_names
                    if not (name.startswith('system.') or name.startswith('_'))
                ]

                # Convert MongoDB Timestamp object to datetime
                cluster_time = ping_result["$clusterTime"]["clusterTime"]
                health_status["mongodb"]["timestamp"] = str(datetime.datetime.fromtimestamp(cluster_time.time).isoformat())
                health_status["status"] = "healthy"
                failed = False

                if not shortResponse:
                    # Get server info and collection stats
                    collection_stats = await asyncio.gather(
                        *[self.db.command("collStats", collection_name) for collection_name in self.available_collections]
                    )
                    # List all available collections in the database
                    for stats, collection_name in zip(collection_stats, self.available_collections):
                        health_status["mongodb"]["collections"][collection_name] = {
                            "document_count" : stats.get("count", 0),
                            "size_bytes" : stats.get("size", 0)
                        }


        except Exception as e:
            health_status["error"]= str(e)
            logger.error(f"Health check failed: {e}")
            traceback.print_exc()

        return (failed, health_status)

    async def get_collection_info(self) -> Dict[str, Any]:
        """Retrieve information about the current MongoDB collection."""
        failed, info = await self.get_mongo_info(shortResponse=False)
        if failed:
            raise ConnectionError(f"Failed to retrieve MongoDB info: {info.get('error','Unknown error')}")

        for coll in self.available_collections:
            indexes = []
            search_indexes = []
            logger.debug(f"Retrieving info for collection: {coll}")

            try:
                async for idx in self.get_collection(coll).list_indexes():
                    indexes.append(idx)

                async for sidx in self.get_collection(coll).list_search_indexes():
                    indx = sidx
                    # clean up vector index info for readability
                    indx.pop("statusDetail", None)
                    indx.pop("latestDefinitionVersion", None)
                    search_indexes.append(indx)
            except PyMongoError as e:
                # some collections don't allow index listing
                logger.error(f"Failed to retrieve indexes for collection {coll}")
                continue

            coll_info = {
                "indexes": [
                    {
                        "name": idx.get("name"),
                        "key": idx.get("key"),
                        "type": idx.get("type", "standard")
                    } for idx in indexes
                ]
            }

            # Only add search_indexes field if there are search indexes
            if search_indexes:
                coll_info["search_indexes"] = [
                    sidx for sidx in search_indexes
                ]

            # build the collection info structure
            if "collections" not in info["mongodb"]:
                info["mongodb"]["collections"] = {}
            if coll not in info["mongodb"]["collections"]:
                info["mongodb"]["collections"][coll] = {}
            info["mongodb"]["collections"][coll].update(coll_info)
        return info

    async def vector_search(self, collection: str, vector_qry: str, filters: list = None, limit: int = 10, num_candidates: int = 100, index: str = None, vector_path: str = None, projection: dict = None) -> List[Dict[str, Any]]:
        """
        Perform vector search using MongoDB's $search aggregation pipeline

        Args:
            vector_qry: The vectorized embeddings of the query string for similarity search
            limit: Maximum number of results to return
            num_candidates: Number of candidates to consider during search

        Returns:
            List of search results with similarity scores
        """

        try:
            if not collection or not str(collection).strip():
                raise ValueError("mongodb_query_provider.vector_search:collection must be a non-empty string")

            await self.ensure_connection()
            # Prefer caller-supplied values (injected by middleware from the named tool
            # config entry). Fall back to legacy tool_config['tools']['vector_search']
            # for backward-compatible single-tool deployments.
            _vs_cfg = self.tool_config.get('tools', {}).get('vector_search', {})
            _index = index or _vs_cfg.get('index')
            _vector_path = vector_path or _vs_cfg.get('vector_path')
            _projection = self._build_projection(
                projection or _vs_cfg.get('projection'),
                _vector_path,
            )
            if not _index or not _vector_path:
                raise ValueError(
                    f"vector_search: missing 'index' or 'vector_path' for collection '{collection}'. "
                    "Ensure the tool config entry has 'index' and 'vector_path' fields."
                )
            # MongoDB Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": _index,
                        "path": _vector_path,
                        "queryVector": vector_qry,
                        "numCandidates": num_candidates,
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$sort": {
                        "score": -1
                    }
                }
            ]

            if _projection:
                pipeline.insert(2, {"$project": _projection})

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

                # Inject the filter into the pipeline
                pipeline[0]["$vectorSearch"]["filter"] = match_filter
            results = []
            logger.debug(f"Vector search pipeline: {pipeline}")
            logger.debug(f"Vector search collection: {collection}")
            async for doc in self.get_collection(collection).aggregate(pipeline):
                results.append(doc)
            logger.debug(f"Vector search returned {len(results)} results")
            return results

        except PyMongoError as e:
            logger.error(f"Vector search failed: {e}")
            raise

    async def geospatial_search(
        self,
        collection: str,
        longitude: float,
        latitude: float,
        max_distance_meters: Optional[float] = None,
        min_distance_meters: Optional[float] = None,
        filters: list = None,
        limit: int = 10,
        geo_field: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform geospatial search using MongoDB's $geoNear aggregation stage.

        Args:
            collection: Name of the collection to search.
            longitude: Longitude of the search center point.
            latitude: Latitude of the search center point.
            max_distance_meters: Optional maximum distance in meters.
            min_distance_meters: Optional minimum distance in meters.
            filters: Optional list of filters in [field, value] format.
            limit: Maximum number of results to return.
            geo_field: GeoJSON Point field path with a 2dsphere index.

        Returns:
            List of search results with computed distance in meters.
        """
        try:
            await self.ensure_connection()

            if not (-180 <= longitude <= 180):
                raise ValueError("longitude must be between -180 and 180")
            if not (-90 <= latitude <= 90):
                raise ValueError("latitude must be between -90 and 90")
            if limit < 1:
                raise ValueError("limit must be at least 1")
            if max_distance_meters is not None and max_distance_meters < 0:
                raise ValueError("max_distance_meters must be >= 0")
            if min_distance_meters is not None and min_distance_meters < 0:
                raise ValueError("min_distance_meters must be >= 0")
            if (
                min_distance_meters is not None
                and max_distance_meters is not None
                and min_distance_meters > max_distance_meters
            ):
                raise ValueError("min_distance_meters cannot be greater than max_distance_meters")

            geo_config = {}
            if self.tool_config:
                geo_config = self.tool_config.get("tools", {}).get("geospatial_search", {})

            resolved_geo_field = geo_field or geo_config.get("location_field")

            geo_near_stage = {
                "$geoNear": {
                    "near": {
                        "type": "Point",
                        "coordinates": [longitude, latitude],
                    },
                    "distanceField": "distance_meters",
                    "spherical": True,
                    "key": resolved_geo_field,
                }
            }

            if max_distance_meters is not None:
                geo_near_stage["$geoNear"]["maxDistance"] = max_distance_meters
            if min_distance_meters is not None:
                geo_near_stage["$geoNear"]["minDistance"] = min_distance_meters

            if filters:
                match_filter = {}
                if len(filters) > 1:
                    match_filter = {"$and": []}
                    for key, value in filters:
                        match_filter["$and"].append({key: value})
                else:
                    key, value = filters[0]
                    match_filter[key] = value

                geo_near_stage["$geoNear"]["query"] = match_filter

            pipeline = [
                geo_near_stage,
                {
                    "$sort": {
                        "distance_meters": 1,
                    }
                },
                {
                    "$limit": limit,
                },
            ]

            projection = geo_config.get("projection")
            if projection:
                pipeline.append({"$project": projection})

            results = []
            async for doc in self.get_collection(collection).aggregate(pipeline):
                results.append(doc)
            return results

        except PyMongoError as e:
            logger.error(f"Geospatial search failed: {e}")
            raise

    async def text_search(self, collection: str, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform text search using MongoDB's $search aggregation pipeline

        Args:
            query_text: The text query for search
            limit: Maximum number of results to return

        Returns:
            List of search results with relevance scores
        """
        try:
            await self.ensure_connection()
            # MongoDB Atlas Text Search aggregation pipeline
            pipeline = [
                {
                    "$search": {
                        "index": self.tool_config['tools']['text_search']['index'],
                        "text": {
                            "query": query_text,
                            "path": self.tool_config['tools']['text_search']['fields_searched']
                        }
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "searchScore"}
                    }
                },
                {
                    "$limit": limit
                },
                {
                    "$project": self.tool_config['tools']['text_search']['projection']
                },
                {
                    "$sort": {
                        "score": -1
                    }
                }
            ]

            results = []
            async for doc in self.get_collection(collection).aggregate(pipeline):
                results.append(doc)
            logger.info(f"Text search returned {len(results)} results")
            return results

        except PyMongoError as e:
            logger.error(f"Text search failed: {e}")
            raise

    async def hybrid_search(
        self,
        collection: str,
        vector_qry: List[float],
        query_text: str,
        limit: int = 10,
        num_candidates: int = 100,
        filters: list = None,
        vector_index: str = None,
        text_index: str = None,
        vector_path: str = None,
        text_fields: Optional[List[str]] = None,
        vector_weight: float = 0.6,
        text_weight: float = 0.4,
        projection: dict = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and full-text BM25
        using MongoDB Atlas $rankFusion in a single aggregation pipeline.

        Returns results with score_details showing the fusion score, so the
        audience can see both retrieval signals contributing to ranking.
        """
        try:
            if not collection or not str(collection).strip():
                raise ValueError("hybrid_search: collection must be a non-empty string")

            await self.ensure_connection()

            _hs_cfg = self.tool_config.get("tools", {}).get("hybrid_search", {})
            _vector_index = vector_index or _hs_cfg.get("vector_index")
            _text_index   = text_index   or _hs_cfg.get("text_index")
            _vector_path  = vector_path  or _hs_cfg.get("vector_path")
            _text_fields  = text_fields  or _hs_cfg.get("text_fields") or ["description"]
            _projection   = self._build_projection(
                projection or _hs_cfg.get("projection"),
                _vector_path,
            )

            if not _vector_index or not _vector_path:
                raise ValueError(
                    f"hybrid_search: missing 'vector_index' or 'vector_path' for collection '{collection}'. "
                    "Ensure the tool config has these fields."
                )
            if not _text_index:
                raise ValueError(
                    f"hybrid_search: missing 'text_index' for collection '{collection}'. "
                    "An Atlas Search full-text index is required for $rankFusion."
                )

            # $rankFusion does not accept pre-filters at the pipeline level.
            # Filters are applied as a $match stage AFTER fusion scoring,
            # which correctly narrows the fused result set.
            post_filter: Optional[Dict] = None
            if filters:
                if len(filters) > 1:
                    post_filter = {"$and": [{k: v} for k, v in filters]}
                else:
                    post_filter = {filters[0][0]: filters[0][1]}

            vector_pipeline: List[Dict] = [
                {
                    "$vectorSearch": {
                        "index": _vector_index,
                        "path": _vector_path,
                        "queryVector": vector_qry,
                        "numCandidates": num_candidates,
                        "limit": limit * 5,
                    }
                }
            ]

            text_pipeline: List[Dict] = [
                {
                    "$search": {
                        "index": _text_index,
                        "text": {"query": query_text, "path": _text_fields},
                    }
                },
                {"$limit": limit * 5},
            ]

            pipeline: List[Dict] = [
                {
                    "$rankFusion": {
                        "input": {
                            "pipelines": {
                                "vector": vector_pipeline,
                                "text": text_pipeline,
                            }
                        },
                        "combination": {
                            "weights": {
                                "vector": vector_weight,
                                "text": text_weight,
                            }
                        },
                        "scoreDetails": True,
                    }
                },
                {
                    "$addFields": {
                        "score_details": {
                            "fusion_score": {"$meta": "searchScore"},
                            "vector_weight": vector_weight,
                            "text_weight": text_weight,
                        }
                    }
                },
            ]

            if post_filter:
                pipeline.append({"$match": post_filter})

            pipeline.append({"$limit": limit})

            # Apply inclusion projections only — exclusion projections conflict with
            # score_details (inclusion field) in the same $project stage.
            # Instead, strip excluded fields (like the embedding) from results in Python.
            inclusion_proj: Optional[Dict] = None
            fields_to_strip: List[str] = []

            if _projection:
                non_id = {k: v for k, v in _projection.items() if k != "_id"}
                if non_id and all(v == 1 for v in non_id.values()):
                    # Inclusion projection — safe to use as $project, add score_details.
                    inclusion_proj = {**_projection, "score_details": 1}
                else:
                    # Exclusion projection — collect fields to strip in Python instead.
                    fields_to_strip = [k for k, v in _projection.items() if v == 0]
            # Always strip the embedding vector field from results.
            if _vector_path and _vector_path not in fields_to_strip:
                fields_to_strip.append(_vector_path)

            if inclusion_proj:
                pipeline.append({"$project": inclusion_proj})

            results: List[Dict] = []
            async for doc in self.get_collection(collection).aggregate(pipeline):
                for field in fields_to_strip:
                    doc.pop(field, None)
                results.append(doc)
            return results

        except PyMongoError as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    async def agg_pipeline(self, collection: str, pipeline: List[Dict]) -> List[Dict[str, Any]]:
        """
        Perform MongoDB's aggregation pipeline

        Args:
            pipeline: The pipeline to execute (list of aggregation stages)

        Returns:
            List of results
        """
        try:
            await self.ensure_connection()
            # MongoDB Atlas aggregation pipeline
            results = []
            async for doc in self.get_collection(collection).aggregate(pipeline):
                results.append(doc)
            return results

        except PyMongoError as e:
            logger.error(f"pipeline query failed: {e}")
            raise
