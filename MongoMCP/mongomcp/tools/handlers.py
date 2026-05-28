"""
Plain async handler functions for collection-backed query tools.

These are never registered directly with @mcp.tool(). register_query_tools()
wraps them under config-driven names so multiple tool entries can point at the
same underlying handler with different collection/index values.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Annotated

from bson import ObjectId
from pydantic import Field

try:
    # Available inside a FastMCP request context (MCP path). On the invoke_llm/API
    # path the token is passed explicitly, so import failure is non-fatal.
    from fastmcp.server.dependencies import get_access_token
except Exception:  # pragma: no cover - optional dependency path
    get_access_token = None

logger = logging.getLogger(__name__)

# Aggregation stages that write to a collection. Read-mode custom pipelines must
# never contain these; aggregate_write mode requires at least one.
_WRITE_STAGES = frozenset({"$merge", "$out"})

# Token placeholder pattern: {{ name }} — used to inject caller params into a
# stored pipeline/filter/update/document template.
_TOKEN_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _substitute_params(template: Any, params: Dict[str, Any]) -> Any:
    """Recursively substitute {{name}} placeholders in a config template.

    - A whole-value token ("{{count}}") is replaced with the param's native type
      (int/bool/list/dict preserved).
    - An embedded token ("prefix-{{id}}") does string interpolation.
    Raises KeyError if a referenced placeholder was not supplied.
    """
    if isinstance(template, dict):
        return {k: _substitute_params(v, params) for k, v in template.items()}
    if isinstance(template, list):
        return [_substitute_params(v, params) for v in template]
    if isinstance(template, str):
        whole = _TOKEN_RE.fullmatch(template.strip())
        if whole:
            key = whole.group(1)
            if key not in params:
                raise KeyError(f"missing pipeline parameter '{key}'")
            return params[key]

        def _repl(match: "re.Match") -> str:
            key = match.group(1)
            if key not in params:
                raise KeyError(f"missing pipeline parameter '{key}'")
            return str(params[key])

        return _TOKEN_RE.sub(_repl, template)
    return template


def _pipeline_write_stages(pipeline: List[Dict[str, Any]]) -> set:
    """Return the set of write stages ($merge/$out) present in a pipeline."""
    found = set()
    for stage in pipeline or []:
        if isinstance(stage, dict):
            found |= _WRITE_STAGES & set(stage.keys())
    return found


def _get_nested_field(doc: Any, path: str) -> Any:
    """Read a possibly dot-nested field (e.g. 'a.b.c') from a document dict.

    Returns None if any segment is missing or an intermediate value is not a dict.
    """
    cur = doc
    for part in str(path).split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _cast_object_ids(ids: List[Any]) -> List[Any]:
    """Cast a list of _id values for a $in match, preserving native types.

    A valid 24-char hex string is cast to ObjectId; an existing ObjectId (or any
    other value, e.g. a string _id that is not a hex ObjectId) is passed through
    unchanged. Mirrors the "match either" requirement — string _ids are NOT auto
    cast elsewhere, so callers passing ObjectId hex strings still match.
    """
    out: List[Any] = []
    for v in ids or []:
        if isinstance(v, ObjectId):
            out.append(v)
        elif isinstance(v, str):
            try:
                out.append(ObjectId(v))
            except Exception:
                out.append(v)
        else:
            out.append(v)
    return out


# The non-negotiable floor that rerank_ids and the multi_step `rerank` step ALWAYS return:
# the join key plus both scores. A caller/author `projection` only ADDS top-level fields on
# top of this floor — it can never drop _id, relevance_score, or score.
_RERANK_FLOOR_FIELDS = ["_id", "relevance_score", "score"]


def _project_doc(doc: Any, fields: List[str]) -> Any:
    """Return a shallow copy of `doc` keeping only the top-level keys in `fields`.

    Missing keys are simply absent; a non-dict input is returned unchanged.
    """
    if not isinstance(doc, dict):
        return doc
    return {k: doc[k] for k in fields if k in doc}


def _extract_scopes(token: Any) -> tuple:
    """Return (scopes:set, client_id:str) from a dict token, AccessToken, or the
    ambient FastMCP request token. Mirrors the logic in upsert_document."""
    scopes: set = set()
    client_id = ""
    if token is None and get_access_token is not None:
        try:
            token = get_access_token()
        except Exception:
            token = None
    if isinstance(token, dict):
        scopes = set(token.get("scope", []))
        client_id = token.get("agent_key", "")
    elif token is not None:
        scopes = set(getattr(token, "scopes", []) or [])
        client_id = getattr(token, "client_id", "")
    return scopes, client_id


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
                "results": results,
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

    async def vector_rerank_search(
        query_text: Annotated[str, Field(description="Natural language query. Used for vector retrieval and Voyage reranking.")],
        limit: Annotated[int, Field(default=10, description="Number of final results to return after reranking.", ge=1, le=50)] = 10,
        candidates: Annotated[int, Field(default=50, description="How many vector-search hits to retrieve before reranking.", ge=1, le=200)] = 50,
        num_candidates: Annotated[int, Field(default=200, description="Vector search ANN candidate pool size.", ge=10, le=1000)] = 200,
        filters: Annotated[Optional[List], Field(default=None, description="Optional list of filters to narrow results.")] = None,
        collection: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        index: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        vector_path: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        projection: Annotated[Optional[Dict], Field(default=None, description="Injected from tool config by middleware.")] = None,
        rerank_field: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware — document field (dot-path) whose text is reranked.")] = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not collection or not str(collection).strip():
                raise ValueError("handlers.vector_rerank_search:collection must be a non-empty string")
            if not query_text or not isinstance(query_text, str):
                return {"error": "handlers.vector_rerank_search:query_text must be a non-empty string"}
            if not rerank_field or not str(rerank_field).strip():
                return {"error": "handlers.vector_rerank_search:rerank_field must be set in the tool config"}

            embedding_result = await llm_client.generate_embedding(query_text)
            vector_qry = embedding_result.get("vector") if isinstance(embedding_result, dict) else embedding_result
            if not vector_qry or not isinstance(vector_qry, list):
                return {"error": f"Embedding generation returned unexpected format: {type(embedding_result)}"}

            # Stage 1 — vector retrieval (over-fetch candidates for the reranker).
            retrieve_n = max(candidates, limit)
            results = await mongo_server.vector_search(
                collection, vector_qry, filters, retrieve_n, num_candidates,
                index=index, vector_path=vector_path, projection=projection,
            )

            # Stage 2 — build parallel doc/text lists, skipping docs with no rerank text.
            candidate_docs: List[Any] = []
            documents: List[str] = []
            for doc in results:
                text = _get_nested_field(doc, rerank_field)
                if text is None or not str(text).strip():
                    continue
                candidate_docs.append(doc)
                documents.append(str(text))

            if not documents:
                return {
                    "results": results[:limit],
                    "count": min(len(results), limit),
                    "query_info": {
                        "search_type": "vector (rerank skipped — no rerank_field text found)",
                        "rerank_field": rerank_field,
                        "limit": limit,
                    },
                }

            # Stage 3 — Voyage rerank; map reranked order back to original documents.
            ranked = await llm_client.rerank(query_text, documents, top_k=limit)
            reranked_docs: List[Any] = []
            for item in ranked:
                idx = item.get("index")
                if idx is None or idx < 0 or idx >= len(candidate_docs):
                    continue
                doc = candidate_docs[idx]
                if isinstance(doc, dict):
                    doc = {**doc, "relevance_score": item.get("relevance_score")}
                reranked_docs.append(doc)

            return {
                "results": reranked_docs,
                "count": len(reranked_docs),
                "query_info": {
                    "embedding_model": embedding_result.get("embedding_model") if isinstance(embedding_result, dict) else None,
                    "search_type": "vector + Voyage rerank",
                    "rerank_field": rerank_field,
                    "candidates_retrieved": len(results),
                    "candidates_reranked": len(documents),
                    "limit": limit,
                    "num_candidates": num_candidates,
                },
            }
        except Exception as e:
            logger.error(f"Rerank search failed: {e}")
            return {"error": f"Error executing vector_rerank_search: {str(e)}"}

    async def rerank_documents(
        query_text: Annotated[str, Field(description="Natural language query. Candidates are reranked against this — no retrieval or embedding is performed.")],
        documents: Annotated[Optional[List[Dict]], Field(default=None, description="Array of candidate result documents to rerank directly (e.g. the output of a prior search step).")] = None,
        limit: Annotated[int, Field(default=10, description="Number of final results to return after reranking.", ge=1, le=50)] = 10,
        min_relevance: Annotated[float, Field(default=0.0, description="Optional floor on relevance_score; candidates scoring below this are dropped.", ge=0.0, le=1.0)] = 0.0,
        rerank_field: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware — document field (dot-path) whose text is reranked.")] = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not query_text or not isinstance(query_text, str):
                return {"error": "handlers.rerank_documents:query_text must be a non-empty string"}
            if not rerank_field or not str(rerank_field).strip():
                return {"error": "handlers.rerank_documents:rerank_field must be set in the tool config"}
            if documents is None or not isinstance(documents, list):
                return {"error": "handlers.rerank_documents:documents must be a non-empty list of result objects"}
            if not documents:
                return {"results": [], "count": 0, "query_info": {"search_type": "rerank (no documents supplied)", "rerank_field": rerank_field, "limit": limit}}

            # Build parallel doc/text lists, skipping docs with no rerank text.
            candidate_docs: List[Any] = []
            texts: List[str] = []
            for doc in documents:
                text = _get_nested_field(doc, rerank_field)
                if text is None or not str(text).strip():
                    continue
                candidate_docs.append(doc)
                texts.append(str(text))

            if not texts:
                return {
                    "results": documents[:limit],
                    "count": min(len(documents), limit),
                    "query_info": {
                        "search_type": "rerank (skipped — no rerank_field text found)",
                        "rerank_field": rerank_field,
                        "limit": limit,
                    },
                }

            # Voyage rerank; map reranked order back to original documents, applying floor.
            ranked = await llm_client.rerank(query_text, texts, top_k=limit)
            reranked_docs: List[Any] = []
            for item in ranked:
                idx = item.get("index")
                if idx is None or idx < 0 or idx >= len(candidate_docs):
                    continue
                score = item.get("relevance_score")
                if min_relevance and score is not None and score < min_relevance:
                    continue
                doc = candidate_docs[idx]
                if isinstance(doc, dict):
                    doc = {**doc, "relevance_score": score}
                reranked_docs.append(doc)

            return {
                "results": reranked_docs,
                "count": len(reranked_docs),
                "query_info": {
                    "search_type": "Voyage rerank (documents)",
                    "rerank_field": rerank_field,
                    "candidates_supplied": len(documents),
                    "candidates_reranked": len(texts),
                    "min_relevance": min_relevance,
                    "limit": limit,
                },
            }
        except Exception as e:
            logger.error(f"Rerank documents failed: {e}")
            return {"error": f"Error executing rerank_documents: {str(e)}"}

    async def rerank_ids(
        query_text: Annotated[str, Field(description="Natural language query. Candidates are reranked against this — no vector retrieval or embedding is performed.")],
        candidate_ids: Annotated[Optional[List], Field(default=None, description="Array of document _id values (ObjectId hex strings or ObjectIds) to fetch from the collection and rerank.")] = None,
        limit: Annotated[int, Field(default=10, description="Number of final results to return after reranking.", ge=1, le=50)] = 10,
        candidates: Annotated[int, Field(default=100, description="Max candidate ids to fetch/rerank; incoming ids are truncated to this cap to bound cost.", ge=1, le=500)] = 100,
        min_relevance: Annotated[float, Field(default=0.0, description="Optional floor on relevance_score; candidates scoring below this are dropped.", ge=0.0, le=1.0)] = 0.0,
        projection: Annotated[Optional[List[str]], Field(default=None, description="Extra top-level field names to return, IN ADDITION to the always-included floor (_id, relevance_score, score). Omit for floor-only — drops the large source body.")] = None,
        collection: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        rerank_field: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware — document field (dot-path) whose text is reranked.")] = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not collection or not str(collection).strip():
                raise ValueError("handlers.rerank_ids:collection must be a non-empty string")
            if not query_text or not isinstance(query_text, str):
                return {"error": "handlers.rerank_ids:query_text must be a non-empty string"}
            if not rerank_field or not str(rerank_field).strip():
                return {"error": "handlers.rerank_ids:rerank_field must be set in the tool config"}
            if candidate_ids is None or not isinstance(candidate_ids, list) or not candidate_ids:
                return {"error": "handlers.rerank_ids:candidate_ids must be a non-empty list of _id values"}

            # The floor (_id, relevance_score, score) is ALWAYS returned; a caller
            # `projection` only ADDS top-level fields on top of it.
            extra = list(projection) if projection else []
            keep_fields = list(dict.fromkeys(list(_RERANK_FLOOR_FIELDS) + extra))

            # Fetch the candidate docs by _id. Project down to the fields we return PLUS
            # rerank_field (needed transiently to feed the reranker) — this keeps the
            # DB->server payload small. relevance_score isn't a stored field, so it's
            # excluded from the $project (it's attached after reranking).
            ids = _cast_object_ids(candidate_ids[:candidates])
            fetch_proj = {f: 1 for f in keep_fields if f != "relevance_score"}
            fetch_proj[rerank_field] = 1
            docs = await mongo_server.agg_pipeline(
                collection, [{"$match": {"_id": {"$in": ids}}}, {"$project": fetch_proj}]
            )

            # Delegate the actual reranking to rerank_documents (option 2).
            result = await rerank_documents(
                query_text=query_text,
                documents=docs,
                limit=limit,
                min_relevance=min_relevance,
                rerank_field=rerank_field,
            )
            if isinstance(result, dict) and isinstance(result.get("results"), list):
                # Trim the source body (e.g. rerank_field) from the returned docs.
                result["results"] = [_project_doc(d, keep_fields) for d in result["results"]]
            if isinstance(result, dict) and "query_info" in result:
                result["query_info"]["source"] = "id_lookup"
                result["query_info"]["candidate_ids_supplied"] = len(candidate_ids)
                result["query_info"]["candidate_ids_fetched"] = len(docs)
                result["query_info"]["collection"] = collection
                result["query_info"]["projection"] = keep_fields
            return result
        except Exception as e:
            logger.error(f"Rerank ids failed: {e}")
            return {"error": f"Error executing rerank_ids: {str(e)}"}

    async def text_search(
        query_text: Annotated[str, Field(description="Keywords or phrases to search for.")],
        limit: Annotated[int, Field(default=10, description="Maximum number of results to return.", ge=1, le=100)] = 10,
        projection: Annotated[Optional[Dict], Field(default=None, description="Optional $project fields for the results.")] = None,
        collection: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not query_text:
                return {"error": "query_text is required"}
            results = await mongo_server.text_search(collection, query_text, limit, projection=projection)
            return {
                "results": results,
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
        projection: Annotated[Optional[Dict], Field(default=None, description="Optional $project fields for the results.")] = None,
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
                projection=projection,
            )
            return {
                "results": results,
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
                "results": results,
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

    async def custom_pipeline(
        params: Annotated[Optional[Dict], Field(default=None, description="Values for the stored procedure's declared placeholders (e.g. {\"status\": \"open\"}). Substituted into the configured template.")] = None,
        limit: Annotated[Optional[int], Field(default=None, description="Optional result limit for read operations.", ge=1, le=1000)] = None,
        collection: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        operation: Annotated[Optional[str], Field(default=None, description="Injected from tool config by middleware.")] = None,
        pipeline: Annotated[Optional[List], Field(default=None, description="Injected from tool config by middleware.")] = None,
        filter: Annotated[Optional[Dict], Field(default=None, description="Injected from tool config by middleware.")] = None,
        update: Annotated[Optional[Dict], Field(default=None, description="Injected from tool config by middleware.")] = None,
        document: Annotated[Optional[Any], Field(default=None, description="Injected from tool config by middleware.")] = None,
        upsert: Annotated[Optional[bool], Field(default=None, description="Injected from tool config by middleware.")] = None,
        multi: Annotated[Optional[bool], Field(default=None, description="Injected from tool config by middleware.")] = None,
        token: Any = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not collection or not str(collection).strip():
                return {"error": "handlers.custom_pipeline: collection must be provided in tool config"}
            op = (operation or "aggregate").lower()
            p = params if isinstance(params, dict) else {}

            # ---- Authorization: writes require scope; delete requires stricter scope ----
            read_ops = {"aggregate", "read", "find"}
            write_ops = {"aggregate_write", "update", "insert"}
            if op not in read_ops:
                scopes, client_id = _extract_scopes(token)
                needed = "delete" if op == "delete" else "write"
                if needed not in scopes:
                    logger.error(f"custom_pipeline: insufficient scope '{needed}' for op '{op}' (agent {client_id})")
                    return {"error": f"Insufficient scope: '{needed}' permission required for operation '{op}'."}

            # ---- Read / aggregate ----
            if op in read_ops:
                pl = _substitute_params(pipeline or [], p)
                if not isinstance(pl, list):
                    return {"error": "custom_pipeline: configured 'pipeline' must be a list of stages"}
                bad = _pipeline_write_stages(pl)
                if bad:
                    return {"error": f"custom_pipeline: write stage(s) {sorted(bad)} not allowed in read operation"}
                final_pipeline = list(pl)
                if limit is not None and not any("$limit" in s for s in final_pipeline if isinstance(s, dict)):
                    final_pipeline.append({"$limit": limit})
                results = await mongo_server.agg_pipeline(collection, final_pipeline)
                return {
                    "results": results,
                    "count": len(results),
                    "query_info": {"operation": op, "stages_count": len(final_pipeline), "limit_applied": limit},
                }

            # ---- Aggregate write ($merge / $out) ----
            if op == "aggregate_write":
                pl = _substitute_params(pipeline or [], p)
                if not isinstance(pl, list) or not pl:
                    return {"error": "custom_pipeline: aggregate_write requires a non-empty 'pipeline'"}
                if not _pipeline_write_stages(pl):
                    return {"error": "custom_pipeline: aggregate_write requires a terminal $merge or $out stage"}
                await mongo_server.agg_pipeline(collection, pl)
                return {"status": "ok", "query_info": {"operation": op, "stages_count": len(pl)}}

            # ---- Update (filter + update doc, upsert/multi) ----
            if op == "update":
                f = _substitute_params(filter or {}, p)
                u = _substitute_params(update or {}, p)
                if not isinstance(u, dict) or not u:
                    return {"error": "custom_pipeline: update requires a non-empty 'update' document"}
                res = await mongo_server.run_update(collection, f, u, upsert=bool(upsert), multi=bool(multi))
                return {"status": "ok", "result": res, "query_info": {"operation": op}}

            # ---- Insert ----
            if op == "insert":
                doc = _substitute_params(document, p)
                if not isinstance(doc, (dict, list)) or not doc:
                    return {"error": "custom_pipeline: insert requires a non-empty 'document' object or array"}
                res = await mongo_server.run_insert(collection, doc)
                return {"status": "ok", "result": res, "query_info": {"operation": op}}

            # ---- Delete ----
            if op == "delete":
                f = _substitute_params(filter or {}, p)
                if not isinstance(f, dict) or not f:
                    return {"error": "custom_pipeline: delete requires a non-empty 'filter' (refusing to delete all docs)"}
                res = await mongo_server.run_delete(collection, f, multi=bool(multi))
                return {"status": "ok", "result": res, "query_info": {"operation": op}}

            return {"error": f"custom_pipeline: unknown operation '{op}'"}
        except KeyError as e:
            return {"error": f"custom_pipeline parameter error: {str(e)}"}
        except Exception as e:
            logger.error(f"custom_pipeline failed: {e}")
            return {"error": f"Error executing custom_pipeline: {str(e)}"}

    return {
        "vector_search": vector_search,
        "rerank_search": vector_rerank_search,
        "vector_rerank_search": vector_rerank_search,
        "rerank_documents": rerank_documents,
        "rerank_ids": rerank_ids,
        "text_search": text_search,
        "geospatial_search": geospatial_search,
        "hybrid_search": hybrid_search,
        "custom_pipeline": custom_pipeline,
    }
