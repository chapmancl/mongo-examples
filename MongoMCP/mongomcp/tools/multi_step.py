"""
mongomcp.tools.multi_step
=========================
Multi-step read-only query functions.

A "function" is a stored, config-driven chain of read-only query steps. Each step
runs a query (vector search, geospatial search, or aggregation); values extracted
from its results are bound to named variables that later steps reference via
whole-value ``{{token}}`` placeholders. This lets one AI request run e.g. "vector
search -> collect _ids -> second query keyed on those _ids" without the model
round-tripping between calls.

Key invariant — NATIVE TYPES BETWEEN STEPS:
    Steps run against ``mongo_server`` methods that return raw BSON. An ``_id`` stays
    an ``ObjectId`` while it is extracted from step N and injected into step N+1's
    ``$in`` filter, so the match actually works. Serialization (default=str) happens
    exactly once, on the final output step.

Read-only only: every step is validated to contain no write stages
($merge/$out) and no dangerous operators ($function/$where/$accumulator). There is
no write path here at all.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .handlers import _substitute_params, _pipeline_write_stages, _get_nested_field, _cast_object_ids, _project_doc, _RERANK_FLOOR_FIELDS, _TOKEN_RE

logger = logging.getLogger(__name__)

# Operators that can execute arbitrary code or write; never allowed in a function step.
_FORBIDDEN_OPERATORS = frozenset({"$function", "$where", "$accumulator", "$out", "$merge"})

# Step handlers supported in v1. Mirror the pinned search handlers in handlers.py
# (everything EXCEPT custom_pipeline — the 'aggregate' step already covers read-only
# pipeline logic, and custom_pipeline's write modes have no place in a read-only chain).
_SUPPORTED_USES = frozenset({
    "vector_search", "rerank_search", "vector_rerank_search", "rerank",
    "text_search", "geospatial_search", "hybrid_search", "aggregate",
})

# Caps to keep a broad early step from blowing up a later $in / result set.
_MAX_EXTRACT = 1000
_DEFAULT_STEP_LIMIT = 1000


def _resolve_int(value: Any, binds: Dict[str, Any], default: int) -> int:
    """Resolve a numeric step field (limit / num_candidates) to a native int.

    Handles the case where the field is a whole-value ``{{token}}`` bound to a
    declared parameter: _substitute_params already returns the param's native value
    for a whole-value token, but a caller (or the LLM) may still supply a declared
    ``type:"integer"`` param as a JSON string ("50"). $vectorSearch rejects a string
    limit/numCandidates (Location65160 / 8575100), so we coerce to int here as a
    belt-and-suspenders guard. Falls back to *default* on a missing token or an
    un-coercible value rather than failing the whole step.
    """
    try:
        resolved = _substitute_params(value, binds)
    except KeyError:
        return default
    if resolved is None or isinstance(resolved, bool):
        return default
    try:
        return int(resolved)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------
# Static analysis helpers (used by both define-time validation and docs)
# ----------------------------------------------------------------------------

def _collect_tokens(obj: Any, acc: Set[str]) -> None:
    """Recursively collect every ``{{name}}`` placeholder found in a structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_tokens(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_tokens(v, acc)
    elif isinstance(obj, str):
        for m in _TOKEN_RE.finditer(obj):
            acc.add(m.group(1))


def _collect_operators(obj: Any, acc: Set[str]) -> None:
    """Recursively collect every MongoDB operator ($-prefixed key) in a pipeline."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("$"):
                acc.add(k)
            _collect_operators(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_operators(v, acc)


def validate_function_config(cfg: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """Statically validate a function config (steps chain).

    Checks, per step in order:
      - ``uses`` is supported;
      - every ``{{token}}`` resolves to a declared parameter or a bind produced by an
        EARLIER step's ``extract`` (guarantees no unresolved token at run time);
      - no forbidden operators and no write stages in aggregate steps;
      - vector_search steps declare index + vector_path.

    Returns (errors, info). ``info`` carries the produced-bind names for diagnostics.
    Does NOT execute anything — safe to run cross-database.
    """
    errors: List[str] = []
    steps = cfg.get("steps")
    if not isinstance(steps, list) or not steps:
        return (["multi_step config must have a non-empty 'steps' list"], {})

    declared: Set[str] = set((cfg.get("parameters") or {}).keys())
    produced: Set[str] = set()
    names: List[str] = []

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step {i} must be an object")
            continue
        name = step.get("name") or f"step{i}"
        names.append(name)
        uses = step.get("uses", "aggregate")
        if uses not in _SUPPORTED_USES:
            errors.append(f"step '{name}': unsupported 'uses' {uses!r} (allowed: {sorted(_SUPPORTED_USES)})")

        # Token resolvability — tokens may only reference params or earlier extracts.
        toks: Set[str] = set()
        _collect_tokens(step.get("pipeline"), toks)
        _collect_tokens(step.get("query_text"), toks)
        _collect_tokens(step.get("filters"), toks)
        _collect_tokens(step.get("longitude"), toks)
        _collect_tokens(step.get("latitude"), toks)
        _collect_tokens(step.get("max_distance_meters"), toks)
        _collect_tokens(step.get("min_distance_meters"), toks)
        _collect_tokens(step.get("candidate_ids"), toks)
        _collect_tokens(step.get("documents"), toks)
        unresolved = toks - declared - produced
        if unresolved:
            errors.append(
                f"step '{name}': unresolved placeholders {sorted(unresolved)} "
                f"(not a declared parameter or an earlier step's extract)"
            )

        if uses == "aggregate":
            pipeline = step.get("pipeline")
            if not isinstance(pipeline, list) or not pipeline:
                errors.append(f"step '{name}': aggregate step requires a non-empty 'pipeline' list")
            else:
                ops: Set[str] = set()
                _collect_operators(pipeline, ops)
                forbidden = ops & _FORBIDDEN_OPERATORS
                if forbidden:
                    errors.append(f"step '{name}': forbidden operator(s) {sorted(forbidden)} not allowed")
        elif uses == "vector_search":
            if not step.get("index") or not step.get("vector_path"):
                errors.append(f"step '{name}': vector_search step requires 'index' and 'vector_path'")
        elif uses in ("rerank_search", "vector_rerank_search"):
            if not step.get("index") or not step.get("vector_path"):
                errors.append(f"step '{name}': {uses} step requires 'index' and 'vector_path'")
            if not step.get("rerank_field"):
                errors.append(f"step '{name}': {uses} step requires 'rerank_field'")
        elif uses == "rerank":
            # Decoupled rerank: reranks a candidate set produced by an EARLIER step.
            if not step.get("rerank_field"):
                errors.append(f"step '{name}': rerank step requires 'rerank_field'")
            if not step.get("documents") and not step.get("candidate_ids"):
                errors.append(
                    f"step '{name}': rerank step requires 'documents' or 'candidate_ids' "
                    f"(a whole-value {{{{token}}}} bound to an earlier step's extract)"
                )
        elif uses == "hybrid_search":
            if not step.get("vector_index") or not step.get("text_index") or not step.get("vector_path"):
                errors.append(f"step '{name}': hybrid_search step requires 'vector_index', 'text_index' and 'vector_path'")
        elif uses == "geospatial_search":
            if not step.get("geo_field"):
                errors.append(f"step '{name}': geospatial_search step requires 'geo_field'")
            if step.get("longitude") is None or step.get("latitude") is None:
                errors.append(f"step '{name}': geospatial_search step requires 'longitude' and 'latitude'")

        # Register this step's extracts for later steps.
        extract = step.get("extract") or {}
        if not isinstance(extract, dict):
            errors.append(f"step '{name}': 'extract' must be an object of {{bind_name: rule}}")
        else:
            produced |= set(extract.keys())

    output = cfg.get("output")
    if output and output not in names:
        errors.append(f"'output' {output!r} does not name any step (steps: {names})")

    return (errors, {"produced_binds": sorted(produced), "step_names": names})


# ----------------------------------------------------------------------------
# Runtime extraction + execution
# ----------------------------------------------------------------------------

def apply_extract(rule: Dict[str, Any], results: List[Dict[str, Any]]) -> Any:
    """Pull a value out of a step's raw results into a bind, preserving native types.

    rule = {"field": "<dot.path>", "mode": "list"|"set"|"scalar"|"docs"}
      list   (default) — every row's field value, in order (capped at _MAX_EXTRACT).
      set    — deduped (order-preserving), capped.
      scalar — first row's field value (or None).
      docs   — the whole rows (full documents), capped. Feeds a downstream rerank
               step's 'documents' input so the upstream projection + score carry
               through and rerank just appends relevance_score (no re-fetch).
    Rows missing the field are skipped.
    """
    field = rule.get("field", "_id")
    mode = rule.get("mode", "list")
    if mode == "docs":
        return results[:_MAX_EXTRACT]
    values: List[Any] = []
    for row in results:
        v = _get_nested_field(row, field)
        if v is not None:
            values.append(v)
    if mode == "scalar":
        return values[0] if values else None
    if mode == "set":
        seen: Set[str] = set()
        deduped: List[Any] = []
        for v in values:
            key = str(v)
            if key not in seen:
                seen.add(key)
                deduped.append(v)
        return deduped[:_MAX_EXTRACT]
    return values[:_MAX_EXTRACT]


def _resolve_step_filters(raw_filters: Any, binds: Dict[str, Any]) -> Optional[List]:
    """Substitute params into a step's [field, value] filter list, then DROP any pair whose
    value is null or an empty/whitespace string.

    This makes an OMITTED optional param a no-op: when {{param}} defaults to "" or null the
    pair would otherwise become an exclusionary ``["field", ""]`` filter that matches nothing.
    Falsy-but-valid values (0, False) are preserved — only None and empty/whitespace strings
    are pruned. Returns None when nothing remains (so no filter is applied).
    """
    if not raw_filters:
        return None
    resolved = _substitute_params(raw_filters, binds)
    if not isinstance(resolved, list):
        return resolved
    pruned: List[Any] = []
    for pair in resolved:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            _field, value = pair
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            pruned.append([_field, value])
        else:
            pruned.append(pair)
    return pruned or None


def build_multi_step_handler(mongo_server, llm_client):
    """Return an async ``multi_step`` handler capturing mongo_server + llm_client.

    Registered like the other pinned handlers. ``steps`` and ``output`` are injected
    authoritatively from the tool config by the middleware; ``params`` carries the
    caller-supplied declared parameters.
    """

    async def _run_step(step: Dict[str, Any], binds: Dict[str, Any]) -> List[Dict[str, Any]]:
        uses = step.get("uses", "aggregate")
        collection = step.get("collection")
        if not collection or not str(collection).strip():
            raise ValueError(f"multi_step: step '{step.get('name')}' has no 'collection'")

        if uses == "vector_search":
            query_text = _substitute_params(step.get("query_text", ""), binds)
            if not query_text or not isinstance(query_text, str):
                raise ValueError(f"multi_step: step '{step.get('name')}' vector_search needs a 'query_text'")
            embedding_result = await llm_client.generate_embedding(query_text)
            vector_qry = embedding_result.get("vector") if isinstance(embedding_result, dict) else embedding_result
            if not vector_qry or not isinstance(vector_qry, list):
                raise ValueError("multi_step: embedding generation returned unexpected format")
            filters = _resolve_step_filters(step.get("filters"), binds)
            return await mongo_server.vector_search(
                collection, vector_qry, filters,
                _resolve_int(step.get("limit", 10), binds, 10),
                _resolve_int(step.get("num_candidates", 100), binds, 100),
                index=step.get("index"), vector_path=step.get("vector_path"),
                projection=step.get("projection"),
            )

        if uses == "geospatial_search":
            geo_field = step.get("geo_field")
            if not geo_field:
                raise ValueError(f"multi_step: step '{step.get('name')}' geospatial_search needs a 'geo_field'")

            def _num(v: Any) -> Optional[float]:
                v = _substitute_params(v, binds)
                return float(v) if v is not None else None

            longitude = _num(step.get("longitude"))
            latitude = _num(step.get("latitude"))
            if longitude is None or latitude is None:
                raise ValueError(
                    f"multi_step: step '{step.get('name')}' geospatial_search needs 'longitude' and 'latitude'"
                )
            filters = _resolve_step_filters(step.get("filters"), binds)
            return await mongo_server.geospatial_search(
                collection=collection,
                longitude=longitude,
                latitude=latitude,
                max_distance_meters=_num(step.get("max_distance_meters")),
                min_distance_meters=_num(step.get("min_distance_meters")),
                filters=filters,
                limit=_resolve_int(step.get("limit", 10), binds, 10),
                geo_field=geo_field,
            )

        if uses == "text_search":
            query_text = _substitute_params(step.get("query_text", ""), binds)
            if not query_text or not isinstance(query_text, str):
                raise ValueError(f"multi_step: step '{step.get('name')}' text_search needs a 'query_text'")
            return await mongo_server.text_search(
                collection, query_text, _resolve_int(step.get("limit", 10), binds, 10),
            )

        if uses == "hybrid_search":
            query_text = _substitute_params(step.get("query_text", ""), binds)
            if not query_text or not isinstance(query_text, str):
                raise ValueError(f"multi_step: step '{step.get('name')}' hybrid_search needs a 'query_text'")
            embedding_result = await llm_client.generate_embedding(query_text)
            vector_qry = embedding_result.get("vector") if isinstance(embedding_result, dict) else embedding_result
            if not vector_qry or not isinstance(vector_qry, list):
                raise ValueError("multi_step: embedding generation returned unexpected format")
            filters = _resolve_step_filters(step.get("filters"), binds)
            return await mongo_server.hybrid_search(
                collection=collection,
                vector_qry=vector_qry,
                query_text=query_text,
                limit=_resolve_int(step.get("limit", 10), binds, 10),
                num_candidates=_resolve_int(step.get("num_candidates", 100), binds, 100),
                filters=filters,
                vector_index=step.get("vector_index"),
                text_index=step.get("text_index"),
                vector_path=step.get("vector_path"),
                text_fields=step.get("text_fields"),
                vector_weight=step.get("vector_weight", 0.6),
                text_weight=step.get("text_weight", 0.4),
                projection=step.get("projection"),
            )

        if uses in ("rerank_search", "vector_rerank_search"):
            query_text = _substitute_params(step.get("query_text", ""), binds)
            if not query_text or not isinstance(query_text, str):
                raise ValueError(f"multi_step: step '{step.get('name')}' {uses} needs a 'query_text'")
            rerank_field = step.get("rerank_field")
            if not rerank_field or not str(rerank_field).strip():
                raise ValueError(f"multi_step: step '{step.get('name')}' rerank_search needs a 'rerank_field'")
            embedding_result = await llm_client.generate_embedding(query_text)
            vector_qry = embedding_result.get("vector") if isinstance(embedding_result, dict) else embedding_result
            if not vector_qry or not isinstance(vector_qry, list):
                raise ValueError("multi_step: embedding generation returned unexpected format")
            filters = _resolve_step_filters(step.get("filters"), binds)
            limit = _resolve_int(step.get("limit", 10), binds, 10)
            candidates = _resolve_int(step.get("candidates", 50), binds, 50)
            retrieve_n = max(candidates, limit)
            results = await mongo_server.vector_search(
                collection, vector_qry, filters, retrieve_n,
                _resolve_int(step.get("num_candidates", 200), binds, 200),
                index=step.get("index"), vector_path=step.get("vector_path"),
                projection=step.get("projection"),
            )
            # Native BSON preserved: reranked docs are the same raw docs, reordered with a
            # relevance_score added — so _id stays an ObjectId for downstream $in chaining.
            candidate_docs: List[Any] = []
            documents: List[str] = []
            for doc in results:
                text = _get_nested_field(doc, rerank_field)
                if text is None or not str(text).strip():
                    continue
                candidate_docs.append(doc)
                documents.append(str(text))
            if not documents:
                return results[:limit]
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
            return reranked_docs

        if uses == "rerank":
            # Decoupled rerank (Design B): rescore a candidate set produced by an EARLIER
            # step. Feed it either a 'documents' array (bound from that step's full results
            # via a mode:'docs' extract) or a 'candidate_ids' _id list. No retrieval or
            # embedding happens here — recall was already established upstream.
            query_text = _substitute_params(step.get("query_text", ""), binds)
            if not query_text or not isinstance(query_text, str):
                raise ValueError(f"multi_step: step '{step.get('name')}' rerank needs a 'query_text'")
            rerank_field = step.get("rerank_field")
            if not rerank_field or not str(rerank_field).strip():
                raise ValueError(f"multi_step: step '{step.get('name')}' rerank needs a 'rerank_field'")
            limit = _resolve_int(step.get("limit", 10), binds, 10)
            candidates = _resolve_int(step.get("candidates", 100), binds, 100)
            min_relevance = step.get("min_relevance", 0.0)
            # Output shape: the floor (_id, relevance_score, score) is ALWAYS returned; an
            # optional step 'projection' (list of top-level field names) ADDS fields on top.
            step_projection = step.get("projection")
            extra = list(step_projection) if isinstance(step_projection, list) else []
            keep_fields = list(dict.fromkeys(list(_RERANK_FLOOR_FIELDS) + extra))
            # Candidate source. Preferred: a 'documents' array bound from an earlier step's
            # full results (mode:'docs' extract) — the upstream projection + retrieval score
            # pass straight through; rerank only appends relevance_score (no re-fetch). Those
            # docs are trimmed ONLY if the author set a projection (else preserved as-is).
            # Fallback: a 'candidate_ids' _id list re-fetched from the collection, projected
            # to keep_fields + rerank_field so the large source body / embedding stays out.
            raw_docs = _substitute_params(step.get("documents"), binds) if step.get("documents") is not None else None
            if isinstance(raw_docs, list) and raw_docs:
                results = raw_docs[:candidates]
                trim_fields = keep_fields if step_projection else None
            else:
                raw_ids = _substitute_params(step.get("candidate_ids"), binds) if step.get("candidate_ids") is not None else None
                if not isinstance(raw_ids, list) or not raw_ids:
                    raise ValueError(
                        f"multi_step: step '{step.get('name')}' rerank needs a non-empty "
                        f"'documents' or 'candidate_ids' (bound from an earlier step's extract)"
                    )
                # Native BSON preserved: docs are fetched by their native/cast _id and
                # carried through with relevance_score added, so _id stays usable downstream.
                ids = _cast_object_ids(raw_ids[:candidates])
                fetch_proj = {f: 1 for f in keep_fields if f != "relevance_score"}
                fetch_proj[rerank_field] = 1
                results = await mongo_server.agg_pipeline(
                    collection, [{"$match": {"_id": {"$in": ids}}}, {"$project": fetch_proj}]
                )
                trim_fields = keep_fields
            candidate_docs = []
            documents = []
            for doc in results:
                text = _get_nested_field(doc, rerank_field)
                if text is None or not str(text).strip():
                    continue
                candidate_docs.append(doc)
                documents.append(str(text))
            if not documents:
                out = results[:limit]
                return [_project_doc(d, trim_fields) for d in out] if trim_fields else out
            ranked = await llm_client.rerank(query_text, documents, top_k=limit)
            reranked_docs = []
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
                    if trim_fields:
                        doc = _project_doc(doc, trim_fields)
                reranked_docs.append(doc)
            return reranked_docs

        # ---- aggregate (read-only) ----
        pipeline = _substitute_params(step.get("pipeline") or [], binds)
        if not isinstance(pipeline, list) or not pipeline:
            raise ValueError(f"multi_step: step '{step.get('name')}' aggregate needs a non-empty 'pipeline'")
        bad_stages = _pipeline_write_stages(pipeline)
        if bad_stages:
            raise ValueError(f"multi_step: step '{step.get('name')}' has write stage(s) {sorted(bad_stages)}")
        ops: Set[str] = set()
        _collect_operators(pipeline, ops)
        forbidden = ops & _FORBIDDEN_OPERATORS
        if forbidden:
            raise ValueError(f"multi_step: step '{step.get('name')}' forbidden operator(s) {sorted(forbidden)}")
        if not any(isinstance(s, dict) and "$limit" in s for s in pipeline):
            pipeline = [*pipeline, {"$limit": _resolve_int(step.get("limit", _DEFAULT_STEP_LIMIT), binds, _DEFAULT_STEP_LIMIT)}]
        return await mongo_server.agg_pipeline(collection, pipeline)

    async def multi_step(
        params: Optional[Dict[str, Any]] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        output: Optional[str] = None,
        collection: Optional[str] = None,  # optional default; steps carry their own
        token: Any = None,
    ) -> Dict[str, Any]:
        """Dynamic docstring loaded from JSON configuration"""
        try:
            if not steps or not isinstance(steps, list):
                return {"error": "multi_step: no 'steps' configured"}
            binds: Dict[str, Any] = dict(params or {})
            step_results: Dict[str, List[Dict[str, Any]]] = {}
            last: List[Dict[str, Any]] = []
            for i, step in enumerate(steps):
                # Apply an optional default collection to steps that omit one.
                if not step.get("collection") and collection:
                    step = {**step, "collection": collection}
                name = step.get("name") or f"step{i}"
                results = await _run_step(step, binds)
                step_results[name] = results
                last = results
                for bind_name, rule in (step.get("extract") or {}).items():
                    if isinstance(rule, dict):
                        binds[bind_name] = apply_extract(rule, results)
            final = step_results.get(output, last) if output else last
            return {
                "results": final,
                "count": len(final),
                "query_info": {
                    "steps": [s.get("name") or f"step{i}" for i, s in enumerate(steps)],
                    "output": output,
                },
            }
        except ValueError as e:
            return {"error": str(e)}
        except KeyError as e:
            return {"error": f"multi_step parameter error: {str(e)}"}
        except Exception as e:
            logger.error(f"multi_step failed: {e}")
            return {"error": f"Error executing multi_step: {str(e)}"}

    return multi_step
