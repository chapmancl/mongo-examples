"""
mongomcp.agent.function_builder
===============================
Universal tools that let an agent build, version, and publish **read-only** query
functions into any data domain's ``mcp_tools`` document.

Two MCP tools, registered on the always-on agent domain (surface as
``agent_define_query_function`` / ``agent_promote_query_function`` /
``agent_delete_query_function``):

- ``define_query_function`` (scope ``fn_define``) — validate + persist a multi-step
  read-only function as ``fn_<name>`` in ``mcp_tools[Name=<tool_name>]``. Auto-active
  in the ``dev`` stage. Versioned in the memory system (``strategy:query_function``).
- ``promote_query_function`` (scope ``fn_publish``) — move a function ONE step along the
  linear lifecycle ``prod ↔ dev ↔ disabled`` (no direct prod↔disabled jumps).
- ``delete_query_function`` (scope ``fn_define``) — remove a function from ``mcp_tools``.
  Only ``disabled``-stage functions may be deleted (lifecycle: prod → dev → disabled → delete).

Containment (there is no per-container fence — any container can write any domain):
  scope gate · tool_name must be an active domain · step collections must belong to
  that domain · read-only validation (multi_step) · fn_ namespace + no shadowing.
Index tuning is a best-effort static check that never blocks a definition.
"""

import datetime
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Annotated

from pydantic import Field
from fastapi import Depends
from fastmcp.server.dependencies import get_access_token, AccessToken, get_http_request

from ..tools.multi_step import validate_function_config
from ..memory.memservice import MemoryService
from ..mongodb_client import MongoDBClient
from .mcp_tools import build_mcp_http_call

logger = logging.getLogger(__name__)

_VALID_STAGES = ("disabled", "dev", "prod")

# Linear lifecycle: a function moves exactly ONE step at a time along this chain, in
# either direction. Promote up (disabled → dev → prod) or demote down (prod → dev →
# disabled); direct prod↔disabled jumps are forbidden. Deletion is only allowed from
# the terminal 'disabled' stage (prod → dev → disabled → delete).
_STAGE_CHAIN = ("prod", "dev", "disabled")


def _extract_scopes(token: Any) -> tuple:
    """Return (scopes:set, agent_id:str) from an AccessToken or dict token."""
    scopes: Set[str] = set()
    agent_id = ""
    if token is None:
        try:
            token = get_access_token()
        except Exception:
            token = None
    if isinstance(token, dict):
        scopes = set(token.get("scope", []) or [])
        agent_id = token.get("agent_name") or token.get("agent_key", "")
    elif token is not None:
        scopes = set(getattr(token, "scopes", []) or [])
        claims = getattr(token, "claims", {}) or {}
        agent_id = claims.get("agent_name") or getattr(token, "client_id", "")
    return scopes, agent_id


def _fn_key(name: str) -> str:
    """Normalise a function name to the fn_ namespace."""
    name = str(name).strip()
    return name if name.startswith("fn_") else f"fn_{name}"


def _domain_collections(doc: Dict[str, Any]) -> Set[str]:
    """Collect every collection referenced by an existing domain's tools/steps."""
    collections: Set[str] = set()
    for cfg in (doc.get("tools") or {}).values():
        if isinstance(cfg, dict):
            if cfg.get("collection"):
                collections.add(cfg["collection"])
            for step in cfg.get("steps") or []:
                if isinstance(step, dict) and step.get("collection"):
                    collections.add(step["collection"])
    return collections


def get_function_builder_toolspecs() -> List[Dict[str, Any]]:
    """Static Bedrock toolSpec dicts (bare names) for webui discovery via /agent/llm_tools."""
    return [
        {
            "toolSpec": {
                "name": "define_query_function",
                "description": (
                    "Author a READ-ONLY, multi-step query function and save it into a data "
                    "domain's tool set as fn_<name>. Auto-active in the 'dev' stage — test it, "
                    "then call promote_query_function to publish to 'prod'. Requires the "
                    "'fn_define' scope.\n\n"
                    "A function is an ordered list of steps. Each step runs a query "
                    "(uses='vector_search', 'vector_rerank_search', 'rerank', 'text_search', 'geospatial_search', "
                    "'hybrid_search', or 'aggregate'); values pulled "
                    "from a step's results via 'extract' bind to a name that later steps reference "
                    "with a whole-value {{token}} (e.g. {\"_id\": {\"$in\": \"{{listing_ids}}\"}}). Use "
                    "<domain>_get_collection_info first to pick indexed fields, vector index "
                    "names, and the geo (2dsphere) location field. No write stages ($merge/$out) "
                    "or $function/$where/$accumulator."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "Target data domain name (the 'Name' in mcp_tools) to save the function into."},
                            "name": {"type": "string", "description": "Function name; stored as fn_<name>."},
                            "collection": {"type": "string", "description": "Default collection for steps that omit their own 'collection'. Must belong to tool_name's domain."},
                            "steps": {
                                "type": "array",
                                "description": (
                                    "Ordered steps. Each: {name, uses, collection?, limit?, extract?, filters?}. Per uses:\n"
                                    "- 'aggregate': pipeline (read-only stages).\n"
                                    "- 'vector_search': query_text, index, vector_path, num_candidates?.\n"
                                    "- 'vector_rerank_search': query_text, index, vector_path, rerank_field, candidates?, num_candidates?. Self-contained: vector-retrieve then Voyage rerank; output carries both 'score' (vector) and 'relevance_score' (rerank).\n"
                                    "- 'rerank': query_text, rerank_field, plus ONE candidate feed — 'documents' (PREFERRED) or 'candidate_ids' — and candidates?, limit?, min_relevance?, projection?. Rescores an EARLIER step's results with Voyage. No retrieval/embedding here. Pair after hybrid_search/vector_search/text_search for keyword+concept precision.\n"
                                    "    • 'documents': whole-value {{token}} bound to a prior step's FULL rows via a 'docs'-mode extract. Preferred — the upstream projection AND retrieval 'score' pass straight through and rerank just appends 'relevance_score' (no re-fetch).\n"
                                    "    • 'candidate_ids': whole-value {{token}} bound to a prior step's extracted _id list; the docs are re-fetched by _id (projected to the floor + rerank_field, so the source body/embedding stays out).\n"
                                    "    • 'projection': optional list of EXTRA top-level field names to return, on top of the always-included floor (_id, relevance_score, score). Omit for floor-only.\n"
                                    "- 'text_search': query_text.\n"
                                    "- 'hybrid_search': query_text, vector_index, text_index, vector_path, text_fields?, vector_weight?, text_weight?.\n"
                                    "- 'geospatial_search': longitude, latitude, geo_field, max_distance_meters?, min_distance_meters?.\n"
                                    "extract: {bind_name:{field, mode:'list'|'set'|'scalar'|'docs'}}. list/set/scalar pull a single field per row (mode 'docs' binds the WHOLE rows — use it to feed a later 'rerank' step's 'documents'). A step field may be a whole-value "
                                    "token bound to a declared parameter, e.g. \"limit\": \"{{limit}}\", which keeps the parameter's declared type."
                                ),
                                "items": {"type": "object"},
                            },
                            "parameters": {"type": "object", "description": "Declared caller params: {name:{type, description, default?}}. Reference a param in a step via {{name}}; declare numeric knobs like limit/num_candidates with type 'integer' and pass them as whole-value tokens (e.g. \"limit\": \"{{limit}}\")."},
                            "required": {"type": "array", "items": {"type": "string"}, "description": "Names of required parameters."},
                            "output": {"type": "string", "description": "Which step's results to return (step 'name'). Defaults to the last step."},
                            "description": {"type": "string", "description": "What the function does (shown to the LLM as the tool description)."},
                            "returns": {"type": "string", "description": "Optional description of the return shape."},
                        },
                        "required": ["tool_name", "name", "collection", "steps"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "promote_query_function",
                "description": (
                    "Move a query function ONE step along the linear lifecycle "
                    "prod ↔ dev ↔ disabled. Direct prod↔disabled jumps and no-op transitions "
                    "are refused. Promoting to 'prod' makes it visible to all consumers. "
                    "Requires the 'fn_publish' scope."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "Data domain name holding the function."},
                            "name": {"type": "string", "description": "Function name (with or without the fn_ prefix)."},
                            "to_stage": {"type": "string", "description": "Target stage: disabled, dev, or prod."},
                        },
                        "required": ["tool_name", "name", "to_stage"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "delete_query_function",
                "description": (
                    "Delete a query function from a data domain's tool set. ONLY functions in "
                    "the 'disabled' stage can be deleted. The lifecycle is prod → dev → "
                    "disabled → delete, so demote it step-by-step via promote_query_function "
                    "first. Requires the 'fn_define' scope."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "Data domain name holding the function."},
                            "name": {"type": "string", "description": "Function name (with or without the fn_ prefix)."},
                        },
                        "required": ["tool_name", "name"],
                    }
                },
            }
        },
    ]


def register_function_builder_tools(mcp, settings, mongo_middleware, llm_client) -> Dict[str, Any]:
    """Register define/promote query-function tools on the given FastMCP instance.

    Returns {tool_name: fn} for inclusion in _TOOL_DISPATCH.
    """
    # Dedicated MemoryService for versioning (mirrors register_memory_tools). Connects
    # lazily inside the running event loop on first strategy_store call.
    memory_db = getattr(settings, "memory_db", "mcp_config")
    query_model = getattr(settings, "QUERY_EMBEDDING_MODEL_ID", None)
    _svc = MemoryService(
        db_client=MongoDBClient(settings),
        llm_client=llm_client,
        memory_db_name=memory_db,
        query_embedding_model_id=query_model,
        agent_instructions="",
    )

    def _tools_collection():
        """Sync PyMongo handle on the shared mcp_tools config collection."""
        mongo_middleware.mongo_client.sync_connect_to_mongodb()
        return mongo_middleware.mongo_client.get_collection()

    @mcp.tool()
    async def define_query_function(
        tool_name: Annotated[str, Field(description="Target data domain name (the 'Name' in mcp_tools) to save the function into.")],
        name: Annotated[str, Field(description="Function name; stored as fn_<name>.")],
        collection: Annotated[str, Field(description="Default collection for steps that omit their own 'collection'. Must belong to tool_name's domain.")],
        steps: Annotated[List[Dict[str, Any]], Field(description="Ordered read-only steps (vector_search / aggregate) with optional extract binds.")],
        parameters: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Declared caller params {name:{type,description,default?}} referenced via {{name}}.")] = None,
        required: Annotated[Optional[List[str]], Field(default=None, description="Names of required parameters.")] = None,
        output: Annotated[Optional[str], Field(default=None, description="Which step's results to return (step name). Defaults to last step.")] = None,
        description: Annotated[str, Field(default="", description="What the function does.")] = "",
        returns: Annotated[str, Field(default="", description="Optional description of the return shape.")] = "",
        token: Annotated[AccessToken, Depends(get_access_token)] = None,
    ) -> Dict[str, Any]:
        """Author a read-only multi-step query function into a data domain (auto-active in dev). Requires 'fn_define' scope."""
        scopes, agent_id = _extract_scopes(token)
        if "fn_define" not in scopes:
            return {"error": "Insufficient scope: 'fn_define' permission required to define query functions."}

        # Raw JWT + request base_url for cross-container routing to the owning domain's
        # container (used below to call its get_collection_info). See build_mcp_http_call.
        jwt = token.get("token") if isinstance(token, dict) else getattr(token, "token", None)
        try:
            base_url = get_http_request().base_url
        except Exception:
            base_url = None
        mcp_call = build_mcp_http_call(jwt, base_url)

        fn_key = _fn_key(name)
        cfg = {
            "handler": "multi_step",
            "stage": "dev",
            "collection": collection,
            "steps": steps,
            "parameters": parameters or {},
            "required": required or [],
            "output": output,
            "description": description,
            "returns": returns,
        }

        # --- Static read-only + token-resolvability validation ---
        errors, info = validate_function_config(cfg)
        if errors:
            return {"error": "Function validation failed.", "details": errors}

        # --- Target domain must exist and be active ---
        try:
            coll = _tools_collection()
            doc = coll.find_one({"Name": tool_name})
        except Exception as e:
            logger.error("define_query_function: mcp_tools read failed: %s", e)
            return {"error": f"Could not read mcp_tools config: {e}"}
        if not doc:
            return {"error": f"Unknown data domain '{tool_name}'."}
        if not doc.get("active", False):
            return {"error": f"Data domain '{tool_name}' is not active."}

        # --- Fetch the domain's live collections + the target collection's indexes ---
        # define_query_function is universal (runs on /agent) but MUST NOT touch another
        # domain's database directly. Route the existing get_collection_info tool to the
        # domain's own container over the data-domain route (see build_mcp_http_call) so it
        # answers from its own connection/credentials. This returns ALL real collections in
        # the domain's DB (list_collection_names), so we use it as the authoritative
        # allow-list below — NOT just the collections already wired into existing tools
        # (which would be stale and block authoring against a freshly-added collection).
        # Best-effort: never blocks a define.
        live_collections: Set[str] = set()
        live_ok = False
        live_err: Optional[str] = None
        available_indexes: List[Dict[str, Any]] = []
        try:
            raw_ci = await mcp_call(f"{tool_name}_get_collection_info", {})
            ci = json.loads(raw_ci) if isinstance(raw_ci, str) else raw_ci
            if isinstance(ci, dict) and "collection_info" in ci:
                ci = ci["collection_info"]
            colls = (((ci or {}).get("mongodb") or {}).get("collections") or {}) if isinstance(ci, dict) else {}
            live_collections = {c for c in colls.keys() if c}
            live_ok = bool(live_collections)
            cinfo = colls.get(collection) or {}
            for _idx in cinfo.get("indexes", []) or []:
                _key = _idx.get("key")
                _keys = list(_key.keys()) if isinstance(_key, dict) else (_key if isinstance(_key, list) else [])
                available_indexes.append({"name": _idx.get("name"), "key": _keys})
            for _sidx in cinfo.get("search_indexes", []) or []:
                _nm = _sidx.get("name") if isinstance(_sidx, dict) else _sidx
                if _nm:
                    available_indexes.append({"name": _nm, "type": "search"})
        except Exception as e:
            live_err = str(e)
            logger.warning("define_query_function: get_collection_info lookup FAILED for %s.%s: %s", tool_name, collection, e)
        cfg["available_indexes"] = available_indexes

        # --- Step collections must belong to that domain ---
        # Authoritative allow-list is the domain's LIVE DB collections (list_collection_names
        # via get_collection_info). We only ENFORCE membership when that live enumeration
        # actually succeeded — otherwise we must NOT block on the config-derived set
        # (_domain_collections), which is generated from scoping.read and goes stale the
        # moment a new collection is added but not yet wired into a tool. A read-only step
        # against a truly-absent collection just fails harmlessly at execution time.
        step_collections = {collection} | {s.get("collection") for s in steps if isinstance(s, dict) and s.get("collection")}
        step_collections.discard(None)
        if live_ok:
            known = live_collections | _domain_collections(doc)
            unknown = step_collections - known
            if unknown:
                return {
                    "error": f"Collection(s) {sorted(unknown)} are not part of domain '{tool_name}'.",
                    "known_collections": sorted(known),
                }
        else:
            logger.warning(
                "define_query_function: live collection enumeration unavailable for '%s' (%s); "
                "skipping membership check so authoring is not blocked by a stale config set.",
                tool_name, live_err or "empty get_collection_info result",
            )

        # --- No shadowing of a non-function tool ---
        existing = doc.get("tools") or {}
        if fn_key in existing and existing[fn_key].get("handler") != "multi_step":
            return {"error": f"'{fn_key}' already exists as a non-function tool in '{tool_name}'; refusing to overwrite."}

        # --- Version in memory (strategy_store auto-increments version_seq) ---
        version_seq = None
        strategy_id = None
        try:
            payload = {
                "tool_name": tool_name,
                "collection": collection,
                "config_json": json.dumps(cfg, default=str),
                "available_indexes": available_indexes,
                "stage": "dev",
                "fn_key": fn_key,
            }
            strat = await _svc.strategy_store(
                name=fn_key,
                context=description or f"Read-only query function {fn_key} on {tool_name}.{collection}",
                memory_type="strategy:query_function",
                payload=payload,
                scope=0,
                agent_id=agent_id,
                entities=[tool_name, collection, "query_function"],
                importance=0.9,
                decay_rate=0.0,
            )
            if isinstance(strat, dict):
                version_seq = strat.get("version_seq")
                strategy_id = strat.get("_id") or strat.get("id")
        except Exception as e:
            logger.error("define_query_function: strategy_store versioning failed: %s", e)

        cfg["version_seq"] = version_seq
        cfg["source_strategy_id"] = strategy_id
        cfg["created_by"] = agent_id
        cfg["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # --- Persist the live copy into mcp_tools ---
        try:
            coll.update_one({"Name": tool_name}, {"$set": {f"tools.{fn_key}": cfg}})
        except Exception as e:
            logger.error("define_query_function: mcp_tools write failed: %s", e)
            return {"error": f"Could not write function to mcp_tools: {e}"}

        return {
            "status": "defined",
            "tool_name": tool_name,
            "function": fn_key,
            "stage": "dev",
            "version_seq": version_seq,
            "available_indexes": available_indexes,
            "note": (
                "Function is auto-active in the 'dev' stage. Test it in dev mode, then call "
                "promote_query_function(to_stage='prod') to publish it for all consumers."
            ),
        }

    @mcp.tool()
    async def promote_query_function(
        tool_name: Annotated[str, Field(description="Data domain name holding the function.")],
        name: Annotated[str, Field(description="Function name (with or without the fn_ prefix).")],
        to_stage: Annotated[str, Field(description="Target stage: disabled, dev, or prod.")],
        token: Annotated[AccessToken, Depends(get_access_token)] = None,
    ) -> Dict[str, Any]:
        """Move a query function ONE step along the lifecycle prod ↔ dev ↔ disabled (no direct prod↔disabled jumps). Requires 'fn_publish' scope."""
        scopes, _agent_id = _extract_scopes(token)
        if "fn_publish" not in scopes:
            return {"error": "Insufficient scope: 'fn_publish' permission required to promote query functions."}
        if to_stage not in _VALID_STAGES:
            return {"error": f"Invalid stage '{to_stage}'. Must be one of {list(_VALID_STAGES)}."}

        fn_key = _fn_key(name)
        try:
            coll = _tools_collection()
            doc = coll.find_one({"Name": tool_name})
        except Exception as e:
            return {"error": f"Could not read mcp_tools config: {e}"}
        if not doc:
            return {"error": f"Unknown data domain '{tool_name}'."}
        entry = (doc.get("tools") or {}).get(fn_key)
        if not entry:
            return {"error": f"Function '{fn_key}' not found in domain '{tool_name}'."}
        if entry.get("handler") != "multi_step":
            return {"error": f"'{fn_key}' is not a defined query function; refusing to change its stage."}

        # Enforce single-step moves along the linear lifecycle (both directions).
        current = entry.get("stage") or "dev"
        if current not in _STAGE_CHAIN:
            current = "dev"
        cur_i = _STAGE_CHAIN.index(current)
        tgt_i = _STAGE_CHAIN.index(to_stage)
        if abs(cur_i - tgt_i) != 1:
            return {
                "error": (
                    f"Invalid stage transition '{current}' → '{to_stage}'. Functions move ONE "
                    f"step at a time along the lifecycle prod ↔ dev ↔ disabled; direct "
                    f"prod↔disabled jumps and no-op transitions are not allowed."
                )
            }

        update: Dict[str, Any] = {f"tools.{fn_key}.stage": to_stage}
        if to_stage == "prod":
            update[f"tools.{fn_key}.published_version_seq"] = entry.get("version_seq")
        try:
            coll.update_one({"Name": tool_name}, {"$set": update})
        except Exception as e:
            return {"error": f"Could not update function stage: {e}"}

        return {
            "status": "promoted",
            "tool_name": tool_name,
            "function": fn_key,
            "stage": to_stage,
            "published_version_seq": entry.get("version_seq") if to_stage == "prod" else None,
        }

    @mcp.tool()
    async def delete_query_function(
        tool_name: Annotated[str, Field(description="Data domain name holding the function.")],
        name: Annotated[str, Field(description="Function name (with or without the fn_ prefix).")],
        token: Annotated[AccessToken, Depends(get_access_token)] = None,
    ) -> Dict[str, Any]:
        """Delete a query function from a data domain. Only 'disabled'-stage functions can be deleted (lifecycle prod → dev → disabled → delete). Requires 'fn_define' scope."""
        scopes, _agent_id = _extract_scopes(token)
        if "fn_define" not in scopes:
            return {"error": "Insufficient scope: 'fn_define' permission required to delete query functions."}

        fn_key = _fn_key(name)
        try:
            coll = _tools_collection()
            doc = coll.find_one({"Name": tool_name})
        except Exception as e:
            return {"error": f"Could not read mcp_tools config: {e}"}
        if not doc:
            return {"error": f"Unknown data domain '{tool_name}'."}
        entry = (doc.get("tools") or {}).get(fn_key)
        if not entry:
            return {"error": f"Function '{fn_key}' not found in domain '{tool_name}'."}
        if entry.get("handler") != "multi_step":
            return {"error": f"'{fn_key}' is not a defined query function; refusing to delete."}
        stage = entry.get("stage")
        if stage != "disabled":
            return {
                "error": (
                    f"Refusing to delete '{fn_key}': only 'disabled'-stage functions can be "
                    f"deleted (current stage: '{stage}'). The lifecycle is prod → dev → disabled "
                    f"→ delete; demote it step-by-step via promote_query_function first."
                )
            }

        try:
            coll.update_one({"Name": tool_name}, {"$unset": {f"tools.{fn_key}": ""}})
        except Exception as e:
            return {"error": f"Could not delete function: {e}"}

        return {
            "status": "deleted",
            "tool_name": tool_name,
            "function": fn_key,
            "note": (
                "Removed the live disabled-stage function from mcp_tools. Versioned history in "
                "the memory system (strategy:query_function) is retained."
            ),
        }

    return {
        "define_query_function": getattr(define_query_function, "fn", define_query_function),
        "promote_query_function": getattr(promote_query_function, "fn", promote_query_function),
        "delete_query_function": getattr(delete_query_function, "fn", delete_query_function),
    }
