import asyncio
import json
import os
import queue
import threading
import time
import traceback
from typing import Any, List, Optional

import fastmcp
import mcp.types as mt
import requests
from pydantic import BaseModel

from settings_loader import settings
from mongomcp.agent.cache_utils import SimpleCache
from mongomcp.agent.domain_tools import DomainToolManager
from mongomcp.agent.webui_bedrock_client import WebUiBedrockClient
import conversation_checkpoint as ckpt

import logging
logger = logging.getLogger(__name__)


class QueryResponse(BaseModel):
    content: Optional[dict] = None
    error: Optional[str] = None
    status: Optional[str] = None
    history: Optional[List[Any]] = None
    message: Optional[str] = None
    clear_history: Optional[bool] = None

    def json(self):
        if self.message is not None:
            self.message = self.message.replace("\n", " ").replace("\r", "")
        return self.model_dump_json()


class QueryRequest(BaseModel):
    input: str
    history: Optional[List[Any]] = None
    user_id: Optional[str] = None
    username: Optional[str] = None
    # True when user_id/username were injected from a VERIFIED auth-provider token
    # (vs self-declared). Lets the agent trust the identity instead of treating the
    # app's service token as unverified.
    identity_verified: Optional[bool] = False
    session_id: Optional[str] = None
    # UI-authoritative dynamic-function stage ('dev' | 'prod'). Sent on every query so
    # whichever gunicorn worker serves the turn honors the stage the UI is showing.
    function_stage: Optional[str] = None
    ip: Optional[str] = None


class APIQueryProcessor:
    _NO_TOOLS_ERROR = "No MCP tools configured. Tool discovery may have failed."
    TOOLS_CACHE_TTL_SECONDS = 300  # re-discover tools every 5 minutes

    # Tool suffixes that return stable schema/index data — cache for 1 hour.
    _CACHED_TOOL_SUFFIXES = frozenset({
        "list_indexes",
        "list_search_indexes",
    })
    _CACHED_TOOL_TTL = 3600  # seconds

    def __init__(self):
        self._init_error: Optional[Exception] = None
        self._message_queue: queue.Queue = queue.Queue()

        logger.info(f"Initializing processor with endpoint: {settings.mongo_mcp_root}")
        try:
            self.llm_client = WebUiBedrockClient(settings)
            self.mcp_endpoints: List[str] = []
            self.mcp_endpoint_configs: dict = {}
            self.endpoint_clients: dict = {}
            self.mongo_collection_info: dict = {}
            self.mcp_tools_config: Optional[List[dict]] = None
            # Progressive tool disclosure across data domains. The full per-domain tool
            # catalog lives inside the manager; only core (memory/agent) tools + the
            # describe/activate stubs are exposed to the LLM until a domain is activated.
            # Activation is tracked per session_id (sticky within a conversation).
            self._domain_mgr: Optional[DomainToolManager] = None
            # Per-turn buffer of tool results flagged bypass=true — rendered directly in
            # the UI (bolt-on to the existing [JSON_DATA] flag system). Reset each turn.
            self._bypass_items: list = []
            # Which dynamic-function stage to surface during tool discovery: 'prod'
            # (default) shows only published functions; 'dev' also surfaces
            # in-development functions so they can be iterated before publishing.
            self._function_stage: str = "prod"
            self._session_instructions: Optional[str] = None
            self._tools_fetched_at: Optional[float] = None
            self._base_system_prompt: List[dict] = []
            # Deterministic conversation checkpointing. Per-session cache of the
            # checkpoint memory _id so writes upsert in place (no duplicates) and a
            # marker of which sessions have a checkpoint worth loading on later turns.
            self._checkpoint_ids: dict = {}
            # Per-session completed-turn counter driving the checkpoint write cadence.
            self._checkpoint_turn_count: dict = {}
            # Opt-in cross-turn MCP session reuse (MCP_SESSION_POOL=1). Sessions live on a
            # dedicated background event loop thread so they persist for the life of the
            # process (each turn otherwise runs on its own short-lived asyncio.run loop).
            # Tool calls bridge onto this loop via run_coroutine_threadsafe; a session is
            # evicted+rebuilt on failure (handles idle drops / server restarts).
            # MCP session reuse strategy (MCP_SESSION_POOL), A/B-selectable at deploy time:
            #   off/0        -> fresh fastmcp.Client per tool call (its own initialize handshake)
            #   turn/loop    -> pool per turn: sessions are reused across a turn's concurrent
            #                   tool fan-out on THAT turn's own event loop (worker thread), then
            #                   closed when the turn ends. No cross-thread bridge; parallel
            #                   across requests. Pays the handshake once per endpoint per turn.
            #   container/1  -> pool per container: sessions persist ACROSS turns on one shared
            #                   background loop, so the handshake is paid once per process. But
            #                   every request funnels through that single loop thread.
            _raw_pool = os.getenv("MCP_SESSION_POOL", "").strip().lower()
            if _raw_pool in ("1", "true", "yes", "container"):
                self._pool_mode = "container"
            elif _raw_pool in ("turn", "loop", "perturn", "per-turn"):
                self._pool_mode = "turn"
            else:
                self._pool_mode = "off"
            # Per-endpoint pool of warm sessions. A single shared session would serialise a
            # turn's concurrent tool fan-out; a small pool lets those calls run in parallel
            # while still amortising the initialize handshake. Bounded so we don't open
            # unbounded connections; calls block for an idle session once saturated.
            try:
                self._pool_max = max(1, int(os.getenv("MCP_SESSION_POOL_SIZE", "4")))
            except ValueError:
                self._pool_max = 4
            self._pool_loop = None
            self._pool_loop_thread = None
            self._pool_loop_lock = threading.Lock()
            self._pool_sessions: dict = {}          # container mode: endpoint -> entry (on _pool_loop)
            self._turn_pools: dict = {}             # turn mode: loop-id -> {endpoint -> entry}
            self._turn_pools_lock = threading.Lock()
            self._tool_response_cache = SimpleCache(
                settings, username="webui", session_id="global",
                cache_object_name="tool_response",
            )
            self._discover_tools()
        except Exception as e:
            self._init_error = e
            logger.error(f"Processor initialization failed: {e}")

    @property
    def _headers(self) -> dict:
        """Build auth headers fresh on every call so Cognito tokens are never stale."""
        return {
            "Authorization": f"Bearer {settings.get_auth_token()}",
            "Content-Type": "application/json",
        }

    @property
    def init_error(self) -> Optional[Exception]:
        return self._init_error

    def _emit(self, message, status="Processing"):
        if isinstance(message, Exception):
            resp = QueryResponse(status="Error", error=str(message), message=str(message))
        else:
            resp = QueryResponse(status=status, message=str(message))
        self._message_queue.put(resp.json())

    def pop_queued_messages(self) -> List[str]:
        msgs = []
        try:
            while True:
                msgs.append(self._message_queue.get_nowait())
        except queue.Empty:
            pass
        return msgs

    def read_message_stream(self, timeout=0.1):
        try:
            while True:
                try:
                    yield self._message_queue.get(timeout=timeout)
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error reading message stream: {e}")

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def _discover_tools(self, emit=None):
        asyncio.run(self._discover_tools_async(emit=emit))

    def _looks_like_no_tools_error(self, result: dict) -> bool:
        err = (result or {}).get("error")
        if not isinstance(err, str):
            return False
        return self._NO_TOOLS_ERROR in err

    def _rediscover_tools(self, emit=None) -> bool:
        """Refresh local tool discovery, retrying up to 3 times with back-off."""
        import time
        emit_fn = emit or self._emit
        for attempt in range(1, 4):
            try:
                self._discover_tools(emit=emit_fn)
                if self.mcp_tools_config:
                    return True
            except Exception as e:
                logger.error(f"Tool rediscovery attempt {attempt} failed: {e}")
                emit_fn(f"Tool rediscovery attempt {attempt} failed: {e}", status="Error")
            if attempt < 3:
                emit_fn(f"Retrying tool discovery (attempt {attempt + 1}/3)...", status="Recovering")
                time.sleep(attempt * 2)
        emit_fn("Tool rediscovery exhausted all attempts.", status="Error")
        return False

    async def _discover_tools_async(self, emit=None):
        emit_fn = emit or self._emit
        try:
            # Agentcore gateway serves discovery at GET /  (returns {"services": [...]}).
            # Legacy deployments serve it at /tools_config (returns {"available_tools": [...]}).
            # Try the root first; fall back to /tools_config on 404.
            for discovery_path in ("/", "/tools_config"):
                resp = requests.get(
                    f"{settings.mongo_mcp_root}{discovery_path}",
                    headers=self._headers, timeout=15,
                )
                logger.info("Discovery %s -> HTTP %s", discovery_path, resp.status_code)
                if resp.status_code == 404 and discovery_path == "/":
                    continue
                resp.raise_for_status()
                data = resp.json()
                logger.info("Discovery response keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))
                self.mcp_endpoints = data.get("available_tools") or data.get("services", [])
                break
            logger.info("Discovered endpoints: %s", self.mcp_endpoints)
            emit_fn(f"Discovered endpoints: {self.mcp_endpoints}", status="Discovering Tools")
        except Exception as e:
            logger.error("Discovery failed: %s", e, exc_info=True)
            emit_fn(f"Error fetching endpoint list: {e}", status="Error")
            self.mcp_endpoints = []

        root_fmt = f"{settings.mongo_mcp_root}/{{}}/mcp"
        results = await asyncio.gather(*[
            self._fetch_endpoint(name, root_fmt, emit=emit_fn) for name in self.mcp_endpoints
        ])

        bedrock_tools, agent_prompts = [], {}
        domain_registry: dict = {}
        domain_descriptions: dict = {}
        for name, config, tools, collection_info, agent_prompt, module_description in results:
            self.mcp_endpoint_configs[name] = config
            bedrock_tools.extend(tools)
            # Data-domain tool specs are held back from the LLM until activated.
            if name not in {"memory", "agent"} and tools:
                for t in tools:
                    self._inject_bypass_param(t)
                domain_registry[name] = tools
                domain_descriptions[name] = module_description or agent_prompt or ""
            if collection_info:
                self.mongo_collection_info[name] = collection_info
            if agent_prompt and agent_prompt.strip():
                agent_prompts[name] = agent_prompt.strip()

        # Fetch DB-driven instructions (master_instructions + webui-specific overrides).
        # Only attempt if the memory endpoint actually loaded tools — if memory/mcp is
        # unreachable (e.g. no ingress route on K8s), skip to avoid blocking startup.
        # Also cap each call at 15s so a slow endpoint never kills the gunicorn worker.
        memory_has_tools = any(
            t.get("toolSpec", {}).get("name", "").startswith("memory_")
            for t in bedrock_tools
        )
        db_prompts = []
        if not memory_has_tools:
            logger.info("Skipping strategy recall — memory endpoint has no tools (unreachable or not deployed)")
        else:
            for strategy_name in ("master_instructions", "dynamicmcp_webui_instructions"):
                try:
                    raw = await asyncio.wait_for(
                        self._call_mcp_tool("memory_strategy_recall", {"name": strategy_name}),
                        timeout=15,
                    )
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    strategies = (data or {}).get("strategies") or (data or {}).get("results", [])
                    if strategies:
                        content = strategies[0].get("content", "")
                        if content:
                            db_prompts.append(content)
                except Exception as exc:
                    logger.warning("Failed to load strategy %s: %s", strategy_name, exc)

        if db_prompts:
            system_prompt = [{"text": t} for t in db_prompts]
        else:
            system_prompt = [{"text": t} for t in getattr(settings, "BEDROCK_SYSTEM_PROMPT_TEXTS", [])]
        for ep, pt in agent_prompts.items():
            system_prompt.append({"text": f"***IMPORTANT {ep}: {pt}"})
        # Do NOT preload per-domain collection metadata (indexes/search_indexes) here.
        # It is byte-for-byte identical to what '<domain>_get_collection_info' returns
        # on demand, and preloading the union of every domain is paid on every single
        # conversation. Point the agent at the on-demand tool instead.
        data_domains = [ep for ep in self.mcp_endpoints if ep not in {"memory", "agent"}]
        if data_domains:
            domain_lines = "\n".join(
                f"- {d}: {domain_descriptions.get(d, '') or '(no description)'}"
                for d in data_domains
            )
            system_prompt.append({"text":
                "Data domains available (their query tools are NOT loaded until activated, "
                "to save context):\n" + domain_lines + "\n"
                "Workflow: call 'activate_data_domain' (mode: add/replace/remove/clear) to "
                "load a domain's query tools — it returns those tools, so you usually do NOT "
                "need 'describe_data_domain' first (use describe only to inspect a domain "
                "without loading it). Call '<domain>_get_collection_info' before querying "
                "ONLY when you need schema/index/field names (e.g. a raw aggregate_query or "
                "building filters) or on first contact with an unfamiliar collection; skip it "
                "when running a pre-built vector/hybrid search on a known corpus. For large or "
                "geographic result sets, pass bypass=true to a query tool to render its results "
                "directly in the UI (you get back only a count) instead of returning the data "
                "into the conversation. Until you activate a data domain, only memory and agent "
                "tools are available."
            })

        # Recalling prior conversations: the running conversation is checkpointed
        # automatically (memory_type="conversation"), hidden from normal recall by default.
        system_prompt.append({"text":
            "Recalling past conversations: prior conversations in this browser are saved "
            "automatically as checkpoints. If the user asks to resume or references an "
            "earlier discussion ('what were we doing', 'go back to...'), call 'memory_recall' "
            "with memory_types=[\"conversation\"] and a short query describing the topic. "
            "These are excluded from normal recall, so you must request the type explicitly. "
            "Results are already scoped to the current browser."
        })

        self._base_system_prompt = system_prompt
        self.llm_client.system = system_prompt
        # Expose only the always-on core (memory + agent) plus the describe/activate stubs.
        # Full per-domain specs stay inside the DomainToolManager until activated.
        core_tools = [
            t for t in bedrock_tools
            if t.get("toolSpec", {}).get("name", "").startswith(("memory_", "agent_"))
        ]
        self._domain_mgr = DomainToolManager(
            core_tools=core_tools,
            registry=domain_registry,
            descriptions=domain_descriptions,
        )
        self.mcp_tools_config = self._domain_mgr.base_tool_set()
        self._tools_fetched_at = time.monotonic()
        self.llm_client.configure_tools(self.mcp_tools_config, self._call_mcp_tool)
        # Simple list of the active domains: always-on core (memory/agent) plus any
        # data domains activated so far (none at load).
        active_domains = [ep for ep in self.mcp_endpoints if ep in {"memory", "agent"}]
        active_domains += self._domain_mgr.active_domains("_global")
        emit_fn(f"Active tools: {', '.join(active_domains) or 'none'}", status="Tools Ready")

    async def _fetch_endpoint(self, name, root_fmt, emit=None):
        emit_fn = emit or self._emit
        config = {
            "url": root_fmt.format(name),
            "transport": "http",
            "headers": {"Authorization": f"Bearer {settings.get_auth_token()}"},
        }
        tools, agent_prompt, collection_info, module_description = [], "", {}, ""
        try:
            resp = await asyncio.to_thread(
                lambda: requests.get(
                    f"{settings.mongo_mcp_root}/{name}/llm_tools",
                    headers=self._headers, params={"stage": self._function_stage}, timeout=15,
                ),
            )
            resp.raise_for_status()
            payload = resp.json()
            agent_prompt = payload.get("agent_prompt", "") if isinstance(payload, dict) else ""
            module_description = payload.get("description", "") if isinstance(payload, dict) else ""
            tools = payload.get("tools", []) if isinstance(payload, dict) else []
            for tool in tools:
                if "toolSpec" in tool and "name" in tool["toolSpec"]:
                    tool["toolSpec"]["name"] = f"{name}_{tool['toolSpec']['name']}"
        except Exception as e:
            emit_fn(f"Error fetching tools for {name}: {e}", status="Error")
        # Collection schema/index metadata is intentionally NOT fetched at load. It is
        # identical to the on-demand '<domain>_get_collection_info' tool result, so
        # preloading it into every conversation's context is pure token waste.
        return name, config, tools, collection_info, agent_prompt, module_description

    # ------------------------------------------------------------------
    # Progressive tool disclosure (DomainToolManager)
    # ------------------------------------------------------------------

    def _apply_active_tools(self, session_id: Optional[str]) -> None:
        """Configure the live tool set = base (core + describe/activate stubs) + the
        domains this session has activated. Called at the start of every turn and after
        each activation, so the set is sticky within a conversation."""
        if self._domain_mgr is None:
            return
        tools = self._domain_mgr.active_tool_set(session_id or "_global")
        self.mcp_tools_config = tools
        self.llm_client.configure_tools(tools, self._call_mcp_tool)

    async def _refresh_domain_tools(self, domains=None) -> None:
        """Re-fetch data-domain tool specs from their MCP endpoints and update the
        DomainToolManager registry IN PLACE (active-domain state preserved), so functions
        defined/promoted mid-session (fn_* dynamic query functions) are picked up.

        With domains=None, refreshes every data domain. The server reloads its annotations
        from MongoDB on each /llm_tools call, so grabbing the full set on activation is
        cheap and always current — no restart or full re-discovery needed."""
        if self._domain_mgr is None:
            return
        if isinstance(domains, str):
            domains = [domains]
        if not domains:
            domains = [ep for ep in self.mcp_endpoints if ep not in {"memory", "agent"}]
        targets = [d for d in domains if d and d not in {"memory", "agent"} and d in self.mcp_endpoints]
        if not targets:
            return
        root_fmt = f"{settings.mongo_mcp_root}/{{}}/mcp"
        results = await asyncio.gather(*[
            self._fetch_endpoint(name, root_fmt) for name in targets
        ], return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.warning("Domain tool refresh failed: %s", res)
                continue
            name, _config, tools, _collection_info, agent_prompt, module_description = res
            if not tools:
                continue
            for t in tools:
                self._inject_bypass_param(t)
            self._domain_mgr.set_domain_tools(name, tools, module_description or agent_prompt or None)

    def set_function_stage(self, stage: str) -> dict:
        """Switch which dynamic-function stage is surfaced ('dev' or 'prod') and re-discover.

        'dev' surfaces in-development functions (plus prod) so they can be iterated before
        publishing; 'prod' (default) shows only published functions.
        """
        stage = stage if stage in ("dev", "prod") else "prod"
        self._function_stage = stage
        self._rediscover_tools()
        return {"status": "ok", "function_stage": stage}

    # ------------------------------------------------------------------
    # Bypass: render tool results directly in the UI (bolt-on to [JSON_DATA])
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_bypass_param(tool: dict) -> None:
        """Add an optional 'bypass' boolean to a data-domain tool's inputSchema so the LLM
        can request the result be rendered directly in the UI instead of returned into the
        conversation (saves tokens; for large/geo result sets)."""
        schema = tool.get("toolSpec", {}).get("inputSchema", {}).get("json")
        if isinstance(schema, dict):
            props = schema.setdefault("properties", {})
            props.setdefault("bypass", {
                "type": "boolean",
                "description": (
                    "If true, this tool's full results are rendered directly in the UI "
                    "(map/table) and NOT returned into the conversation — you receive only a "
                    "small acknowledgment with the result count. Use for large or geographic "
                    "result sets you do not need to read or analyze in text."
                ),
            })

    @staticmethod
    def _deep_parse(value):
        """Unwrap up to 3 levels of JSON-string encoding."""
        v = value
        for _ in range(3):
            if not isinstance(v, str):
                break
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                break
        return v

    @staticmethod
    def _unwrap_renderable(data):
        """Tool results are often enveloped as {'results':[...], 'count':N}. When the envelope
        wraps a SINGLE already-renderable object (has jsonDataType, markers, or is GeoJSON),
        unwrap it so the frontend renderer receives the payload directly instead of the wrapper.
        Multi-row/coordinate result arrays are left as-is (the frontend auto-detects those)."""
        if isinstance(data, dict):
            for key in ("results", "data", "items"):
                inner = data.get(key)
                if isinstance(inner, list) and len(inner) == 1 and isinstance(inner[0], dict):
                    el = inner[0]
                    if "jsonDataType" in el or "markers" in el or el.get("type") == "FeatureCollection":
                        return el
        return data

    def _capture_bypass(self, domain: str, tool_name: str, result) -> str:
        """Stash a bypassed tool result for direct UI rendering; return a compact ack."""
        data = self._unwrap_renderable(self._deep_parse(result))
        self._bypass_items.append({"tool": tool_name, "domain": domain, "data": data})
        count = None
        if isinstance(data, dict):
            for k in ("markers", "features", "results", "items"):
                v = data.get(k)
                if isinstance(v, list):
                    count = len(v)
                    break
        elif isinstance(data, list):
            count = len(data)
        self._emit(
            f"Bypass: {tool_name} rendered to UI ({count if count is not None else '?'} items)",
            status="Rendered To UI",
        )
        ack = {
            "status": "rendered_to_ui",
            "tool": tool_name,
            "count": count,
            "note": ("Full results were rendered directly in the UI. Briefly summarize for the "
                     "user; the data is NOT in this conversation."),
        }
        # If the payload carries a 'summary' object, pass it to the LLM so it can narrate
        # real stats (totals, categories, distances) without the full result set in context.
        if isinstance(data, dict) and isinstance(data.get("summary"), dict):
            ack["summary"] = data["summary"]
            ack["note"] = ("Full results were rendered directly in the UI. Use the 'summary' "
                           "object below for your text summary; the full data is NOT in this "
                           "conversation.")
        return json.dumps(ack)

    def _apply_bypass_jsondata(self, jsondata):
        """Captured bypass results take over content.jsondata (single → its data; multiple →
        a 'multi' wrapper). No captures → unchanged (falls back to the [JSON_DATA] block)."""
        if not self._bypass_items:
            return jsondata
        if len(self._bypass_items) == 1:
            return self._bypass_items[0]["data"]
        return {"jsonDataType": "multi", "items": list(self._bypass_items)}

    # ------------------------------------------------------------------
    # MCP tool dispatch
    # ------------------------------------------------------------------

    def _ensure_pool_loop(self):
        """Lazily start the dedicated background event loop that hosts container-level MCP
        sessions. Sessions created on this loop persist across turns (each turn otherwise
        runs on its own short-lived asyncio.run loop). Idempotent and thread-safe."""
        if self._pool_loop is not None and self._pool_loop.is_running():
            return
        with self._pool_loop_lock:
            if self._pool_loop is not None and self._pool_loop.is_running():
                return
            ready = threading.Event()
            loop = asyncio.new_event_loop()

            def _runner():
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()

            t = threading.Thread(target=_runner, daemon=True, name="mcp-session-pool-loop")
            t.start()
            ready.wait(timeout=5)
            self._pool_loop = loop
            self._pool_loop_thread = t

    async def _pool_checkout(self, store, endpoint_name, cfg, mcp_timeout):
        """Check out a warm session from the endpoint's pool in `store` (runs ON the loop
        that owns `store`).

        Returns (client, opened_ms). opened_ms is the initialize cost for a freshly opened
        session and 0 when an idle session is reused. Reuses an idle session if one is free;
        otherwise opens a new one while the pool is below its max; otherwise blocks until a
        busy session is checked back in. The caller MUST return the client via
        _pool_checkin (success) or _pool_discard (failure).
        """
        entry = store.get(endpoint_name)
        if entry is None:
            entry = {"idle": asyncio.Queue(), "size": 0, "max": self._pool_max, "clients": set()}
            store[endpoint_name] = entry

        # Fast path: reuse an idle session with no handshake.
        try:
            return entry["idle"].get_nowait(), 0
        except asyncio.QueueEmpty:
            pass

        # Grow the pool if we're under the cap. Reserve the slot synchronously (before any
        # await) so concurrent checkouts don't collectively exceed max.
        if entry["size"] < entry["max"]:
            entry["size"] += 1
            try:
                client = fastmcp.Client({"mcpServers": {endpoint_name: cfg}}, timeout=mcp_timeout)
                _t_open = time.monotonic()
                await client.__aenter__()
                entry["clients"].add(client)
                return client, int((time.monotonic() - _t_open) * 1000)
            except Exception:
                entry["size"] -= 1
                raise

        # Pool saturated: wait for a busy session to be returned.
        return await entry["idle"].get(), 0

    def _pool_checkin(self, store, endpoint_name, client):
        """Return a healthy session to the idle pool (runs ON the loop that owns `store`)."""
        entry = store.get(endpoint_name)
        if entry is not None and client in entry["clients"]:
            entry["idle"].put_nowait(client)

    async def _pool_discard(self, store, endpoint_name, client):
        """Close and drop a failed session so the pool rebuilds it (runs ON the owning loop)."""
        entry = store.get(endpoint_name)
        if entry is not None and client in entry["clients"]:
            entry["clients"].discard(client)
            entry["size"] -= 1
        try:
            await client.__aexit__(None, None, None)
        except Exception:
            pass

    async def _pool_send(self, store, endpoint_name, cfg, mcp_timeout, do_send, timing):
        """Run one tool call against a pooled session from `store` (ON the owning loop).

        Checks out a warm session, runs the call, and returns the session to the pool.
        Concurrent calls to the same endpoint each check out a distinct session (up to the
        pool max), so a turn's tool fan-out runs in parallel. A failed session is discarded
        so the next call rebuilds it (handles idle drops / server restarts).
        """
        client, opened_ms = await self._pool_checkout(store, endpoint_name, cfg, mcp_timeout)
        timing["open_ms"] = opened_ms
        try:
            result = await do_send(client.session)
        except Exception:
            await self._pool_discard(store, endpoint_name, client)
            raise
        self._pool_checkin(store, endpoint_name, client)
        return result

    def _turn_store(self):
        """Get-or-create the per-turn session store keyed by the current event loop. Sessions
        in it are reused across this turn's concurrent tool calls and closed at turn end
        (_close_turn_pool). Created on the turn's own worker-thread loop (no cross-thread
        bridge), so requests run in parallel across gunicorn threads."""
        loop_id = id(asyncio.get_event_loop())
        with self._turn_pools_lock:
            return self._turn_pools.setdefault(loop_id, {})

    async def _close_turn_pool(self, loop_id):
        """Close every session opened for one turn's loop and drop the store. Called from the
        turn's own loop after the LLM/tool cycle completes, so no sessions leak per turn."""
        with self._turn_pools_lock:
            store = self._turn_pools.pop(loop_id, None)
        if not store:
            return
        for entry in store.values():
            for client in list(entry.get("clients", ())):
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    pass

    async def _call_mcp_tool(self, toolname, tool_input):
        # Domain describe/activate are handled locally by the DomainToolManager — they
        # inspect or change the live tool set and are never forwarded to an MCP endpoint.
        if self._domain_mgr is not None and toolname in DomainToolManager.HANDLED_TOOLS:
            key = getattr(self, "_current_session_id", None) or "_global"
            if toolname in DomainToolManager.MUTATING_TOOLS:
                # Reload the full data-domain tool set from the server before activating so
                # functions defined/promoted mid-session (fn_*) are visible immediately. The
                # server reloads annotations per /llm_tools call, so this stays current.
                await self._refresh_domain_tools()
            result = self._domain_mgr.handle(toolname, tool_input, key)
            if toolname in DomainToolManager.MUTATING_TOOLS:
                self._apply_active_tools(key)
                self._emit(
                    f"Data domains active: {', '.join(self._domain_mgr.active_domains(key)) or 'none'}",
                    status="Domains Updated",
                )
            return json.dumps(result)

        # Bypass: strip the control flag before forwarding — the server tool schema does
        # not declare it. When set, the full result is rendered directly in the UI and only
        # a small ack is returned to the LLM.
        bypass = False
        if isinstance(tool_input, dict) and "bypass" in tool_input:
            bypass = bool(tool_input.pop("bypass"))

        # Conversation recall is browser-scoped by a POST-filter so semantic ranking is
        # preserved (no Atlas index change needed). We over-fetch here, then drop any
        # candidate whose browser tag != this browser and truncate to the requested
        # limit below. Skips scoping when no browser id is known.
        conv_scope_uid = None
        conv_requested_limit = None
        if (
            isinstance(tool_input, dict)
            and toolname in ("memory_recall", "memory_query")
            and "conversation" in (tool_input.get("memory_types") or [])
        ):
            conv_scope_uid = getattr(self, "_current_user_id", None)
            if conv_scope_uid:
                conv_requested_limit = int(tool_input.get("limit") or 5)
                tool_input["limit"] = min(50, max(conv_requested_limit * 5, 25))

        endpoint_name, endpoint_tool_name = self._resolve_endpoint(toolname)
        cfg = self.mcp_endpoint_configs.get(endpoint_name)
        if cfg is None:
            raise RuntimeError(f"No config for endpoint '{endpoint_name}'")

        # Cache stable schema/index calls — they're invoked at the start of every
        # conversation and rarely change. Check cache before opening an MCP connection.
        cache_key = None
        if endpoint_tool_name in self._CACHED_TOOL_SUFFIXES:
            cache_key = SimpleCache.create_cache_key(toolname, tool_input)
            cached = await self._tool_response_cache.get(cache_key)
            if cached is not None:
                self._emit(f"Cached: {toolname}", status="Tool Cache")
                return cached
        # Agent tools run a full sub-agent loop; give them much more time.
        # All other tools use 60s (raises the default 30s httpx connect timeout).
        is_agent_tool = endpoint_name == "agent" or endpoint_tool_name in ("run_prompt",)
        mcp_timeout = 600 if is_agent_tool else 60
        outer_timeout = 620 if is_agent_tool else 90

        _timing = {"open_ms": -1, "send_ms": -1}

        async def _do_send(session):
            _t_send = time.monotonic()
            raw = await session.send_request(
                mt.ClientRequest(
                    mt.CallToolRequest(
                        params=mt.CallToolRequestParams(
                            name=endpoint_tool_name,
                            arguments=tool_input,
                        )
                    )
                ),
                mt.CallToolResult,
            )
            _timing["send_ms"] = int((time.monotonic() - _t_send) * 1000)
            if raw.content and hasattr(raw.content[0], "text"):
                return raw.content[0].text
            if raw.structuredContent is not None:
                return json.dumps(raw.structuredContent)
            return str(raw)

        # Session-reuse strategy (see _pool_mode). Never pooled for agent tools — a long
        # sub-agent run would hold a pooled session for the whole run.
        _mode = "off" if is_agent_tool else self._pool_mode

        async def _run():
            if _mode == "container":
                self._ensure_pool_loop()
                # Bridge onto the persistent background loop (where warm sessions live) and
                # await the result back on this turn's loop.
                cfut = asyncio.run_coroutine_threadsafe(
                    self._pool_send(self._pool_sessions, endpoint_name, cfg, mcp_timeout, _do_send, _timing),
                    self._pool_loop,
                )
                return await asyncio.wrap_future(cfut)
            if _mode == "turn":
                # Pool on THIS turn's own loop — no cross-thread bridge; closed at turn end.
                return await self._pool_send(self._turn_store(), endpoint_name, cfg, mcp_timeout, _do_send, _timing)
            # Default: a fresh client per call (its own initialize handshake).
            client = fastmcp.Client({"mcpServers": {endpoint_name: cfg}}, timeout=mcp_timeout)
            _t_open = time.monotonic()
            async with client:
                _timing["open_ms"] = int((time.monotonic() - _t_open) * 1000)
                return await _do_send(client.session)

        last_exc: Optional[Exception] = None
        # Agent tools are not retried — a second run_prompt call would spawn a
        # duplicate sub-agent.  All other tools retry up to 4 times.
        max_attempts = 1 if is_agent_tool else 4
        _t_total = time.monotonic()
        for attempt in range(max_attempts):
            try:
                result = await asyncio.wait_for(_run(), timeout=outer_timeout)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    wait = 1.0 * (attempt + 1)
                    logger.warning("MCP call to %r failed (attempt %d/%d): %s — retrying in %.1fs", toolname, attempt + 1, max_attempts, exc, wait)
                    await asyncio.sleep(wait)
        else:
            raise last_exc  # type: ignore[misc]

        # Transport-vs-server breakdown for this MCP tool call. session_open is the
        # per-call TLS + MCP initialize handshake (pure setup/transport); send_request
        # is the tools/call round-trip (network + server-side execution); overhead is
        # retry sleeps / wait_for wrapping. Total is the webui-side wall clock.
        _total_ms = int((time.monotonic() - _t_total) * 1000)
        _overhead_ms = max(0, _total_ms - max(0, _timing["open_ms"]) - max(0, _timing["send_ms"]))
        logger.info(
            "MCP timing %s -> endpoint=%s: total=%d ms | session_open=%d ms (setup/transport) | "
            "send_request=%d ms (call round-trip + server) | overhead/retry=%d ms",
            toolname, endpoint_name, _total_ms, _timing["open_ms"], _timing["send_ms"], _overhead_ms,
        )

        # Store result in cache for cacheable tools.
        if cache_key is not None:
            await self._tool_response_cache.set(cache_key, result, ttl=self._CACHED_TOOL_TTL)

        # Browser-scope conversation recall results (semantic order preserved).
        if conv_scope_uid:
            result = self._browser_scope_conversation_results(result, conv_scope_uid, conv_requested_limit)

        if bypass:
            return self._capture_bypass(endpoint_name, endpoint_tool_name, result)
        return result

    def _resolve_endpoint(self, toolname):
        for candidate in sorted(self.mcp_endpoints, key=len, reverse=True):
            if toolname.startswith(f"{candidate}_"):
                return candidate, toolname[len(candidate) + 1:]
        if len(self.mcp_endpoints) == 1:
            return self.mcp_endpoints[0], toolname
        raise RuntimeError(
            f"Cannot resolve endpoint for '{toolname}'. Known: {self.mcp_endpoints}"
        )

    # ------------------------------------------------------------------
    # Query / history
    # ------------------------------------------------------------------

    MAX_CONTEXT_TOKENS = 200_000
    WARN_RATIO = 0.70          # emit warning above this fraction
    RESERVED_TOKENS = 20_000   # headroom for next response
    MAX_HISTORY_MSGS = 20      # hard cap on message count
    CHECKPOINT_TURN_CADENCE = 2  # write the checkpoint every Nth completed turn

    async def _prefetch_session_context(self, username: str) -> str:
        """Pre-fetch recent sessions and user preferences before the first LLM turn.

        Runs two parallel MCP tool calls and returns a formatted Markdown block
        to inject into the system prompt. Falls back gracefully on any error.
        """
        from datetime import datetime, timezone
        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sessions_text = ""
        prefs_text = ""
        try:
            sessions_res, prefs_res = await asyncio.gather(
                self._call_mcp_tool("memory_list_sessions", {"filter": {"username": username}, "limit": 5}),
                self._call_mcp_tool("memory_recall", {
                    "query": "preferences working style tools interests",
                    "username": username,
                    "memory_types": ["user_preference"],
                    "limit": 5,
                }),
                return_exceptions=True,
            )
            if not isinstance(sessions_res, Exception):
                try:
                    data = json.loads(sessions_res) if isinstance(sessions_res, str) else sessions_res
                    sessions = (data or {}).get("sessions", (data or {}).get("results", []))
                    n = len(sessions or [])
                    if n:
                        sessions_text = (
                            "### Prior Session Availability\n"
                            f"- This is a RETURNING user with at least {n} prior session(s). "
                            "Full recaps are NOT preloaded. If the user references earlier work "
                            "('what were we doing', 'go back to…'), fetch on demand via "
                            "memory_list_sessions or memory_recall(memory_types=[\"conversation\"], "
                            "query=…). Otherwise do not load them."
                        )
                    else:
                        sessions_text = (
                            "### Prior Session Availability\n"
                            "- No prior sessions found — treat this as a NEW user."
                        )
                except Exception:
                    pass
            if not isinstance(prefs_res, Exception):
                try:
                    data = json.loads(prefs_res) if isinstance(prefs_res, str) else prefs_res
                    prefs = (data or {}).get("results", [])
                    if prefs:
                        lines = [f"- {p.get('content', '')[:150]}" for p in prefs[:5]]
                        prefs_text = "### User Preferences\n" + "\n".join(lines)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Session pre-fetch error: %s", exc)
        header = (
            f"## Pre-loaded Session Context\n"
            f"**context_loaded:** true  \n"
            f"**username:** {username}  \n"
            f"**fetched_at:** {fetched_at}"
        )
        sections = [header]
        if sessions_text:
            sections.append(sessions_text)
        if prefs_text:
            sections.append(prefs_text)
        if len(sections) == 1:
            sections.append("*(no prior session data found)*")
        return "\n\n".join(sections)

    def _service_agent_name(self) -> str:
        """The agent_name the webui presents to the MCP/memory backend, decoded from
        the outbound service token (e.g. kanopy-staging / kanopy-prod). Cached.
        This is the app's MACHINE credential — never the human user."""
        name = getattr(self, "_service_agent_name_cache", None)
        if name:
            return name
        name = "the app service account"
        try:
            import jwt as _jwt
            payload = _jwt.decode(settings.get_auth_token(), options={"verify_signature": False})
            name = (payload.get("agent_name") or payload.get("cognito:username")
                    or payload.get("username") or payload.get("client_id")
                    or payload.get("sub") or name)
        except Exception as exc:
            logger.warning("Could not decode service agent name from AUTH_TOKEN: %s", exc)
        self._service_agent_name_cache = name
        return name

    def query_with_mcp_tools(self, request: "QueryRequest", emit=None) -> "QueryResponse":
        emit_fn = emit or self._emit
        # Store session_id so sub-agents can inherit it without the LLM needing to pass it.
        self._current_session_id = request.session_id
        # Identity for this turn. When auth is enabled this is the verified user
        # (token sub); in the OSS/disabled build it's the per-browser localStorage id.
        # Either way it is NOT the shared username. Used to scope conversation recall.
        self._current_user_id = request.user_id
        # Whether this turn's identity was VERIFIED by the auth provider (vs self-declared);
        # surfaced to the agent so it never conflates the user with the service token.
        self._identity_verified = bool(getattr(request, "identity_verified", False))
        # Fresh per-turn buffer for bypass=true tool results.
        self._bypass_items = []
        # History is owned entirely by the frontend — use only what was sent in the request.
        incoming_history = list(request.history or [])
        history = self._trim_history(incoming_history)

        is_new_session = not history

        msgs = list(history)
        user_content = [{"text": request.input}]
        self._inject_token_warning(msgs, user_content)
        msgs.append({"role": "user", "content": user_content})
        self.llm_client.message_handler = emit_fn

        # Re-discover tools if the cache is stale.
        age = (time.monotonic() - self._tools_fetched_at) if self._tools_fetched_at else float("inf")
        if age >= self.TOOLS_CACHE_TTL_SECONDS:
            logger.info("Tools cache is stale (%.0fs old) — refreshing before query.", age)
            self._discover_tools(emit=emit_fn)

        _ctx_block: Optional[str] = None
        _ctx_prefetched = False

        async def _invoke():
            nonlocal _ctx_block, _ctx_prefetched
            # Once per conversation: pre-fetch user context on the first message.
            prefetch_name = request.username or request.user_id
            if is_new_session and prefetch_name and not _ctx_prefetched:
                try:
                    _ctx_block = await self._prefetch_session_context(prefetch_name)
                    logger.info("Session context pre-fetched for user=%s (%d chars)", prefetch_name, len(_ctx_block))
                except Exception as exc:
                    logger.warning("Session pre-fetch failed: %s", exc)
                    _ctx_block = "## Pre-loaded Session Context\n**context_loaded:** false"
                _ctx_prefetched = True

            effective_system = list(self._base_system_prompt)
            # Always disambiguate the USER identity from the app's service token so the
            # agent never treats a verified user as the machine service credential.
            _uname = request.username or request.user_id or "unknown"
            _svc = self._service_agent_name()
            if self._identity_verified:
                effective_system.append({"text":
                    f"USER IDENTITY: '{_uname}' is AUTHENTICATED and VERIFIED by the auth provider "
                    f"(identity_verified: true). Trust this identity: personalize and recall "
                    f"their sessions/preferences WITHOUT asking for their name. The service token "
                    f"'{_svc}' is only the app's machine credential to the memory backend, NOT the user."})
            else:
                effective_system.append({"text":
                    f"USER IDENTITY: '{_uname}' is SELF-DECLARED (identity_verified: false; not "
                    f"behind a verifying auth provider). Confirm the name before recalling personal sessions/"
                    f"preferences. The service token '{_svc}' is the app's machine credential, not the user."})
            if _ctx_block:
                effective_system.append({"text": _ctx_block})
            # Mode 1: load the deterministic checkpoint for this session and inject it.
            # Only when resuming (new/empty history — e.g. after a page refresh that
            # minted an empty round-trip) or when this process has checkpointed this
            # session before, so short conversations don't pay a per-turn fetch.
            if request.session_id and (is_new_session or request.session_id in self._checkpoint_ids):
                _ckpt_block = await self._load_checkpoint(request.session_id)
                if _ckpt_block:
                    effective_system.append({"text": _ckpt_block})
            self.llm_client.system = effective_system
            # UI-authoritative function stage: gunicorn runs multiple workers, each with its
            # own _function_stage. The UI sends the stage it is displaying on every query, so
            # whichever worker serves this turn converges to it here (refreshing the domain
            # registry IN PLACE at the new stage, preserving active domains). This keeps the
            # dev/prod button and what the model actually sees in sync regardless of worker.
            req_stage = getattr(request, "function_stage", None)
            if req_stage in ("dev", "prod") and req_stage != self._function_stage:
                logger.info("Function stage sync: worker %s -> %s (from UI)", self._function_stage, req_stage)
                self._function_stage = req_stage
                await self._refresh_domain_tools()
            # Reset the live tool set to this session's active tools (core + activate
            # stub + any domains already activated earlier in this conversation).
            self._apply_active_tools(self._current_session_id)
            try:
                return await self.llm_client.invoke_bedrock_with_tools_text(messages=msgs)
            finally:
                # Close any per-turn pooled MCP sessions opened on this loop (turn mode only;
                # no-op otherwise) so sessions don't leak across turns.
                if self._pool_mode == "turn":
                    await self._close_turn_pool(id(asyncio.get_event_loop()))

        result = asyncio.run(_invoke())
        if self._looks_like_no_tools_error(result):
            emit_fn(
                "MCP tools were unavailable. Reloading backend tool configuration and retrying...",
                status="Recovering",
            )
            recovered = self._rediscover_tools(emit=emit_fn)
            if recovered:
                result = asyncio.run(_invoke())

        if self._looks_like_no_tools_error(result):
            # Keep this backend recovery detail out of the user-visible response.
            result = {
                **result,
                "error": "Tool configuration is reloading. Please retry your request in a few seconds.",
            }

        updated_history = result.get("history", msgs)
        answer = result.get("response_text", "")
        jsondata = self._apply_bypass_jsondata(result.get("jsondata"))
        tool_timings = result.get("tool_timings") or {}
        corrupt = bool(result.get("clear_history"))

        # Emit token usage summary after every response
        usage = result.get("usage") or {}
        in_tok = int(usage.get("inputTokens", 0) or 0)
        out_tok = int(usage.get("outputTokens", 0) or 0)
        total = in_tok + out_tok
        pct = round(total / self.MAX_CONTEXT_TOKENS * 100, 1)
        emit_fn(
            f"Tokens — in: {in_tok:,}  out: {out_tok:,}  total: {total:,}  ({pct}% of {self.MAX_CONTEXT_TOKENS // 1000}k)",
            status="Token Usage",
        )
        self._record_usage_metrics(request, in_tok, out_tok)

        content = {"text": answer, "jsondata": jsondata, "tool_timings": tool_timings}
        if corrupt:
            emit_fn("Corrupt tool state detected — cleaning history and retrying...", status="Recovering")
            # Trigger 2: harvest-before-clear. The extractor works on corrupt history
            # (it ignores toolUse/toolResult pairing), so fold the delta into the
            # checkpoint before the history is sanitized/cleared and lost.
            self._schedule_checkpoint_write(request, list(updated_history or []))
            sanitized = self._sanitize_history(list(updated_history or []))
            # Rebuild msgs with sanitized history + original user input and retry once.
            msgs = sanitized
            msgs.append({"role": "user", "content": user_content})
            result = asyncio.run(_invoke())
            updated_history = result.get("history", msgs)
            answer = result.get("response_text", "")
            jsondata = self._apply_bypass_jsondata(result.get("jsondata"))
            content = {"text": answer, "jsondata": jsondata}
            if result.get("error"):
                return QueryResponse(
                    status="Error",
                    message=result["error"],
                    error=result["error"],
                    history=updated_history,
                )

        # Fold the just-completed turn — including the assistant's answer — into the
        # session checkpoint. Fires after the response so the current answer is
        # recorded now, not lagged to the next user submission.
        self._checkpoint_current_turn(request, updated_history)
        return QueryResponse(
            status="Query Completed", message="Completed",
            content=content,
            history=updated_history,
        )

    def _record_usage_metrics(self, request: "QueryRequest", in_tok: int, out_tok: int) -> None:
        """Best-effort: blind-upsert per-user token usage to the MCP /metrics endpoint.

        Fired in a background thread so it never blocks or breaks a user response.
        Uniqueness is the (browserId, username, ip) triple — the server increments an
        existing row when all three match, otherwise it creates a new one.
        """
        browser_id = request.user_id or ""
        username = request.username or ""
        ip = request.ip or ""
        if not (browser_id or username or ip):
            return
        # Only the real activated data domains for this session. memory/agent are
        # always-on core and are intentionally excluded — turns that touch no data
        # domain are not domain-tracked (the server skips them).
        data_domains: List[str] = []
        if self._domain_mgr is not None:
            data_domains = list(self._domain_mgr.active_domains(request.session_id or "_global"))
        payload = {
            "browserId": browser_id,
            "username": username,
            "ip": ip,
            "sessionId": request.session_id or "",
            "inputTokens": in_tok,
            "outputTokens": out_tok,
            "dataDomains": data_domains,
        }

        def _post() -> None:
            try:
                resp = requests.post(
                    f"{settings.mongo_mcp_root}/metrics",
                    json=payload, headers=self._headers, timeout=10,
                )
                logger.debug(
                    "[METRICS webui] POST %s/metrics -> %s dataDomains=%s resp=%s",
                    settings.mongo_mcp_root, resp.status_code, payload.get("dataDomains"),
                    (resp.text or "")[:300],
                )
            except Exception as exc:
                logger.debug("[METRICS webui] post to %s/metrics failed: %s",
                             settings.mongo_mcp_root, exc)

        threading.Thread(target=_post, daemon=True).start()

    # ------------------------------------------------------------------
    # Conversation checkpointing (deterministic, no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoint_browser_tag(user_id: Optional[str]) -> Optional[str]:
        """Per-user isolation tag. user_id is the verified auth-provider sub when auth
        is enabled, else the per-browser localStorage id; the shared username
        (e.g. 'demo-user') must NOT be the isolation key."""
        return f"browser:{user_id}" if user_id else None

    @staticmethod
    def _browser_scope_conversation_results(result, user_id: str, limit: Optional[int]):
        """Drop conversation-recall results that don't belong to this user, keeping
        semantic order, then truncate to the originally-requested limit. Best-effort:
        returns the input unchanged if it can't be parsed."""
        tag = f"browser:{user_id}"
        try:
            data = json.loads(result) if isinstance(result, str) else result
        except Exception:
            return result
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return result
        kept = [
            d for d in data["results"]
            if isinstance(d, dict) and tag in (d.get("tags") or [])
        ]
        if limit:
            kept = kept[:limit]
        data["results"] = kept
        if "count" in data:
            data["count"] = len(kept)
        return json.dumps(data)

    async def _load_checkpoint(self, session_id: Optional[str]) -> Optional[str]:
        """Deterministically fetch THE checkpoint for a session (Mode 1) and render
        it as a system-prompt block. Direct filter fetch — NOT vector recall."""
        if not session_id:
            return None
        try:
            raw = await self._call_mcp_tool("memory_query", {
                "filter": {"session_id": session_id, "memory_type": ckpt.MEMORY_TYPE},
                "scope": "episodic",
                "limit": 1,
            })
            data = json.loads(raw) if isinstance(raw, str) else raw
            results = (data or {}).get("results", [])
            if not results:
                return None
            doc = results[0]
            doc_id = doc.get("_id") or doc.get("id")
            if doc_id:
                self._checkpoint_ids[session_id] = str(doc_id)
            snap = doc.get("payload") or {}
            if not snap.get("turn_count"):
                return None
            return ckpt.render_context_block(snap)
        except Exception as exc:
            logger.debug("checkpoint load failed for %s: %s", session_id, exc)
            return None

    async def _write_checkpoint(self, session_id: Optional[str], user_id: Optional[str],
                                username: Optional[str], messages: list) -> None:
        """Extract a deterministic snapshot from *messages*, merge into any existing
        checkpoint for the session, and upsert it in place (one doc per session)."""
        if not session_id or not messages:
            return
        try:
            delta = ckpt.extract_delta(messages)
            if not (delta.get("open_threads") or delta.get("answer_leads") or delta.get("pinned_facts")):
                return

            # Single lookup: get the existing checkpoint doc for this session (if any)
            # so we merge into its payload and update in place rather than duplicate.
            existing_id = None
            prev_snap: dict = {}
            prev_importance = 0.0
            raw = await self._call_mcp_tool("memory_query", {
                "filter": {"session_id": session_id, "memory_type": ckpt.MEMORY_TYPE},
                "scope": "episodic",
                "limit": 1,
            })
            data = json.loads(raw) if isinstance(raw, str) else raw
            results = (data or {}).get("results", [])
            if results:
                existing_id = results[0].get("_id") or results[0].get("id")
                prev_snap = results[0].get("payload") or {}
                try:
                    prev_importance = float(results[0].get("importance") or 0.0)
                except (TypeError, ValueError):
                    prev_importance = 0.0

            snap = ckpt.merge_snapshot(prev_snap, delta)
            content = ckpt.render_content(snap)
            tag = self._checkpoint_browser_tag(user_id)
            intake_args: dict = {
                "content": content,
                "memory_type": ckpt.MEMORY_TYPE,
                # Base 0.5, but preserve any value an explicit 👍/👎 feedback has set
                # (0.9 up / 0.2 down) so later writes don't reset the user's signal.
                "importance": prev_importance if prev_importance > 0 else 0.5,
                # Low decay: checkpoint is a durable cross-session archive, kept out of
                # default recall by the DEFAULT_EXCLUDED_MEMORY_TYPES $nin filter instead.
                "decay_rate": 0.001,
                "session_id": session_id,
                "payload": snap,
                "tags": [tag] if tag else [],
            }
            if username:
                intake_args["username"] = username
            if existing_id:
                intake_args["_id"] = str(existing_id)
            res = await self._call_mcp_tool("memory_intake", intake_args)
            data = json.loads(res) if isinstance(res, str) else res
            new_id = (data or {}).get("id")
            if new_id:
                self._checkpoint_ids[session_id] = str(new_id)
        except Exception as exc:
            logger.debug("checkpoint write failed for %s: %s", session_id, exc)

    async def _bump_checkpoint_importance(self, session_id: Optional[str], importance: float) -> None:
        """Set the durable importance of a session's conversation checkpoint in place
        (payload-only update; no re-embed). Driven by explicit 👍/👎 feedback so
        validated answers rank higher on later recall."""
        if not session_id:
            return
        try:
            cid = self._checkpoint_ids.get(session_id)
            if not cid:
                raw = await self._call_mcp_tool("memory_query", {
                    "filter": {"session_id": session_id, "memory_type": ckpt.MEMORY_TYPE},
                    "scope": "episodic",
                    "limit": 1,
                })
                data = json.loads(raw) if isinstance(raw, str) else raw
                results = (data or {}).get("results", [])
                if not results:
                    return
                cid = results[0].get("_id") or results[0].get("id")
            if not cid:
                return
            await self._call_mcp_tool("memory_intake", {
                "_id": str(cid),
                "memory_type": ckpt.MEMORY_TYPE,
                "importance": importance,
            })
            self._checkpoint_ids[session_id] = str(cid)
        except Exception as exc:
            logger.debug("importance bump failed for %s: %s", session_id, exc)

    def _checkpoint_current_turn(self, request: "QueryRequest", updated_history: list) -> None:
        """Fold recently-completed turns into the session checkpoint AFTER the assistant
        response, on a 2-turn cadence: every Nth completed turn triggers a write that
        folds the current history window (the idempotent merge dedups any overlap).
        Odd tail turns are flushed by the unload beacon -> /checkpoint/finalize. The
        quality gate in _write_checkpoint skips non-substantive turns."""
        sid = getattr(request, "session_id", None)
        if not sid or not updated_history:
            return
        n = self._checkpoint_turn_count.get(sid, 0) + 1
        self._checkpoint_turn_count[sid] = n
        if n % self.CHECKPOINT_TURN_CADENCE != 0:
            return  # write every Nth completed turn; the beacon flushes the tail
        self._schedule_checkpoint_write(request, list(updated_history))

    def _schedule_checkpoint_write(self, request: "QueryRequest", messages: list) -> None:
        """Fire-and-forget checkpoint write on a background thread so it never blocks
        or breaks a user response (mirrors the metrics pattern)."""
        if not messages or not getattr(request, "session_id", None):
            return
        sid = request.session_id
        uid = request.user_id
        uname = request.username
        snapshot_msgs = list(messages)

        def _run() -> None:
            try:
                asyncio.run(self._write_checkpoint(sid, uid, uname, snapshot_msgs))
            except Exception as exc:
                logger.debug("checkpoint thread error for %s: %s", sid, exc)

        threading.Thread(target=_run, daemon=True).start()

    def finalize_checkpoint(self, session_id: str, user_id: Optional[str],
                            username: Optional[str], history: Optional[List[Any]]) -> "QueryResponse":
        """Fold any remaining live turns into the session's checkpoint before it is
        abandoned (e.g. on 'clear chat'), so the archived checkpoint is complete."""
        if session_id and history:
            req = QueryRequest(input="", history=history, user_id=user_id,
                               username=username, session_id=session_id)
            self._schedule_checkpoint_write(req, list(history))
        return QueryResponse(status="Finalized", message="Checkpoint finalized")

    # ------------------------------------------------------------------
    # History trimming / sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_history(history: list) -> list:
        """Strip incomplete tool-use turns, keeping only clean text-only turns."""
        clean = []
        for msg in history:
            content = msg.get("content", [])
            if not isinstance(content, list):
                clean.append(msg)
                continue
            text_blocks = [
                b for b in content
                if isinstance(b, dict) and "text" in b
                and "toolUse" not in b and "toolResult" not in b
            ]
            if text_blocks:
                clean.append({"role": msg["role"], "content": text_blocks})
        return clean

    @staticmethod
    def _estimate_tokens(obj) -> int:
        if not obj:
            return 0
        if isinstance(obj, str):
            return max(1, len(obj) // 4)
        return max(1, len(json.dumps(obj, default=str)) // 4)

    def _estimate_history_tokens(self, history: list) -> int:
        total = 0
        for msg in history:
            total += 6
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    total += self._estimate_tokens(block)
            else:
                total += self._estimate_tokens(content)
        return total

    def _trim_history(self, history: list) -> list:
        """Drop oldest whole turns until under token budget and message cap."""
        if not history:
            return history
        token_limit = self.MAX_CONTEXT_TOKENS - self.RESERVED_TOKENS

        turns, current = [], []
        for msg in history:
            is_new_turn = (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
                and msg["content"]
                and not all(isinstance(b, dict) and "toolResult" in b for b in msg["content"])
            )
            if is_new_turn and current:
                turns.append(current)
                current = [msg]
            else:
                current.append(msg)
        if current:
            turns.append(current)

        def flatten(t):
            return [m for turn in t for m in turn]

        while len(flatten(turns)) > self.MAX_HISTORY_MSGS and len(turns) > 1:
            turns = turns[1:]

        while len(turns) > 1 and self._estimate_history_tokens(flatten(turns)) > token_limit:
            turns = turns[1:]

        result = flatten(turns)
        # Find the first clean user message (not a toolResult-only message)
        # to avoid starting history mid tool-use cycle which Bedrock rejects.
        for i, msg in enumerate(result):
            if msg.get("role") == "user":
                result = result[i:]
                break
        else:
            return []
        # Enforce Bedrock's toolUse/toolResult adjacency invariant on the trimmed
        # history: strips orphaned toolResults, dangling toolUses, and empty-content
        # messages. The canonicaliser lives in the BedrockClient base class so this
        # logic has a single source of truth (webui + agentcore runtimes all share it).
        try:
            self.llm_client._canonicalize_tool_cycles(result)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("tool-cycle canonicalisation skipped: %s", exc)
        return result

    def _inject_token_warning(self, history: list, user_content: list) -> None:
        """Append a context-pressure warning when nearing token limit."""
        estimated = self._estimate_history_tokens(history)
        if estimated < self.MAX_CONTEXT_TOKENS * self.WARN_RATIO:
            return
        pct = int(estimated / self.MAX_CONTEXT_TOKENS * 100)
        user_content.append({"text": (
            f"\n\n[Context Warning] Conversation history is at ~{pct}% of the "
            f"{self.MAX_CONTEXT_TOKENS // 1000}k token limit. "
            "Please review the conversation so far and use the memory intake tool "
            "to save a concise summary of key facts, decisions, and any important data "
            "before it is lost to trimming. Keep your response concise. "
            "Older turns will be dropped on the next turn if the limit is not reduced."
        )})

    def reset(self) -> "QueryResponse":
        """Full reset: drain queue, drop cached MCP clients, re-discover tools."""
        try:
            while True:
                self._message_queue.get_nowait()
        except queue.Empty:
            pass
        self.endpoint_clients = {}
        self._discover_tools()
        return QueryResponse(status="Reset", message="Application reset — tools reloaded", history=[], clear_history=True)

    def get_mcp_config(self) -> dict:
        return {
            "endpoints": self.mcp_endpoints or [],
            "collection_info": self.mongo_collection_info or {},
        }

    def save_pattern(self, user_id: str, session_id: str, history: Optional[List[Any]] = None) -> "QueryResponse":
        message = (
            f"The user marked this conversation as a useful pattern worth saving. "
            f"Please store the key question, approach, and answer from this session "
            f"(session_id: `{session_id}`, user_id: `{user_id}`) to memory as a "
            f"'pattern:query' memory_type entry in the semantic collection with "
            f"high importance (0.9). Include the user's original question and the "
            f"approach used to answer it so it can be reused in future sessions."
        )
        req = QueryRequest(input=message, history=history or [], user_id=user_id, session_id=session_id)
        return self.query_with_mcp_tools(req)

    def record_feedback(self, user_id: str, session_id: str, feedback: str, history: Optional[List[Any]] = None) -> "QueryResponse":
        # Explicit user signal → adjust the durable value of this session's
        # conversation checkpoint so validated answers rank higher on recall.
        try:
            _imp = 0.9 if feedback == "positive" else 0.2
            asyncio.run(self._bump_checkpoint_importance(session_id, _imp))
        except Exception as exc:
            logger.debug("feedback importance bump failed for %s: %s", session_id, exc)
        if feedback == "positive":
            message = (
                f"\U0001f44d The user confirmed this approach **worked** for session `{session_id}` "
                f"(user: {user_id}). "
                f"Please review the conversation history and save the validated pattern to memory. "
                f"Use the memory intake tool twice:\n"
                f"1. A `pattern:query` entry (importance 0.9, semantic scope) containing: "
                f"the user's original question, the approach/strategy used to answer it, "
                f"and the key result — so it can be recalled and reused in future sessions.\n"
                f"2. A `feedback:positive` entry (importance 0.6) recording that session "
                f"`{session_id}` received positive feedback from user `{user_id}`."
            )
        else:
            message = (
                f"\U0001f44e The user indicated this approach **did not work** for session `{session_id}` "
                f"(user: {user_id}). "
                f"Please store a `feedback:negative` memory entry (importance 0.7) that records "
                f"what was tried, why it may have failed, and what to avoid in similar situations. "
                f"Include session_id `{session_id}` and user `{user_id}` in the entry."
            )
        req = QueryRequest(input=message, history=history or [], user_id=user_id, session_id=session_id)
        return self.query_with_mcp_tools(req)

    @staticmethod
    def _trim_history_for_ui(history, max_text_len=2000):
        if not history:
            return history
        trimmed = []
        for msg in history:
            content = msg.get("content")
            if not isinstance(content, list):
                trimmed.append(msg)
                continue
            needs_trim = any(
                isinstance(b, dict) and "toolResult" in b
                and any(
                    isinstance(c, dict) and len(c.get("text", "")) > max_text_len
                    for c in b["toolResult"].get("content", [])
                )
                for b in content
            )
            if not needs_trim:
                trimmed.append(msg)
                continue
            new_content = []
            for b in content:
                if isinstance(b, dict) and "toolResult" in b:
                    tr = b["toolResult"]
                    new_parts = [
                        {"text": c["text"][:max_text_len] + f"... [truncated {len(c['text'])} chars]"}
                        if isinstance(c, dict) and "text" in c and len(c["text"]) > max_text_len
                        else c
                        for c in tr.get("content", [])
                    ]
                    new_content.append({"toolResult": {**tr, "content": new_parts}})
                else:
                    new_content.append(b)
            trimmed.append({**msg, "content": new_content})
        return trimmed
