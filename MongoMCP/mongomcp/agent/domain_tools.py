"""
DomainToolManager — progressive tool disclosure across multiple data domains.

Holds the full per-domain tool catalog but exposes to the LLM only an always-on
*core* tool set (e.g. memory/agent tools) plus two stub tools:

  - describe_data_domain(domains?)  — read-only: list a domain's tools + descriptions.
  - activate_data_domain(domains, mode) — load/unload a domain's query tools into the
    live tool set. mode ∈ {add (default), replace, remove, clear}.

The active domain set is tracked per *key* (e.g. session_id) so activation is sticky
within a conversation and isolated across conversations.

Host-agnostic: this class never touches a Bedrock client and never emits progress.
The host wires it into its own tool dispatch and reconfigures its client from
``active_tool_set(key)`` after a mutating call:

    mgr = DomainToolManager(core_tools, registry, descriptions)
    client.configure_tools(mgr.base_tool_set(), dispatch)      # at load
    ...
    # inside the dispatch handler:
    if name in DomainToolManager.HANDLED_TOOLS:
        result = mgr.handle(name, tool_input, session_key)
        if name in DomainToolManager.MUTATING_TOOLS:
            client.configure_tools(mgr.active_tool_set(session_key), dispatch)
        return json.dumps(result)

The invoke loop must recompute its toolConfig each iteration for mid-conversation
activation to take effect on the next model turn.
"""

import json
from typing import Any, Dict, List, Optional


class DomainToolManager:
    HANDLED_TOOLS = frozenset({"activate_data_domain", "describe_data_domain"})
    MUTATING_TOOLS = frozenset({"activate_data_domain"})
    _MODES = frozenset({"add", "replace", "remove", "clear"})

    def __init__(
        self,
        core_tools: List[dict],
        registry: Dict[str, List[dict]],
        descriptions: Optional[Dict[str, str]] = None,
    ):
        """
        core_tools:   always-on Bedrock toolSpecs (never held back).
        registry:     {domain_name: [Bedrock toolSpec, ...]} — held back until activated.
        descriptions: {domain_name: module description} for the stub/catalog text.
        """
        self._core_tools = list(core_tools)
        self._registry = dict(registry)
        self._descriptions = dict(descriptions or {})
        self._active: Dict[str, set] = {}
        # Stubs are only meaningful when there is at least one activatable domain.
        self._stubs = self._build_stub_specs() if self._registry else []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def domains(self) -> List[str]:
        return list(self._registry.keys())

    def base_tool_set(self) -> List[dict]:
        """Core tools + the describe/activate stubs (no data-domain tools)."""
        return self._core_tools + self._stubs

    def active_tool_set(self, key: Optional[str]) -> List[dict]:
        """Core + stubs + the domains this key has activated (de-duplicated, order-stable)."""
        tools = list(self.base_tool_set())
        seen = {t.get("toolSpec", {}).get("name") for t in tools}
        for domain in sorted(self._active.get(key or "_global", set())):
            for spec in self._registry.get(domain, []):
                nm = spec.get("toolSpec", {}).get("name")
                if nm and nm not in seen:
                    tools.append(spec)
                    seen.add(nm)
        return tools

    def active_domains(self, key: Optional[str]) -> List[str]:
        return sorted(self._active.get(key or "_global", set()))

    def set_domain_tools(
        self, name: str, tools: List[dict], description: Optional[str] = None
    ) -> None:
        """Replace the cached tool specs for a domain.

        Called by the host to refresh a domain mid-session (e.g. after a dynamic ``fn_*``
        query function is defined/promoted server-side) so newly-added tools become
        visible on the next activation without a full re-discovery. Rebuilds the
        describe/activate stubs if this is the first domain ever registered.
        """
        had_registry = bool(self._registry)
        self._registry[name] = list(tools)
        if description is not None:
            self._descriptions[name] = description
        if not had_registry:
            self._stubs = self._build_stub_specs()

    def domain_catalog_lines(self) -> str:
        """A '- name: description' line per domain, for a system-prompt listing."""
        return "\n".join(
            f"- {d}: {self._descriptions.get(d, '') or '(no description)'}"
            for d in self._registry
        )

    def handle(self, name: str, tool_input: Any, key: Optional[str]) -> dict:
        """Handle a stub tool call. Returns a result dict (host serializes to JSON)."""
        if name == "activate_data_domain":
            return self._activate(tool_input, key)
        if name == "describe_data_domain":
            return self._describe(tool_input)
        return {"error": f"DomainToolManager cannot handle tool '{name}'"}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _activate(self, tool_input: Any, key: Optional[str]) -> dict:
        domains = tool_input.get("domains") if isinstance(tool_input, dict) else None
        mode = (tool_input.get("mode") if isinstance(tool_input, dict) else None) or "add"
        mode = str(mode).strip().lower()
        if mode not in self._MODES:
            mode = "add"
        if isinstance(domains, str):
            domains = [domains]
        domains = [str(d).strip() for d in (domains or []) if str(d).strip()]

        active = self._active.setdefault(key or "_global", set())
        requested, unknown = [], []
        for d in domains:
            (requested if d in self._registry else unknown).append(d)

        if mode == "clear":
            active.clear()
        elif mode == "replace":
            active.clear()
            active.update(requested)
        elif mode == "remove":
            for d in requested:
                active.discard(d)
        else:  # add
            active.update(requested)

        active_list = sorted(active)
        result: dict = {
            "mode": mode,
            "active_domains": active_list,
            # Full tool detail (name + description) for the active domains, so a separate
            # describe_data_domain call is unnecessary before activating.
            "domains": {d: self._domain_detail(d) for d in active_list},
        }
        if unknown:
            result["unknown_domains"] = unknown
            result["known_domains"] = list(self._registry.keys())
        return result

    def _domain_detail(self, domain: str) -> dict:
        """{description, tools:[{name, description}]} for one domain."""
        return {
            "description": self._descriptions.get(domain, ""),
            "tools": [
                {
                    "name": s.get("toolSpec", {}).get("name"),
                    "description": s.get("toolSpec", {}).get("description", ""),
                }
                for s in self._registry.get(domain, [])
            ],
        }

    def _describe(self, tool_input: Any) -> dict:
        domains = tool_input.get("domains") if isinstance(tool_input, dict) else None
        if isinstance(domains, str):
            domains = [domains]
        domains = [str(d).strip() for d in (domains or []) if str(d).strip()]
        if not domains:
            domains = list(self._registry.keys())

        described: dict = {}
        unknown = []
        for d in domains:
            if d not in self._registry:
                unknown.append(d)
                continue
            described[d] = self._domain_detail(d)
        result: dict = {"domains": described}
        if unknown:
            result["unknown_domains"] = unknown
            result["known_domains"] = list(self._registry.keys())
        return result

    def _build_stub_specs(self) -> List[dict]:
        data_domains = list(self._registry.keys())
        example = json.dumps(data_domains[:2]) if data_domains else "[]"
        describe = {
            "toolSpec": {
                "name": "describe_data_domain",
                "description": (
                    "Inspect one or more data domains WITHOUT loading their tools. Returns "
                    "each tool's name and description. Usually optional — activate_data_domain "
                    "already returns this for the domains it loads; use describe only to look "
                    "at a domain before deciding whether to activate it. Available domains: "
                    + ", ".join(data_domains) + ". Omit 'domains' to describe every domain."
                ),
                "inputSchema": {"json": {
                    "type": "object",
                    "properties": {
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Data domain name(s) to describe, e.g. {example}. Omit for all.",
                        }
                    },
                }},
            }
        }
        activate = {
            "toolSpec": {
                "name": "activate_data_domain",
                "description": (
                    "Load one or more data domains' query tools into your available tools and "
                    "return those tools (name + description) — so you can activate directly "
                    "without a separate describe_data_domain call. Available domains: "
                    + ", ".join(data_domains) + ". 'mode': 'add' (default) adds the given "
                    "domains to those already active; 'replace' makes them the ONLY active "
                    "domains; 'remove' unloads the given domains; 'clear' unloads ALL data "
                    "domains (domains is ignored). Call '<domain>_get_collection_info' before "
                    "querying ONLY when you need schema/index/field names (e.g. a raw "
                    "aggregate_query or building filters) or on first contact with an "
                    "unfamiliar collection; skip it when running a pre-built vector/hybrid "
                    "search on a known corpus."
                ),
                "inputSchema": {"json": {
                    "type": "object",
                    "properties": {
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Data domain name(s) to act on, e.g. {example}. Pass [] when mode='clear'.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["add", "replace", "remove", "clear"],
                            "description": "How to change the active data-domain set. Default 'add'.",
                        },
                    },
                    "required": ["domains"],
                }},
            }
        }
        return [describe, activate]
