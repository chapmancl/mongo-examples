import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ToolRouter:
    """Select a subset of Bedrock toolSpecs relevant to a question.

    Supports two routing strategies:
    1. **LLM routing** — sends tool names + descriptions to the LLM and asks it
       to pick the relevant ones.  Good when the question is ambiguous or when
       the tool catalog is large.
    2. **Static routing** — accepts a pre-built list of `endpoint.toolname` strings
       (from a JSON spec, MongoDB config, etc.) and filters the catalog directly.
       No LLM call, deterministic and fast.

    Both strategies produce the same output: a filtered list of Bedrock toolSpec
    dicts ready to pass to `BedrockClient.configure_tools()`.

    Usage (client-side, inside CachedQueryProcessor):
        router = ToolRouter(all_tools, llm_client=self.llm_client)
        filtered = await router.route_for_question(question, routing_prompt)
        self.llm_client.configure_tools(filtered, callback)

    Usage (server-side, inside mongo_mcp.py):
        router = ToolRouter(all_tools, llm_client=llm_client)
        filtered = await router.route_for_question(question, prompt)
        return {"tools": filtered}

    Usage (static/deterministic):
        router = ToolRouter(all_tools)
        filtered = router.select_tools(["endpoint1.vector_search", "endpoint2.text_search"])
    """

    def __init__(
        self,
        tool_catalog: List[Dict[str, Any]],
        llm_client: Optional[Any] = None,
        message_handler: Optional[Callable] = None,
        settings: Optional[Any] = None,
        memory_fns: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            tool_catalog: Full list of Bedrock toolSpec dicts (with endpoint prefix already applied).
            llm_client: A BedrockClient (or subclass) instance for LLM routing. Not needed for static routing.
            message_handler: Optional progress callback matching (message, status) signature.
            settings: Application settings object.
            memory_fns: Dict of {tool_name: async_fn} from build_memory_dispatch().
                        When provided, pattern routing uses memory_strategy_recall/store
                        instead of the legacy PatternCache.
        """
        self.tool_catalog = tool_catalog
        self.llm_client = llm_client
        self.message_handler = message_handler or (lambda msg, status="Processing": None)
        self._memory_fns = memory_fns or {}

        # Build lookup indexes once
        self._by_name: Dict[str, Dict[str, Any]] = {}
        for tool in tool_catalog:
            spec = tool.get("toolSpec", {})
            name = spec.get("name", "")
            if name:
                self._by_name[name] = tool

        self._last_pattern: Optional[str] = None
        self._last_hints: Optional[Dict[str, Any]] = None
        self._last_selected_tools: Optional[List[str]] = None

    # ------------------------------------------------------------------
    #  Memory tool tracking
    # ------------------------------------------------------------------

    _MEMORY_TOOL_NAMES = {
        "intake", "recall", "reflect", "query", "list_sessions",
        "schema_declare", "strategy_store", "strategy_recall", "get_instructions",
    }

    @staticmethod
    def _is_memory_tool(tool_name: str) -> bool:
        """Check if a tool belongs to the memory layer."""
        return tool_name in ToolRouter._MEMORY_TOOL_NAMES

    def _separate_memory_tools(self, tools: List[Dict[str, Any]]) -> tuple:
        """Separate tools into memory and non-memory categories.
        
        Returns:
            Tuple of (memory_tools, non_memory_tools)
        """
        memory_tools = []
        non_memory_tools = []
        for tool in tools:
            name = tool.get("toolSpec", {}).get("name", "")
            if self._is_memory_tool(name):
                memory_tools.append(tool)
            else:
                non_memory_tools.append(tool)
        return memory_tools, non_memory_tools

    def _separate_memory_tool_names(self, tool_names: List[str]) -> tuple:
        """Separate tool names into memory and non-memory categories.
        
        Returns:
            Tuple of (memory_tool_names, non_memory_tool_names)
        """
        memory_names = []
        non_memory_names = []
        for name in tool_names:
            if self._is_memory_tool(name):
                memory_names.append(name)
            else:
                non_memory_names.append(name)
        return memory_names, non_memory_names

    # ------------------------------------------------------------------
    #  Static routing — deterministic, no LLM
    # ------------------------------------------------------------------

    def select_tools(self, tool_refs: List[str]) -> List[Dict[str, Any]]:
        """Filter the catalog to tools matching a list of references.

        Each ref can be:
        - An exact prefixed tool name:  "endpoint1_vector_search"
        - A dot-separated shorthand:    "endpoint1.vector_search"
          (converted to "endpoint1_vector_search" internally)
        - A bare tool name (matches any endpoint): "vector_search"

        Returns the matching subset of the full toolSpec catalog, preserving order.
        """
        normalized = set()
        bare_names = set()
        for ref in tool_refs:
            # Normalize dot notation to underscore prefix
            if "." in ref:
                parts = ref.split(".", 1)
                normalized.add(f"{parts[0]}_{parts[1]}")
            elif "_" in ref and ref in self._by_name:
                normalized.add(ref)
            else:
                # Bare tool name — match against any endpoint
                bare_names.add(ref)

        selected = []
        for tool in self.tool_catalog:
            name = tool.get("toolSpec", {}).get("name", "")
            if name in normalized:
                selected.append(tool)
            elif bare_names:
                # Check if the unprefixed portion matches any bare name
                suffix = name.rsplit("_", 1)[-1] if "_" in name else name
                if suffix in bare_names:
                    selected.append(tool)

        return selected

    # ------------------------------------------------------------------
    #  LLM routing — ask the model which tools are relevant
    # ------------------------------------------------------------------

    async def try_pattern_match(
        self,
        question: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Check memory strategies for a routing pattern match without invoking the LLM.

        Calls memory_strategy_recall with the question as query. The top result's
        payload.tools is used to filter the catalog. Memory tools are always added back.

        Returns:
            Tuple of (filtered_tools, hint_text) on a hit, or ([], None) on a miss.
        """
        self._last_pattern = None
        self._last_hints = None
        self._last_selected_tools = None

        recall_fn = self._memory_fns.get("strategy_recall")
        if recall_fn is None:
            return [], None

        try:
            result = await recall_fn(query=question, limit=5, similarity_threshold=0.0)
            results = result.get("strategies", result.get("results", []))
            if not results:
                logger.info("[STRATEGY RECALL] MISS — no matching pattern found")
                return [], None

            top = results[0]
            payload = top.get("payload") or {}
            cached_tools = payload.get("tools", [])
            if not cached_tools:
                logger.info("[STRATEGY RECALL] MISS — top result has no tools in payload")
                return [], None

            filtered = self.select_tools(cached_tools)
            # Always add memory tools back
            memory_tools, _ = self._separate_memory_tools(self.tool_catalog)
            filtered.extend(memory_tools)

            if not filtered:
                return [], None

            strategy_name = top.get("strategy_id") or top.get("strategy_key", question)
            self._last_pattern = strategy_name
            self._last_hints = top
            self._last_selected_tools = cached_tools

            # Build hint text from payload — prefer playbook, merge parent_playbook if extends
            hint_text = self._format_strategy_hints(top)

            hit_count = top.get("hit_count", 0)
            self.message_handler(
                f"Strategy recall hit — reusing {len(filtered)} tools | pattern: {strategy_name}"
                + (f" (hits: {hit_count})" if hit_count else "")
                + (" (with playbook)" if hint_text else " (no playbook yet)"),
                status="Tool Routing",
            )
            logger.info(
                "[STRATEGY RECALL] HIT pattern='%s' tools=%s boosted_score=%.3f hits=%d",
                strategy_name,
                cached_tools,
                top.get("boosted_score", 0.0),
                hit_count,
            )
            return filtered, hint_text

        except Exception as exc:
            logger.warning("Strategy recall failed: %s", exc)
            return [], None

    async def route_via_llm(
        self,
        question: str,
        routing_prompt: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Route a question via LLM tool selection (no pattern-cache lookup).

        Resets state variables at entry. After a successful LLM routing, the
        selected pattern is auto-saved to the cache for future cache hits.
        Memory tools are always added to the result and never saved to cache.

        Returns:
            Tuple of (filtered toolSpec list, hint_text — always None for LLM routing).
        """
        if self.llm_client is None:
            raise RuntimeError("LLM routing requires an llm_client.")

        self._last_pattern = None
        self._last_hints = None
        self._last_selected_tools = None

        # Get all non-memory tools for the LLM to select from
        non_memory_tools = [tool for tool in self.tool_catalog 
                           if not self._is_memory_tool(tool.get("toolSpec", {}).get("name", ""))]
        memory_tools = [tool for tool in self.tool_catalog 
                       if self._is_memory_tool(tool.get("toolSpec", {}).get("name", ""))]
        
        tool_summary = [
            {"name": name, "description": spec.get("toolSpec", {}).get("description", "")}
            for name, spec in self._by_name.items()
            if not self._is_memory_tool(name)  # Exclude memory tools from LLM selection
        ]

        if not routing_prompt:
            routing_prompt = self._default_routing_prompt()

        self.message_handler("Routing question to select relevant tools...", status="Tool Routing")

        user_text = (
            f"{routing_prompt}\n\n"
            f"Available tools:\n{json.dumps(tool_summary, indent=2)}\n\n"
            f"Question: {question}"
        )
        try:
            response_text = await self.llm_client.invoke_bedrock_text(user_text)
        except Exception as e:
            logger.warning(f"ToolRouter LLM call failed: {e}")
            # Return full catalog (including memory) on error
            return list(self.tool_catalog), None

        logger.debug(f"LLM routing response: {response_text}")
        routing = self._parse_routing_response(response_text)
        selected_names = routing["tools"]
        pattern = routing.get("pattern")
        filtered = [self._by_name[n] for n in selected_names if n in self._by_name]
        
        # Always add memory tools back (they're always available, never routed)
        filtered.extend(memory_tools)
        
        self._last_pattern = pattern
        # Store only non-memory selected names for pattern cache
        self._last_selected_tools = [n for n in selected_names if not self._is_memory_tool(n)]

        if filtered:
            self.message_handler(
                f"Router selected {len(filtered)} of {len(self.tool_catalog)} tools"
                + (f" | pattern: {pattern}" if pattern else ""),
                status="Tool Routing",
            )
        else:
            self.message_handler(
                "Router: no tools needed — LLM will answer directly",
                status="Tool Routing",
            )

        # Auto-save pattern → tools mapping via memory_strategy_store.
        # Only save non-memory tools; duplicate detection is handled by the
        # memory layer's similarity threshold on recall.
        non_memory_selected = [n for n in selected_names if not self._is_memory_tool(n)]
        store_fn = self._memory_fns.get("strategy_store")

        if pattern and store_fn and non_memory_selected:
            try:
                composite = self._build_strategy_content(pattern, non_memory_selected)
                await store_fn(
                    name=pattern,
                    description=composite,
                    payload={"tools": non_memory_selected},
                    schema_version="routing_pattern",
                )
            except Exception as exc:
                logger.warning("Failed to store routing strategy: %s", exc)

        return filtered, None

    async def route_for_question(
        self,
        question: str,
        routing_prompt: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Pattern-cache lookup followed by LLM routing on a miss.

        Checks the pattern cache first (no LLM). Falls back to LLM routing
        and auto-saves the result for future cache hits.

        Returns:
            Tuple of (filtered toolSpec list, hint_text or None).
        """
        if self.llm_client is None:
            raise RuntimeError("LLM routing requires an llm_client. Use select_tools() for static routing.")

        filtered, hint_text = await self.try_pattern_match(question)
        if filtered:
            return filtered, hint_text
        return await self.route_via_llm(question, routing_prompt)

    # ------------------------------------------------------------------
    #  Multi-candidate selection (kept for external callers)
    # ------------------------------------------------------------------

    async def _select_best_candidate(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Pick the best candidate from a pre-ranked list (top result wins)."""
        if not candidates:
            return None
        return candidates[0]

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_strategy_content(pattern: str, tool_names: Optional[List[str]] = None) -> str:
        """Build composite embedding text for a routing pattern (mirrors PatternCache._build_embedding_text)."""
        parts = [f"Pattern: {pattern}"]
        if tool_names:
            parts.append(f"Tools: {', '.join(tool_names)}")
        return "\n".join(parts)

    @staticmethod
    def _format_strategy_hints(result: Dict[str, Any]) -> Optional[str]:
        """Build hint text from a memory_strategy_recall result.

        Prefers payload.playbook, then merges parent_playbook (from extends),
        then falls back to legacy query_hints / output_hint format.
        """
        payload = result.get("payload") or {}
        playbook = payload.get("playbook")
        parent_playbook = result.get("parent_playbook")

        if playbook and parent_playbook:
            return f"{parent_playbook}\n\n---\n\n{playbook}"
        if playbook:
            return playbook
        if parent_playbook:
            return parent_playbook

        # Legacy fallback: query_hints + output_hint
        query_hints = payload.get("query_hints")
        output_hint = payload.get("output_hint")
        if not query_hints and not output_hint:
            return None

        parts = ["[OUTPUT FORMAT — Use the following format for output data]"]
        if query_hints:
            parts.append("\nTool calls that worked for a similar question:")
            for qh in query_hints:
                parts.append(f"  Tool: {qh['tool_name']}")
                parts.append(f"  Input: {json.dumps(qh['tool_input'], default=str)}")
        if output_hint:
            parts.append(
                "\nWrap your output data in [JSON_DATA_START] and [JSON_DATA_END] tags "
                f"using the following format:\n[JSON_DATA_START]\n{output_hint}\n[JSON_DATA_END]"
            )
        return "\n".join(parts)

    @staticmethod
    def _scope_from_tools(tool_names: List[str]) -> Optional[str]:
        """Derive an endpoint scope string from prefixed tool names.

        Tool names are formatted as ``{endpoint}_{tool}`` (e.g.
        ``shipwreckSearch_geospatial_search``).  This extracts the unique
        endpoint prefix(es) and joins them with ``+``.

        Returns None if no endpoints can be extracted.
        """
        endpoints = set()
        for name in tool_names:
            parts = name.split("_")
            if len(parts) >= 2:
                for i in range(1, len(parts)):
                    candidate = "_".join(parts[:i])
                    if any(c.isupper() for c in candidate):
                        endpoints.add(candidate)
                        break
                else:
                    endpoints.add(parts[0])
            else:
                endpoints.add(name)
        return "+".join(sorted(endpoints)) if endpoints else None

    def get_tool_summary(self) -> List[Dict[str, str]]:
        """Return lightweight name+description list for external use."""
        return [
            {"name": name, "description": spec.get("toolSpec", {}).get("description", "")}
            for name, spec in self._by_name.items()
        ]

    async def record_pattern(
        self, history: list, response_text: str, jsondata: Any = None, question: Optional[str] = None,
    ) -> None:
        """Generate a PII-free playbook from a completed interaction and save it.

        Sends the interaction to the LLM to produce a generalized, reusable
        playbook with typed placeholders (e.g. [person name], [location])
        instead of real user data. Saves via memory_strategy_store.
        """
        if self._last_pattern is None or self.llm_client is None:
            return

        store_fn = self._memory_fns.get("strategy_store")
        if store_fn is None:
            return

        try:
            # Extract tool calls and output structure from the interaction history
            tool_calls = self._extract_tool_calls(history, allowed_tools=self._last_selected_tools)
            output_hint = self._extract_output_hint(response_text)
            if output_hint is None and jsondata is not None:
                output_hint = self._skeleton_from_jsondata(jsondata)

            if not tool_calls and not output_hint:
                return

            has_json_output = output_hint is not None

            interaction_summary = f"Question: {question or '(unknown)'}\n\n"
            interaction_summary += "Tool calls made:\n"
            for tc in tool_calls:
                interaction_summary += f"  Tool: {tc['tool_name']}\n"
                interaction_summary += f"  Input: {json.dumps(tc['tool_input'], default=str)}\n\n"
            if has_json_output:
                interaction_summary += f"Output JSON structure:\n{output_hint}\n"

            if has_json_output:
                output_format_instructions = (
                    "### Output Format\n"
                    "This pattern produces structured JSON data. "
                    "Wrap the data in [JSON_DATA_START] and [JSON_DATA_END] tags:\n"
                    "[JSON_DATA_START]\n"
                    "{...skeleton with placeholders...}\n"
                    "[JSON_DATA_END]\n"
                )
            else:
                output_format_instructions = (
                    "### Output Format\n"
                    "Use standard Markdown formatting. Do NOT wrap the response in JSON_DATA tags.\n"
                )

            prompt = (
                "You are a pattern extraction agent. Given a successful LLM interaction below, "
                "produce a reusable PLAYBOOK that another LLM can follow for similar future questions.\n\n"
                "CRITICAL RULES:\n"
                "1. Replace ALL personally identifying information with typed placeholders in square brackets: "
                "[person name], [location], [coordinates], [date], [address], [phone], [email], [company name], etc.\n"
                "2. Replace specific values (city names, coordinates, counts, collection names) with descriptive "
                "placeholders like [geographic location], [latitude], [longitude], [search radius], [collection name].\n"
                "3. Keep tool names exactly as-is — never rename or generalize tool names.\n"
                "4. Keep the JSON output structure exactly — only replace specific values with placeholders.\n"
                "5. Keep domain-specific nouns (shipwrecks, weather, listings, etc.) — these are NOT PII.\n"
                "6. Only include a JSON output format section if the interaction actually produced structured JSON data. "
                "Most interactions should use normal Markdown output.\n\n"
                "Return the playbook in this EXACT format (no extra text):\n\n"
                "## Pattern: [1-2 sentence description of what this query does]\n\n"
                "### Example Queries (PII-free)\n"
                "- [generalized version of the original question with placeholders]\n"
                "- [another phrasing a user might use]\n\n"
                "### Steps\n"
                "1. [what to do first]\n"
                "2. [what to do next]\n\n"
                "### Tool Calls (adapt values to the current question)\n"
                "- tool_name: {\"param\": \"[placeholder]\", ...}\n\n"
                f"{output_format_instructions}\n"
                "---\n"
                f"INTERACTION TO GENERALIZE:\n{interaction_summary}"
            )

            self.message_handler("Generating PII-free playbook from interaction...", status="Strategy Store")
            playbook = await self.llm_client.invoke_bedrock_text(prompt)

            if not playbook or len(playbook) < 50:
                logger.warning("LLM returned empty/short playbook, skipping save")
                return

            # Extract PII-free example queries from the playbook
            example_queries = []
            in_examples = False
            for line in playbook.splitlines():
                if "example queries" in line.lower() and line.strip().startswith("#"):
                    in_examples = True
                    continue
                if in_examples:
                    if line.strip().startswith("#"):
                        break
                    stripped = line.strip().lstrip("- ").strip()
                    if stripped:
                        example_queries.append(stripped)

            tool_names = self._last_selected_tools or []
            composite = self._build_strategy_content(self._last_pattern, tool_names)
            payload: Dict[str, Any] = {"tools": tool_names, "playbook": playbook}
            if output_hint:
                payload["output_hint"] = output_hint
            if example_queries:
                payload["example_queries"] = example_queries
            if question:
                payload["example_queries"] = list(
                    dict.fromkeys([question] + payload.get("example_queries", []))
                )[:10]

            await store_fn(
                name=self._last_pattern,
                description=composite,
                payload=payload,
                schema_version="routing_pattern",
            )
            self.message_handler(
                f"Saved playbook for pattern: {self._last_pattern}",
                status="Strategy Store",
            )
        except Exception as exc:
            logger.warning("Failed to record pattern playbook: %s", exc)

    # ------------------------------------------------------------------
    #  Interaction extraction helpers (moved from PatternCache)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_calls(
        history: list, allowed_tools: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Pull tool name + input from Bedrock toolUse blocks in history."""
        calls = []
        allowed = set(allowed_tools) if allowed_tools else None
        for msg in history:
            if msg.get("role") != "assistant":
                continue
            for block in msg.get("content", []):
                if "toolUse" in block:
                    tu = block["toolUse"]
                    name = tu.get("name", "")
                    if allowed is None or name in allowed:
                        calls.append({"tool_name": name, "tool_input": tu.get("input", {})})
        return calls

    @staticmethod
    def _extract_output_hint(response_text: str) -> Optional[str]:
        """Extract JSON skeleton between [JSON_DATA_START] and [JSON_DATA_END] tags."""
        start = response_text.find("[JSON_DATA_START]")
        end = response_text.find("[JSON_DATA_END]")
        if start != -1 and end != -1 and end > start:
            raw = response_text[start + len("[JSON_DATA_START]"):end].strip()
            try:
                obj = json.loads(raw)
                return json.dumps(obj, indent=2, default=str)
            except json.JSONDecodeError:
                return raw if raw else None
        return None

    @staticmethod
    def _skeleton_from_jsondata(jsondata: Any) -> Optional[str]:
        """Build a JSON skeleton with placeholder values from a parsed object."""
        if jsondata is None:
            return None
        try:
            def _placeholder(v: Any) -> Any:
                if isinstance(v, str):
                    return "[string]"
                if isinstance(v, bool):
                    return False
                if isinstance(v, (int, float)):
                    return 0
                if isinstance(v, list):
                    return [_placeholder(v[0])] if v else []
                if isinstance(v, dict):
                    return {k: _placeholder(vv) for k, vv in v.items()}
                return None
            return json.dumps(_placeholder(jsondata), indent=2, default=str)
        except Exception:
            return None

    @staticmethod
    def _default_routing_prompt() -> str:
        return (
            "You are a tool routing agent. Given a user question and a list of available tools, "
            "return a JSON object with exactly two keys:\n"
            "  \"tools\": a JSON array of tool name strings — only the tools needed to answer the question. "
            "Prefer fewer tools. "
            "An empty array [] is a valid and preferred response when no tools are needed — "
            "for conversational questions, clarifications, or questions the LLM can answer from context alone, "
            "return an empty tools list and let the LLM respond directly.\n"
            "  \"pattern\": a short natural-language description of the query intent (1-2 sentences). "
            "Keep domain-specific nouns and verbs (e.g. 'weather', 'shipwrecks', 'geospatial', 'listings'). "
            "Only replace specific proper-noun values like city names, coordinates, or counts with "
            "short descriptive placeholders in square brackets. "
            "The pattern is used for semantic similarity matching, so it must read like a real query summary.\n"
            "Good: 'Search for weather station data near a geographic area using geospatial queries "
            "and return results with map coordinates'\n"
            "Good: 'Find shipwreck records near a coastal region using geospatial search'\n"
            "Bad:  'Find [entity] near [location]'  (too abstract, loses domain meaning)\n"
            "Bad:  'Search [collection] with [query_type]'  (generic placeholders kill semantic matching)\n"
            "Return ONLY valid JSON. No explanation, no markdown."
        )

    @staticmethod
    def _parse_routing_response(text: str) -> Dict[str, Any]:
        """Extract {tools: [...], pattern: str} from the LLM routing response.

        Falls back gracefully: if no JSON object is found, attempts to extract
        a bare tool-name array for backward compatibility.
        """
        # Try to find a JSON object first
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                tools = obj.get("tools", [])
                if isinstance(tools, list):
                    return {
                        "tools": [str(t) for t in tools if isinstance(t, str)],
                        "pattern": obj.get("pattern"),
                    }
            except json.JSONDecodeError:
                pass

        # Fallback: bare JSON array of tool names
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            try:
                names = json.loads(text[arr_start:arr_end + 1])
                if isinstance(names, list):
                    return {"tools": [str(n) for n in names if isinstance(n, str)], "pattern": None}
            except json.JSONDecodeError:
                pass

        # Last resort: comma-split
        names = [n.strip().strip('"').strip("'") for n in text.split(",") if n.strip()]
        return {"tools": names, "pattern": None}
