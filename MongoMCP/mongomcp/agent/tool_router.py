import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .pattern_cache import PatternCache

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
    ):
        """
        Args:
            tool_catalog: Full list of Bedrock toolSpec dicts (with endpoint prefix already applied).
            llm_client: A BedrockClient (or subclass) instance for LLM routing. Not needed for static routing.
            message_handler: Optional progress callback matching (message, status) signature.
            settings: Application settings object (reserved for future use).
        """
        self.tool_catalog = tool_catalog
        self.llm_client = llm_client
        self.message_handler = message_handler or (lambda msg, status="Processing": None)

        # Build lookup indexes once
        self._by_name: Dict[str, Dict[str, Any]] = {}
        for tool in tool_catalog:
            spec = tool.get("toolSpec", {})
            name = spec.get("name", "")
            if name:
                self._by_name[name] = tool

        self._pattern_cache: Optional[PatternCache] = (
            PatternCache(settings) if settings else None
        )
        self._last_pattern: Optional[str] = None
        self._last_hints: Optional[Dict[str, Any]] = None

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

    async def route_for_question(
        self,
        question: str,
        routing_prompt: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Use the LLM to select tools relevant to a question.

        Checks the PatternCache first. On a hit, returns cached tools and
        sets self._last_hints so the caller can inject contextual hints.

        Returns:
            Tuple of (filtered toolSpec list, hint_text or None).
        """
        if self.llm_client is None:
            raise RuntimeError("LLM routing requires an llm_client. Use select_tools() for static routing.")

        self._last_pattern = None
        self._last_hints = None
        self._last_selected_tools = None

        # --- Pattern cache lookup (unscoped — find best match across all endpoints) ---
        if self._pattern_cache is not None:
            try:
                self._pattern_cache._tool_scope = None  # search all endpoints
                self._pattern_cache.reset_connection()
                candidates = await self._pattern_cache.find_best_match(question)
                if candidates:
                    chosen = await self._select_best_candidate(question, candidates)
                    if chosen is not None:
                        cached_tools = chosen.get("tools", [])
                        filtered = self.select_tools(cached_tools)
                        if filtered:
                            self._last_pattern = chosen.get("pattern", question)
                            self._last_hints = chosen
                            self._last_selected_tools = cached_tools
                            hint_text = PatternCache.format_hints(chosen)

                            # Bump hit_count for the selected pattern
                            if self._pattern_cache is not None:
                                try:
                                    col = await self._pattern_cache._collection()
                                    phash = self._pattern_cache._make_key(chosen["pattern"])
                                    await col.update_one(
                                        {"pattern_hash": phash},
                                        {"$inc": {"hit_count": 1}, "$set": {"last_used": time.time()}},
                                    )
                                except Exception:
                                    pass

                            self.message_handler(
                                f"Pattern cache hit — reusing {len(filtered)} tools | pattern: {self._last_pattern}"
                                + (f" (hits: {chosen.get('hit_count', 0)})" if chosen.get("hit_count") else "")
                                + (" (with playbook)" if hint_text else " (no playbook yet — click \U0001f44d to save)"),
                                status="Tool Routing",
                            )
                            return filtered, hint_text
            except Exception as e:
                logger.warning(f"Pattern cache lookup failed, falling back to LLM: {e}")

        # --- LLM routing ---
        tool_summary = [
            {"name": name, "description": spec.get("toolSpec", {}).get("description", "")}
            for name, spec in self._by_name.items()
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
            return list(self.tool_catalog), None

        logger.debug(f"LLM routing response: {response_text}")
        routing = self._parse_routing_response(response_text)
        selected_names = routing["tools"]
        pattern = routing.get("pattern")
        filtered = [self._by_name[n] for n in selected_names if n in self._by_name]
        self._last_pattern = pattern
        self._last_selected_tools = selected_names

        self.message_handler(
            f"Router selected {len(filtered)} of {len(self.tool_catalog)} tools"
            + (f" | pattern: {pattern}" if pattern else ""),
            status="Tool Routing",
        )

        # Auto-save pattern → tools mapping so future similar queries skip LLM routing.
        # Check for near-duplicate patterns first to avoid proliferation.
        # Scope to the endpoints of the selected tools so patterns don't cross-contaminate.
        if pattern and self._pattern_cache is not None and filtered:
            try:
                scope = self._scope_from_tools(selected_names)
                self._pattern_cache._tool_scope = scope
                self._pattern_cache.reset_connection()
                existing = await self._pattern_cache.find_similar_pattern(pattern)
                if existing:
                    logger.info(f"Merging into existing pattern: '{existing}' (new was: '{pattern}')")
                    self._last_pattern = existing
                    await self._pattern_cache.set(existing, selected_names)
                else:
                    await self._pattern_cache.set(pattern, selected_names)
            except Exception as e:
                logger.warning(f"Failed to cache routing pattern: {e}")

        return filtered, None

    # ------------------------------------------------------------------
    #  Multi-candidate selection
    # ------------------------------------------------------------------

    async def _select_best_candidate(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Pick the best cached pattern for *question* from a list of candidates.

        If there is only one candidate, return it directly (no LLM call).
        Otherwise, ask the LLM to choose, presenting each candidate's pattern,
        similarity score, and hit_count (a proxy for user-validated quality).
        """
        if len(candidates) == 1:
            return candidates[0]

        # Build a numbered list for the LLM
        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}. Pattern: {c['pattern']}\n"
                f"   Similarity: {c.get('score', 0):.3f}\n"
                f"   User-confirmed hits: {c.get('hit_count', 0)}\n"
                f"   Tools: {', '.join(c.get('tools', []))}"
            )

        prompt = (
            "You are selecting the best cached pattern to answer a user question.\n\n"
            f"User question: {question}\n\n"
            "Candidate patterns (ranked by vector similarity):\n"
            + "\n".join(lines)
            + "\n\n"
            "Consider both semantic similarity AND the hit count — a higher hit count "
            "means more users have confirmed that pattern works well.\n\n"
            "Reply with ONLY the number of the best candidate (e.g. '1'). "
            "If none of the candidates are a good fit, reply 'NONE'."
        )

        try:
            self.message_handler(
                f"Choosing among {len(candidates)} cached patterns...",
                status="Tool Routing",
            )
            answer = await self.llm_client.invoke_bedrock_text(prompt)
            answer = answer.strip()

            if answer.upper() == "NONE":
                logger.info("LLM rejected all cached pattern candidates")
                return None

            # Parse the number
            idx = int(answer.split()[0].strip(".)")) - 1
            if 0 <= idx < len(candidates):
                logger.info(
                    f"LLM selected candidate {idx + 1}: '{candidates[idx]['pattern']}'"
                )
                return candidates[idx]
        except (ValueError, IndexError):
            logger.warning(f"Could not parse LLM candidate selection: '{answer}'")
        except Exception as e:
            logger.warning(f"LLM candidate selection failed: {e}")

        # Fallback: pick the candidate with the highest hit_count, breaking ties by score
        return max(candidates, key=lambda c: (c.get("hit_count", 0), c.get("score", 0)))

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

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
        instead of real user data.
        """
        if self._pattern_cache is None or self._last_pattern is None:
            return
        if self.llm_client is None:
            return

        try:
            # Build a compact summary of what happened for the LLM
            tool_calls = PatternCache.extract_tool_calls(history, allowed_tools=self._last_selected_tools)
            output_hint = PatternCache.extract_output_hint(response_text)
            if output_hint is None and jsondata is not None:
                output_hint = PatternCache._skeleton_from_jsondata(jsondata)

            if not tool_calls and not output_hint:
                return

            # Determine whether this interaction produced structured JSON output.
            # Only interactions with JSON data should get an Output Format section.
            has_json_output = output_hint is not None

            interaction_summary = f"Question: {question or '(unknown)'}\n\n"
            interaction_summary += "Tool calls made:\n"
            for tc in tool_calls:
                interaction_summary += f"  Tool: {tc['tool_name']}\n"
                interaction_summary += f"  Input: {json.dumps(tc['tool_input'], default=str)}\n\n"
            if has_json_output:
                interaction_summary += f"Output JSON structure:\n{output_hint}\n"

            # Build the output format section conditionally
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

            self.message_handler("Generating PII-free playbook from interaction...", status="Pattern Cache")
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
                        break  # hit next section
                    stripped = line.strip().lstrip("- ").strip()
                    if stripped:
                        example_queries.append(stripped)

            # Set scope to the endpoints of the selected tools
            scope = self._scope_from_tools(self._last_selected_tools or [])
            self._pattern_cache._tool_scope = scope
            self._pattern_cache.reset_connection()
            await self._pattern_cache.set(
                pattern=self._last_pattern,
                tool_names=self._last_selected_tools or [],
                playbook=playbook,
                example_queries=example_queries or None,
            )
            self.message_handler(
                f"Saved playbook for pattern: {self._last_pattern}",
                status="Pattern Cache",
            )
        except Exception as e:
            logger.warning(f"Failed to record pattern playbook: {e}")

    @staticmethod
    def _default_routing_prompt() -> str:
        return (
            "You are a tool routing agent. Given a user question and a list of available tools, "
            "return a JSON object with exactly two keys:\n"
            "  \"tools\": a JSON array of tool name strings — only the tools needed to answer the question. Prefer fewer tools.\n"
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
