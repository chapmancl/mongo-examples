import json
import logging
from typing import Any, Callable, Dict, List, Optional

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
    ):
        """
        Args:
            tool_catalog: Full list of Bedrock toolSpec dicts (with endpoint prefix already applied).
            llm_client: A BedrockClient (or subclass) instance for LLM routing. Not needed for static routing.
            message_handler: Optional progress callback matching (message, status) signature.
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
    ) -> List[Dict[str, Any]]:
        """Use the LLM to select tools relevant to a question.

        The LLM receives only tool names and descriptions (no full schemas)
        and returns a JSON list of selected tool names.

        Args:
            question: The user's question.
            routing_prompt: System-level instruction for the routing LLM. If None,
                a sensible default is used.

        Returns:
            Filtered list of Bedrock toolSpec dicts.
        """
        if self.llm_client is None:
            raise RuntimeError("LLM routing requires an llm_client. Use select_tools() for static routing.")

        tool_summary = [
            {"name": name, "description": spec.get("toolSpec", {}).get("description", "")}
            for name, spec in self._by_name.items()
        ]

        if not routing_prompt:
            routing_prompt = self._default_routing_prompt()

        user_message = (
            f"{routing_prompt}\n\n"
            f"Available tools:\n{json.dumps(tool_summary, indent=2)}\n\n"
            f"Question: {question}\n\n"
            "Return ONLY a JSON array of tool name strings. No explanation."
        )

        self.message_handler("Routing question to select relevant tools...", status="Tool Routing")

        # Use the LLM without any tools configured — pure text completion
        original_tools = self.llm_client.mcp_tools
        original_setup = self.llm_client.llm_setup
        try:
            self.llm_client.mcp_tools = None
            self.llm_client.llm_setup = False

            response = await self.llm_client.invoke_bedrock_with_tools(
                request={"messages": [{"role": "user", "content": [{"text": user_message}]}]},
            )
        finally:
            self.llm_client.mcp_tools = original_tools
            self.llm_client.llm_setup = original_setup

        selected_names = self._parse_tool_names(response)
        filtered = [self._by_name[n] for n in selected_names if n in self._by_name]

        self.message_handler(
            f"Router selected {len(filtered)} of {len(self.tool_catalog)} tools",
            status="Tool Routing"
        )
        return filtered

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def get_tool_summary(self) -> List[Dict[str, str]]:
        """Return lightweight name+description list for external use."""
        return [
            {"name": name, "description": spec.get("toolSpec", {}).get("description", "")}
            for name, spec in self._by_name.items()
        ]

    @staticmethod
    def _default_routing_prompt() -> str:
        return (
            "You are a tool routing agent. Given a user question and a list of available tools, "
            "select ONLY the tools that are needed to answer the question. "
            "Prefer fewer tools. Always include get_collection_info if the question "
            "requires understanding the data schema. "
            "Return your answer as a JSON array of tool name strings."
        )

    @staticmethod
    def _parse_tool_names(response: Dict[str, Any]) -> List[str]:
        """Extract a list of tool name strings from the LLM response."""
        raw = response.get("response", "")
        if isinstance(raw, list):
            return [str(item) for item in raw if isinstance(item, str)]

        text = str(raw)
        # Find the JSON array in the response text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                names = json.loads(text[start:end + 1])
                if isinstance(names, list):
                    return [str(n) for n in names if isinstance(n, str)]
            except json.JSONDecodeError:
                pass

        # Fallback: split comma-separated names
        return [n.strip().strip('"').strip("'") for n in text.split(",") if n.strip()]
