import json
import logging
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from .webui_bedrock_client import WebUiBedrockClient
from .tool_router import ToolRouter

logger = logging.getLogger(__name__)


class PromptAgent:
    """Focused sub-agent that runs a Bedrock invoke loop with a specific
    prompt and an optionally filtered tool set.

    Tool calls inside the loop route through the caller-provided mcp_call_fn
    (APIQueryProcessor._call_mcp_tool), so they reach real MCP servers through
    the load balancer — no internal dispatch table is needed here.

    Usage (from mcp_processor.py)::

        agent = PromptAgent(
            settings=settings,
            mcp_call_fn=self._call_mcp_tool,
            tool_catalog=sub_catalog,   # agent_ tools excluded to prevent recursion
        )
        result_json = await agent.run(prompt="...", tool_names=["vector_search"])
    """

    _TOOLSPECS_PATH = Path(__file__).parent / "agent_tools.json"

    @classmethod
    def get_toolspecs(cls) -> List[Dict[str, Any]]:
        """Load agent toolSpec dicts from the bundled JSON file."""
        with open(cls._TOOLSPECS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("agent_tools", [])

    def __init__(
        self,
        settings: Any,
        mcp_call_fn: Callable,
        tool_catalog: List[Dict[str, Any]],
        save_fn: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        settings     : Application settings passed to WebUiBedrockClient.
        mcp_call_fn  : Async callable ``(toolname: str, tool_input: dict) -> str``.
                       Should be ``APIQueryProcessor._call_mcp_tool`` so tool calls
                       route through the load balancer to the correct container.
        tool_catalog : Full list of Bedrock toolSpec dicts available to sub-agents.
                       Caller should exclude ``agent_`` tools to prevent recursion.
        save_fn      : Optional callable ``(data: dict, agent_id: str, tool_name: str,
                       prompt_name: str) -> None``.  When provided, called after every
                       run() to persist the conversation snapshot — mirrors
                       mongo_mcp.invoke_llm._save_output_snapshot.
                       Pass ``mongo_middleware.save_llm_conversation`` from the webui.
        """
        self.settings = settings
        self._mcp_call = mcp_call_fn
        self._tool_catalog = tool_catalog
        self._save_fn = save_fn

    # ------------------------------------------------------------------
    # Tool filtering
    # ------------------------------------------------------------------

    def _build_tool_config(
        self, tool_names: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Return a filtered toolSpec list.

        Memory tools are always included so the sub-agent can use recall/intake
        regardless of what the caller requested.
        """

        catalog_names = [t.get("toolSpec", {}).get("name", "") for t in self._tool_catalog]
        logger.debug(
            "[PromptAgent] _build_tool_config: catalog=%d tools, requested tool_names=%s",
            len(self._tool_catalog),
            tool_names,
        )

        if not tool_names:
            logger.warning(
                "[PromptAgent] _build_tool_config: tool_names is empty — using full catalog (%d tools). "
                "Parent LLM should supply tool_names from strategy_recall.",
                len(self._tool_catalog),
            )
            return self._tool_catalog

        router = ToolRouter(tool_catalog=self._tool_catalog)
        filtered = router.select_tools(tool_names)

        # Always add memory tools back so the sub-agent can use recall/intake
        # regardless of what tool_names the caller supplied.
        mem_tools, _ = router._separate_memory_tools(self._tool_catalog)
        seen = {t["toolSpec"]["name"] for t in filtered}
        added_mem = []
        for mt in mem_tools:
            if mt["toolSpec"]["name"] not in seen:
                filtered.append(mt)
                added_mem.append(mt["toolSpec"]["name"])
        if added_mem:
            logger.debug("[PromptAgent] _build_tool_config: auto-added memory tools: %s", added_mem)

        matched_names = [t.get("toolSpec", {}).get("name", "") for t in filtered]
        unmatched = [n for n in tool_names if not any(
            n == name or name.endswith(f"_{n}") for name in catalog_names
        )]
        logger.debug(
            "[PromptAgent] _build_tool_config: filtered to %d tools: %s",
            len(filtered),
            matched_names,
        )
        if unmatched:
            logger.warning(
                "[PromptAgent] _build_tool_config: requested tool_names not found in catalog: %s",
                unmatched,
            )

        return filtered

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        context: Optional[Any] = None,
        memory_id: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        system_instructions: Optional[str] = None,
        emit_fn: Optional[Callable] = None,
        token: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute a Bedrock invoke loop and return the result as a dict.

        Shape: ``{"response_text": "...", "usage": {...}, "error"?: "..."}``

        Parameters
        ----------
        prompt     : Task instruction for the sub-agent.
        context    : Optional dict or string appended to the prompt as context.
        memory_id  : ObjectId hex of a memory document.  The sub-agent is
                     instructed to load it; if the document's payload.tools
                     list exists it is used as the default tool_names.
        tool_names : Optional tool name filter (bare, dot-notation, or prefixed).
                     Memory tools are always added.  Pass None for all tools.
        session_id : Session identifier forwarded to memory tools.
        emit_fn    : Optional callable matching the parent emit signature
                     ``(message, status=...)`` — when provided, sub-agent
                     progress messages are forwarded to the UI stream.
        """

        # --- 1. Build filtered tool config ---
        filtered_tools = self._build_tool_config(tool_names)
        logger.debug(
            "[PromptAgent] run: tool_names=%s memory_id=%s session_id=%s",
            tool_names,
            memory_id,
            session_id,
        )
        logger.debug(
            "[PromptAgent] run: invoking with %d tools: %s",
            len(filtered_tools),
            [t.get("toolSpec", {}).get("name", "") for t in filtered_tools],
        )

        # --- 2. Compose full prompt ---
        full_prompt = prompt
        if session_id:
            full_prompt = f"[Session ID: {session_id}]\n\n{full_prompt}"
        else:
            full_prompt = (
                "[Session Setup] No session_id was provided. Before doing any other work, "
                "create one now in the format '{username}:{session_id}:{YYYY-MM-DDTHH:MM}' using "
                "the current username, session identifier, and current date/time. "
                "This session_id makes it possible to recall this sub-agent's memories later "
                "by querying with this session_id. Use this value as session_id for all memory "
                "tool calls in this task.\n\n"
                + full_prompt
            )
        if memory_id:
            full_prompt = (
                f"{full_prompt}\n\n"
                f"[Context Note: Memory document ID {memory_id} contains "
                f"relevant context and the available tool list. Use memory tools to load it.]"
            )

        context_str: Optional[str] = None
        if context is not None:
            context_str = (
                json.dumps(context) if not isinstance(context, str) else context
            )

        # --- 3. Fresh client — isolated from the parent LLM's shared state ---
        client = WebUiBedrockClient(self.settings)
        if emit_fn is not None:
            client.message_handler = emit_fn
        sub_agent_system = [
            {
                "text": (
                    "You are a focused sub-agent. Complete the assigned task using "
                    "the available tools. Be concise and return a clear final answer "
                    "when done."
                )
            }
        ]
        if system_instructions:
            sub_agent_system.insert(0, {"text": system_instructions})
        client.system = sub_agent_system
        client.configure_tools(filtered_tools, self._mcp_call)

        # --- 3b. Save initial snapshot (before invoke — no usage/history/stats yet) ---
        _doc_id: Optional[str] = None
        if self._save_fn is not None:
            try:
                _agent_id = "agent"
                if isinstance(token, dict):
                    _agent_id = token.get("agent_key")
                elif token is not None:
                    _agent_id = token.client_id
                _initial = {"prompt": prompt, "input_context": context_str}
                _result_or_coro = self._save_fn(_initial, _agent_id, "agent_run_prompt", "user_prompt")
                if inspect.isawaitable(_result_or_coro):
                    _doc_id = await _result_or_coro
                else:
                    _doc_id = _result_or_coro
            except Exception as _save_err:
                logger.warning("[PromptAgent] Failed to save initial snapshot: %s", _save_err)

        # --- 4. Invoke ---
        output: Dict[str, Any] = {"prompt": prompt}
        try:
            result = await client.invoke_bedrock_with_tools_text(
                prompt=full_prompt,
                context=context_str,
            )
            output["response_text"] = result.get("response_text")
            output["usage"] = result.get("usage")
            output["history"] = result.get("history")
            output["stats"] = result.get("stats")
            if result.get("error"):
                output["error"] = result["error"]
        except Exception as invoke_err:
            logger.error("[PromptAgent] invoke failed: %s", invoke_err)
            output["error"] = str(invoke_err)

        if memory_id:
            output["memory_id"] = memory_id
        if session_id:
            output["session_id"] = session_id


        # --- 5. Persist conversation snapshot (best-effort, always runs) ---
        if self._save_fn is not None:
            try:
                agent_id = "agent"
                if isinstance(token, dict):
                    agent_id = token.get("agent_key")
                elif token is not None:
                    agent_id = token.client_id
                result_or_coro = self._save_fn(output, agent_id, "agent_run_prompt", "user_prompt", _doc_id)
                if inspect.isawaitable(result_or_coro):
                    await result_or_coro
            except Exception as save_err:
                logger.warning("[PromptAgent] Failed to save conversation snapshot: %s", save_err)

        return output
