import asyncio
import json
import queue
import traceback
from typing import Any, List, Optional

import fastmcp
import mcp.types as mt
import requests
from pydantic import BaseModel

from aws_settings import settings
from mongomcp.agent.webui_bedrock_client import WebUiBedrockClient

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
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
    session_id: Optional[str] = None


class APIQueryProcessor:
    def __init__(self):
        self._init_error: Optional[Exception] = None
        self._message_queue: queue.Queue = queue.Queue()
        self._history_cleared: bool = False
        self._history: Optional[List[Any]] = None

        logger.info(f"Initializing processor with endpoint: {settings.mongo_mcp_root}")
        try:
            self._headers = {
                "Authorization": f"Bearer {settings.AUTH_TOKEN}",
                "Content-Type": "application/json",
            }
            self.llm_client = WebUiBedrockClient(settings)
            self.mcp_endpoints: List[str] = []
            self.mcp_endpoint_configs: dict = {}
            self.endpoint_clients: dict = {}
            self.mongo_collection_info: dict = {}
            self.mcp_tools_config: Optional[List[dict]] = None
            self._discover_tools()
        except Exception as e:
            self._init_error = e
            logger.error(f"Processor initialization failed: {e}")

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

    def _discover_tools(self):
        asyncio.run(self._discover_tools_async())

    async def _discover_tools_async(self):
        try:
            resp = requests.get(
                f"{settings.mongo_mcp_root}/tools_config",
                headers=self._headers, timeout=15,
            )
            resp.raise_for_status()
            self.mcp_endpoints = resp.json().get("available_tools", [])
            self._emit(f"Discovered endpoints: {self.mcp_endpoints}", status="Discovering Tools")
        except Exception as e:
            self._emit(f"Error fetching endpoint list: {e}", status="Error")
            self.mcp_endpoints = []

        root_fmt = f"{settings.mongo_mcp_root}/{{}}/mcp"
        results = await asyncio.gather(*[
            self._fetch_endpoint(name, root_fmt) for name in self.mcp_endpoints
        ])

        bedrock_tools, agent_prompts = [], {}
        for name, config, tools, collection_info, agent_prompt in results:
            self.mcp_endpoint_configs[name] = config
            bedrock_tools.extend(tools)
            if collection_info:
                self.mongo_collection_info[name] = collection_info
            if agent_prompt and agent_prompt.strip():
                agent_prompts[name] = agent_prompt.strip()

        system_prompt = [
            {"text": t}
            for t in getattr(settings, "BEDROCK_SYSTEM_PROMPT_TEXTS", [])
        ]
        for ep, pt in agent_prompts.items():
            system_prompt.append({"text": f"***IMPORTANT {ep}: {pt}"})
        non_empty = {k: v for k, v in self.mongo_collection_info.items() if v}
        if non_empty:
            system_prompt.append({"text": json.dumps(non_empty)})

        self.llm_client.system = system_prompt
        self.mcp_tools_config = bedrock_tools
        self.llm_client.configure_tools(bedrock_tools, self._call_mcp_tool)
        self._emit(
            f"Ready: {len(bedrock_tools)} tools from {len(self.mcp_endpoints)} endpoint(s)",
            status="Tools Ready",
        )

    async def _fetch_endpoint(self, name, root_fmt):
        config = {
            "url": root_fmt.format(name),
            "transport": "http",
            "headers": {"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
        }
        tools, agent_prompt, collection_info = [], "", {}
        try:
            resp = await asyncio.to_thread(
                requests.get, f"{settings.mongo_mcp_root}/{name}/llm_tools",
                headers=self._headers, timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
            agent_prompt = payload.get("agent_prompt", "") if isinstance(payload, dict) else ""
            tools = payload.get("tools", []) if isinstance(payload, dict) else []
            for tool in tools:
                if "toolSpec" in tool and "name" in tool["toolSpec"]:
                    tool["toolSpec"]["name"] = f"{name}_{tool['toolSpec']['name']}"
            self._emit(f"Fetched {len(tools)} tools from {name}", status="Tools Discovered")
        except Exception as e:
            self._emit(f"Error fetching tools for {name}: {e}", status="Error")
        try:
            resp = await asyncio.to_thread(
                requests.get, f"{settings.mongo_mcp_root}/{name}/collection_info",
                headers=self._headers, timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
            collection_info = payload.get("collection_info", {}) if isinstance(payload, dict) else payload
        except Exception as e:
            self._emit(f"Error fetching collection info for {name}: {e}", status="Error")
        return name, config, tools, collection_info, agent_prompt

    # ------------------------------------------------------------------
    # MCP tool dispatch
    # ------------------------------------------------------------------

    async def _call_mcp_tool(self, toolname, tool_input):
        endpoint_name, endpoint_tool_name = self._resolve_endpoint(toolname)
        client = self.endpoint_clients.get(endpoint_name)
        if client is None:
            cfg = self.mcp_endpoint_configs.get(endpoint_name)
            if cfg is None:
                raise RuntimeError(f"No config for endpoint '{endpoint_name}'")
            client = fastmcp.Client({"mcpServers": {endpoint_name: cfg}})
            self.endpoint_clients[endpoint_name] = client

        async def _run():
            async with client:
                return await client.session.send_request(
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

        result = await asyncio.wait_for(_run(), timeout=90)
        if result.content and hasattr(result.content[0], "text"):
            return result.content[0].text
        if result.structuredContent is not None:
            return json.dumps(result.structuredContent)
        return str(result)

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

    def query_with_mcp_tools(self, request: "QueryRequest") -> "QueryResponse":
        incoming = None if self._history_cleared else request.history
        self._history_cleared = False
        # Always overwrite from the incoming value — even an empty list means "fresh session".
        # Only fall back to the in-RAM history when the caller sent nothing (None).
        if incoming is not None:
            self._history = incoming
        if self._history is None:
            self._history = []

        self._history = self._trim_history(self._history)

        msgs = list(self._history)
        user_content = [{"text": request.input}]
        self._inject_token_warning(msgs, user_content)
        msgs.append({"role": "user", "content": user_content})
        self.llm_client.message_handler = self._emit

        async def _invoke():
            self.llm_client.configure_tools(self.mcp_tools_config or [], self._call_mcp_tool)
            return await self.llm_client.invoke_bedrock_with_tools_text(messages=msgs)

        result = asyncio.run(_invoke())
        self._history = result.get("history", msgs)
        answer = result.get("response_text", "")
        jsondata = result.get("jsondata")

        # Emit token usage summary after every response
        usage = result.get("usage") or {}
        in_tok = int(usage.get("inputTokens", 0) or 0)
        out_tok = int(usage.get("outputTokens", 0) or 0)
        total = in_tok + out_tok
        pct = round(total / self.MAX_CONTEXT_TOKENS * 100, 1)
        self._emit(
            f"Tokens — in: {in_tok:,}  out: {out_tok:,}  total: {total:,}  ({pct}% of {self.MAX_CONTEXT_TOKENS // 1000}k)",
            status="Token Usage",
        )

        content = {"text": answer, "jsondata": jsondata}
        return QueryResponse(
            status="Query Completed", message="Completed",
            content=content, history=self._history,
        )

    # ------------------------------------------------------------------
    # History trimming
    # ------------------------------------------------------------------

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
        for i, msg in enumerate(result):
            if msg.get("role") == "user":
                return result[i:]
        return []

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

    def clear_history(self) -> "QueryResponse":
        self._history = None
        self._history_cleared = True
        try:
            while True:
                self._message_queue.get_nowait()
        except queue.Empty:
            pass
        return QueryResponse(status="Clear History", message="History cleared", history=[])

    def get_history(self) -> "QueryResponse":
        return QueryResponse(
            status="Get History", message="Completed",
            history=self._trim_history_for_ui(self._history),
        )

    def get_mcp_config(self) -> dict:
        return {
            "endpoints": self.mcp_endpoints or [],
            "collection_info": self.mongo_collection_info or {},
        }

    def save_pattern(self, user_id: str, session_id: str) -> "QueryResponse":
        message = (
            f"The user marked this conversation as a useful pattern worth saving. "
            f"Please store the key question, approach, and answer from this session "
            f"(session_id: `{session_id}`, user_id: `{user_id}`) to memory as a "
            f"'pattern:query' memory_type entry in the semantic collection with "
            f"high importance (0.9). Include the user's original question and the "
            f"approach used to answer it so it can be reused in future sessions."
        )
        req = QueryRequest(input=message, user_id=user_id, session_id=session_id)
        return self.query_with_mcp_tools(req)

    def record_feedback(self, user_id: str, session_id: str, feedback: str) -> "QueryResponse":
        sentiment = "positive" if feedback == "positive" else "negative"
        emoji = "\U0001f44d" if sentiment == "positive" else "\U0001f44e"
        message = (
            f"{emoji} The user gave **{sentiment} feedback** on session `{session_id}` "
            f"(user: {user_id}). "
            f"Please store this interaction pattern to memory using the intake tool - "
            f"include the session_id, user_id, feedback sentiment, and any relevant context "
            f"from the conversation history as a 'feedback:{sentiment}' memory_type entry."
        )
        req = QueryRequest(input=message, user_id=user_id, session_id=session_id)
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
