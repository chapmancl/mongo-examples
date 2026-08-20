"""Deterministic conversation checkpointing (no LLM).

Extracts a compact, structured snapshot from Bedrock ``messages`` so that context
lost to history trimming or corruption-clearing can be reloaded on a later turn.

The extraction is purely deterministic: it scans message content blocks for user
text, tool names, a distilled answer lead, and fact patterns. It does NOT depend
on toolUse/toolResult pairing, so it works unchanged on a *corrupt* history that
Bedrock has rejected (dangling toolUse / orphaned toolResult).

The snapshot is stored as a single ``memory_type="conversation"`` episodic memory
per session (upserted in place), with the structured fields under ``payload`` and
a rendered text blob as ``content`` (used for embedding + display).
"""

from typing import Any, Dict, List
import re

MEMORY_TYPE = "conversation"

# Bounds so a single checkpoint doc stays small no matter how long the conversation
# runs. Oldest entries are dropped when a cap is exceeded.
_MAX_USER_TURNS = 40
_MAX_TOOL_CALLS = 60
_MAX_ANSWER_LEADS = 8
_MAX_PINNED_FACTS = 80
_MAX_OPEN_THREADS = 30

# Per-user-turn character cap. Kills pathological pastes (whole documents pasted
# into the chat) that would otherwise bloat the checkpoint and its embedding.
_MAX_USER_TURN_LEN = 100
# Distilled-answer bound.
_MAX_ANSWER_LEAD_LEN = 240

# Fact patterns worth preserving verbatim — the concrete tokens a model cannot
# reconstruct. Mined ONLY from user/assistant text (never tool-result bodies),
# and paths are restricted to real file/URL shapes so slash-delimited prose
# (e.g. "arrays/ObjectId/UUID") is no longer captured as a "fact".
# Order matters: extraction consumes matched spans left-to-right, so URLs are
# matched (and blanked) before path patterns can carve fragments out of them.
_FACT_PATTERNS = [
    re.compile(r"https?://[^\s)>\]]+"),                 # URLs (first)
    re.compile(r"\b[0-9a-fA-F]{24}\b"),                 # Mongo ObjectId
    re.compile(r"(?:/[A-Za-z0-9_.\-]+){2,}"),           # absolute path /a/b/c
    re.compile(r"\b[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+\.[A-Za-z]{1,5}\b"),  # rel path w/ .ext
]

_MD_STRIP = re.compile(r"[*_`>#]+")


def _distill_answer(text: str) -> str:
    """Deterministically distil an assistant answer to a short lead (no LLM).

    Takes the first non-empty paragraph, strips markdown markers, and keeps the
    first couple of sentences. Lossy but zero-cost — good enough to seed the
    embedded ``content`` with answer vocabulary for fast-path recall.
    """
    if not text:
        return ""
    para = ""
    for chunk in re.split(r"\n\s*\n", text.strip()):
        c = chunk.strip()
        if not c:
            continue
        # Skip pure markdown headings (e.g. '## The Score') so the lead is real prose.
        if all(ln.lstrip().startswith("#") for ln in c.splitlines()):
            continue
        para = c
        break
    if not para:
        return ""
    clean = re.sub(r"\s+", " ", _MD_STRIP.sub("", para)).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    return " ".join(sentences[:2]).strip()[:_MAX_ANSWER_LEAD_LEN]


def _iter_blocks(msg: Dict[str, Any]):
    """Yield content blocks of a message, tolerating non-list content."""
    content = msg.get("content")
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict):
                yield b
    elif isinstance(content, str):
        yield {"text": content}


def _extract_facts(text: str) -> List[str]:
    facts: List[str] = []
    if not text:
        return facts
    work = text
    for pat in _FACT_PATTERNS:
        for m in pat.findall(work):
            if m and m not in facts:
                facts.append(m)
        # Blank matched spans so later (broader) patterns can't carve fragments
        # out of an already-captured URL or path.
        work = pat.sub(" ", work)
    return facts


def _cap_tail(items: List[Any], cap: int) -> List[Any]:
    return items[-cap:] if len(items) > cap else items


def empty_snapshot() -> Dict[str, Any]:
    return {
        "user_turns": [],
        "tool_calls": [],
        "answer_leads": [],
        "pinned_facts": [],
        "open_threads": [],
        "turn_count": 0,
    }


def extract_delta(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract a structured snapshot fragment from a slice of Bedrock messages.

    Robust to corrupt histories: only reads block *content*, never relies on
    toolUse/toolResult pairing.
    """
    snap = empty_snapshot()
    if not messages:
        return snap

    facts: List[str] = []
    pending_answer = ""

    def _flush_answer():
        """Close out the current turn: distil its FINAL assistant message into a
        single answer_lead, dropping intermediate 'let me…' narration blocks."""
        nonlocal pending_answer
        if pending_answer:
            lead = _distill_answer(pending_answer)
            if lead:
                facts.extend(_extract_facts(lead))
                snap["answer_leads"].append(lead)
        pending_answer = ""

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            # Collect this assistant message's text + tool names. Keep only the
            # LATEST assistant message's text as the turn's pending answer, so
            # intermediate narration messages are overwritten by the final answer.
            msg_texts: List[str] = []
            for block in _iter_blocks(msg):
                if "toolUse" in block and isinstance(block["toolUse"], dict):
                    name = block["toolUse"].get("name")
                    if name:
                        snap["tool_calls"].append(name)
                    continue
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    msg_texts.append(t.strip())
            if msg_texts:
                pending_answer = "\n".join(msg_texts)
            continue
        # user / other roles
        for block in _iter_blocks(msg):
            if "toolResult" in block or "toolUse" in block:
                continue
            text = block.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if role == "user":
                clean = text.strip()
                # Skip system-injected context-warning noise.
                if clean.startswith("[Context Warning]"):
                    continue
                # A new user turn closes out the previous turn's answer.
                _flush_answer()
                facts.extend(_extract_facts(clean))
                is_question = "?" in clean
                clean = clean[:_MAX_USER_TURN_LEN]
                snap["user_turns"].append(clean)
                snap["turn_count"] += 1
                if is_question:
                    snap["open_threads"].append(clean)

    # Flush the final turn's answer (no trailing user turn to trigger it).
    _flush_answer()

    # Dedup facts preserving order.
    for fct in facts:
        if fct not in snap["pinned_facts"]:
            snap["pinned_facts"].append(fct)
    return snap


def _uniq_tail(items: List[Any], cap: int) -> List[Any]:
    """De-duplicate preserving first-seen order, then keep the tail up to cap."""
    seen = set()
    out: List[Any] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return _cap_tail(out, cap)


def merge_snapshot(prev: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a freshly-extracted delta into the accumulated snapshot, with caps.

    De-duplicating the text arrays makes the merge idempotent: re-folding an
    overlapping message range (possible across gunicorn workers) never duplicates
    entries. turn_count is derived from the de-duplicated user_turns so it stays
    consistent regardless of how the folds were sliced.
    """
    prev = prev or empty_snapshot()
    merged = {
        "user_turns": _uniq_tail(list(prev.get("user_turns", [])) + list(new.get("user_turns", [])), _MAX_USER_TURNS),
        "answer_leads": _uniq_tail(list(prev.get("answer_leads", [])) + list(new.get("answer_leads", [])), _MAX_ANSWER_LEADS),
        "open_threads": _uniq_tail(list(prev.get("open_threads", [])) + list(new.get("open_threads", [])), _MAX_OPEN_THREADS),
    }
    # tool_calls: collapse immediate repeats, keep tail.
    tools = list(prev.get("tool_calls", [])) + list(new.get("tool_calls", []))
    collapsed: List[str] = []
    for t in tools:
        if not collapsed or collapsed[-1] != t:
            collapsed.append(t)
    merged["tool_calls"] = _cap_tail(collapsed, _MAX_TOOL_CALLS)
    # pinned_facts: unique, keep tail.
    merged["pinned_facts"] = _uniq_tail(
        list(prev.get("pinned_facts", [])) + list(new.get("pinned_facts", [])), _MAX_PINNED_FACTS)
    merged["turn_count"] = len(merged["user_turns"])
    return merged


def render_content(snap: Dict[str, Any]) -> str:
    """Render the snapshot to a text blob for embedding + storage."""
    snap = snap or empty_snapshot()
    parts: List[str] = [f"Conversation checkpoint ({snap.get('turn_count', 0)} user turns summarised)."]
    if snap.get("user_turns"):
        parts.append("User asked about: " + " | ".join(snap["user_turns"][-12:]))
    if snap.get("open_threads"):
        parts.append("Open questions: " + " | ".join(snap["open_threads"][-8:]))
    if snap.get("answer_leads"):
        parts.append("Resolved: " + " | ".join(snap["answer_leads"][-8:]))
    return "\n".join(parts)


def render_context_block(snap: Dict[str, Any]) -> str:
    """Render the snapshot as a Markdown block to inject into the system prompt."""
    snap = snap or empty_snapshot()
    lines = [
        "## Recovered Conversation Context",
        "*(Earlier turns were trimmed or a prior history was cleared. This is a "
        "deterministic summary of what came before — treat it as authoritative "
        "background, not as new user input.)*",
        f"**turns_summarised:** {snap.get('turn_count', 0)}",
    ]
    if snap.get("user_turns"):
        lines.append("### Earlier user messages")
        lines.extend(f"- {u[:300]}" for u in snap["user_turns"][-15:])
    if snap.get("open_threads"):
        lines.append("### Open threads / unresolved questions")
        lines.extend(f"- {q[:300]}" for q in snap["open_threads"][-10:])
    if snap.get("answer_leads"):
        lines.append("### Prior conclusions")
        lines.extend(f"- {a}" for a in snap["answer_leads"][-8:])
    if snap.get("tool_calls"):
        lines.append("### Tools used earlier")
        lines.append(", ".join(dict.fromkeys(snap["tool_calls"])))
    if snap.get("pinned_facts"):
        lines.append("### Pinned facts (IDs / paths / URLs)")
        lines.extend(f"- {f}" for f in snap["pinned_facts"][-30:])
    return "\n".join(lines)
