"""Verified MCP identity data used by memory ownership checks."""

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Identity:
    """Canonical identity and administrator-managed legacy usernames."""

    username: str
    agent_id: str
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_agent_record(cls, agent_record: dict[str, Any]) -> "Identity":
        agent_name = str(agent_record.get("agent_name") or "").strip()
        agent_id = str(agent_record.get("agent_id") or agent_name).strip()
        aliases = _clean_names(agent_record.get("aliases", ()))
        if not agent_name:
            raise ValueError("agent identity record is missing agent_name")
        return cls(username=agent_name, agent_id=agent_id, aliases=aliases)

    def matches_username(self, candidate: Optional[str]) -> bool:
        """Return whether a stored username is canonical or a verified alias."""
        if not candidate:
            return False
        normalized = candidate.strip().casefold()
        return normalized in {
            name.casefold() for name in (self.username, *self.aliases)
        }


def _clean_names(names: Iterable[Any]) -> tuple[str, ...]:
    """Normalize configured names and remove duplicates and the canonical name."""
    if isinstance(names, (str, bytes)):
        names = (names,)
    cleaned: list[str] = []
    seen: set[str] = set()
    for name in names or ():
        value = str(name).strip()
        key = value.casefold()
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)
    return tuple(cleaned)