import argparse
import base64
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import jwt

from mongomcp.mongodb_client import MongoDBClient


def parse_scope(value: Union[str, List[str], None]) -> List[str]:
    """Parse comma-separated scope string or validate scope list."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return ["read", "write", "llm:invoke"]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def generate_jwt(agent_name: str, agent_key: str, pvk: str) -> str:
    """Generate JWT using PyJWT library with HS256 algorithm.

    pvk is stored as a plain base64 string in MongoDB and passed directly to
    PyJWT as the HMAC secret — matching how jwt.decode() verifies it in
    MongoMCPMiddleware.check_authorization().
    """
    header = {
        "alg": "HS256",
        "api_key": agent_key,
        "typ": "JWT",
    }
    payload = {
        "agent_name": agent_name,
    }
    return jwt.encode(payload, pvk, algorithm="HS256", headers=header)


def get_or_create_agent_identity_and_token(
    mongo_client: MongoDBClient,
    agent_name: str,
    agent_key: Optional[str] = None,
    pvk: Optional[str] = None,
    scope_csv: Union[str, List[str]] = "read,write,llm:invoke",
    regenerate: bool = False,
) -> Tuple[Dict[str, Any], str, bool]:
    """Read existing identity by agent_name or create/regenerate one, then return token.

    Args:
        mongo_client: MongoDBClient instance.
        agent_name: Username / agent name for the token.
        agent_key: Optional specific API key/UUID (used if creating new).
        pvk: Optional private key base64 (used if creating new).
        scope_csv: Scopes as comma-separated string or list.
        regenerate: If True, generate new keys and replace the existing identity.

    Returns:
        (metadata, token, was_created_or_regenerated)
    """
    if not mongo_client.client:
        mongo_client.sync_connect_to_mongodb()

    config_db = mongo_client.client["mcp_config"]
    collection = config_db["agent_identities"]

    existing = collection.find_one({"agent_name": agent_name}) if not regenerate else None

    if existing and not regenerate:
        existing.pop("_id", None)
        metadata = existing
        resolved_agent_key = metadata.get("agent_key")
        resolved_pvk = metadata.get("pvk")
        changed = False

        if not resolved_agent_key:
            resolved_agent_key = str(uuid.uuid4())
            metadata["agent_key"] = resolved_agent_key
            changed = True
        if not resolved_pvk:
            resolved_pvk = base64.b64encode(os.urandom(32)).decode("ascii")
            metadata["pvk"] = resolved_pvk
            changed = True
        if not metadata.get("scope"):
            metadata["scope"] = parse_scope(scope_csv)
            changed = True

        if changed:
            collection.replace_one({"agent_name": agent_name}, metadata, upsert=True)
        was_created = False
    else:
        resolved_agent_key = agent_key or str(uuid.uuid4())
        resolved_pvk = pvk or base64.b64encode(os.urandom(32)).decode("ascii")
        metadata = {
            "pvk": resolved_pvk,
            "agent_name": agent_name,
            "agent_key": resolved_agent_key,
            "scope": parse_scope(scope_csv),
        }
        collection.replace_one({"agent_name": agent_name}, metadata, upsert=True)
        was_created = True

    token = generate_jwt(
        agent_name=metadata["agent_name"],
        agent_key=metadata["agent_key"],
        pvk=metadata["pvk"],
    )

    return metadata, token, was_created


def format_mcp_server_name(endpoint: str) -> str:
    """Format an endpoint name into a VS Code MCP server key (e.g. 'mongo-memory-aws')."""
    ep = endpoint.strip()
    if ep.startswith("mongo-") or ep.startswith("mongodb-"):
        return f"{ep}-aws" if not ep.endswith("-aws") else ep
    return f"mongo-{ep}-aws"


def build_vscode_mcp_config(
    mongo_mcp_root: str,
    endpoints: List[str],
    token: str,
) -> Dict[str, Any]:
    """Build the standard VS Code mcpServers configuration dictionary.

    Args:
        mongo_mcp_root: Base URL of the MCP server from settings (fallback 'http://localhost:8000').
        endpoints: List of endpoint names (e.g. ['memory', 'agent', 'mongodb-vect2']).
        token: Bearer JWT token.

    Returns:
        Dictionary formatted for .vscode/mcp.json or VS Code settings.
    """
    root = (mongo_mcp_root or "http://localhost:8000").rstrip("/")
    if not endpoints:
        endpoints = ["memory", "agent"]

    mcp_servers = {}
    for ep in endpoints:
        server_key = format_mcp_server_name(ep)
        mcp_servers[server_key] = {
            "url": f"{root}/{ep}/mcp",
            "type": "http",
            "headers": {
                "Authorization": f"Bearer {token}",
            },
        }

    return {
        "servers": mcp_servers,
    }
