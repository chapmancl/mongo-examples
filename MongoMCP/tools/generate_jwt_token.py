#! python

import argparse
import base64
import json
import os
import sys
import uuid
from typing import Any

import jwt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongomcp.mongodb_client import MongoDBClient


def parse_scope(value: str) -> list[str]:
    if not value:
        return ["read", "write", "llm:invoke"]
    return [item.strip() for item in value.split(",") if item.strip()]


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
    # PyJWT handles signing; pvk is used as-is (standard base64 string).
    return jwt.encode(payload, pvk, algorithm="HS256", headers=header)


def get_or_create_agent_identity_and_token(
    mongo_client: MongoDBClient,
    agent_name: str,
    agent_key: str | None = None,
    pvk: str | None = None,
    scope_csv: str = "read,write,llm:invoke",
) -> tuple[dict[str, Any], str, bool]:
    """Read existing identity by agent_name or create one if missing, then return token.

    Returns:
        (metadata, token, was_created)
    """
    collection = mongo_client.db["agent_identities"]
    existing = collection.find_one({"agent_name": agent_name})

    if existing:
        existing.pop("_id", None)
        metadata = existing
        resolved_agent_key = metadata.get("agent_key")
        resolved_pvk = metadata.get("pvk")
        if not resolved_agent_key:
            resolved_agent_key = str(uuid.uuid4())
            metadata["agent_key"] = resolved_agent_key
        if not resolved_pvk:
            resolved_pvk = base64.b64encode(os.urandom(32)).decode("ascii")
            metadata["pvk"] = resolved_pvk
        if not metadata.get("scope"):
            metadata["scope"] = parse_scope(scope_csv)

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
        collection.insert_one(metadata)
        was_created = True

    token = generate_jwt(
        agent_name=metadata["agent_name"],
        agent_key=metadata["agent_key"],
        pvk=metadata["pvk"],
    )

    return metadata, token, was_created


def _load_settings(use_aws: bool):
    if use_aws:
        from AWS_settings import settings
    else:
        from local_settings import settings
    return settings


def _get_settings_mongo_url(settings) -> str:
    mongo_url_value = getattr(settings, "mongo_url", None)
    if callable(mongo_url_value):
        return mongo_url_value()
    if isinstance(mongo_url_value, str) and mongo_url_value:
        return mongo_url_value
    raise ValueError("Could not resolve mongo URL from settings.mongo_url")


def run_standalone(
    agent_name: str,
    agent_key: str | None,
    pvk: str | None,
    scope_csv: str,
    use_aws: bool,
) -> None:
    settings = _load_settings(use_aws=use_aws)
    _ = _get_settings_mongo_url(settings)

    mongo_client = MongoDBClient(settings=settings)
    mongo_client.sync_connect_to_mongodb()
    mongo_client.db = mongo_client.client["mcp_config"]

    metadata, token, was_created = get_or_create_agent_identity_and_token(
        mongo_client=mongo_client,
        agent_name=agent_name,
        agent_key=agent_key,
        pvk=pvk,
        scope_csv=scope_csv,
    )

    status = "created" if was_created else "read"
    print(f"Agent identity {status}: {agent_name}")
    print(json.dumps(metadata, indent=2, default=str))
    print()
    print("JWT:")
    print(token)
    print()
    print("[AWS | local]_settings.py line:")
    print(f'AUTH_TOKEN = "{token}"')

    mongo_client.client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate/read an MCP JWT and agent metadata in mcp_config.agent_identities"
    )
    parser.add_argument("--agent-name", "--username", dest="agent_name", default="webui_chatuser", help="Agent name / username")
    parser.add_argument("--agent-key", default=None, help="Agent UUID used on create")
    parser.add_argument(
        "--pvk",
        default=None,
        help="Private key in base64 format used on create",
    )
    parser.add_argument(
        "--scope",
        default="read,write,llm:invoke",
        help='Comma-separated scopes used on create (default: "read,write,llm:invoke")',
    )
    parser.add_argument(
        "--use-aws",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use AWS settings instead of local settings",
    )

    args = parser.parse_args()
    parse_scope(args.scope)  # validate format early
    run_standalone(
        agent_name=args.agent_name,
        agent_key=args.agent_key,
        pvk=args.pvk,
        scope_csv=args.scope,
        use_aws=args.use_aws,
    )


if __name__ == "__main__":
    main()