#! python

import argparse
import base64
import hashlib
import hmac
import json
import os
import uuid


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _parse_scope(value: str) -> list[str]:
    if not value:
        return ["read", "write", "llm:invoke"]
    return [item.strip() for item in value.split(",") if item.strip()]


def _decode_pvk(pvk: str) -> bytes:
    padding = "=" * (-len(pvk) % 4)
    return base64.b64decode(pvk + padding, validate=True)


def generate_jwt(agent_name: str, agent_key: str, pvk: str, scope: list[str]) -> str:
    header = {
        "alg": "HS256",
        "api_key": agent_key,
        "typ": "JWT",
    }
    payload = {
        "agent_name": agent_name,        
    }

    header_part = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_part = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_part}.{payload_part}".encode("utf-8")

    secret = _decode_pvk(pvk)
    signature = hmac.new(secret, signing_input, hashlib.sha256).digest()
    signature_part = _b64url(signature)

    return f"{header_part}.{payload_part}.{signature_part}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an MCP-style HS256 JWT and metadata"
    )
    parser.add_argument("--agent-name", default="console_chatbot", help="Agent name")
    parser.add_argument("--agent-key", default=None, help="Agent UUID (default: random uuid4)")
    parser.add_argument(
        "--pvk",
        default=None,
        help="Private key in base64 format (default: random 32-byte key)",
    )
    parser.add_argument(
        "--scope",
        default="read,write,llm:invoke",
        help='Comma-separated scopes (default: "read,write,llm:invoke")',
    )

    args = parser.parse_args()

    agent_key = args.agent_key or str(uuid.uuid4())
    pvk = args.pvk or base64.b64encode(os.urandom(32)).decode("ascii")
    scope = _parse_scope(args.scope)

    try:
        _decode_pvk(pvk)
    except Exception as exc:
        raise SystemExit(f"Invalid --pvk value, expected base64: {exc}")

    metadata = {
        "pvk": pvk,
        "agent_name": args.agent_name,
        "agent_key": agent_key,
        "scope": scope,
    }

    token = generate_jwt(
        agent_name=args.agent_name,
        agent_key=agent_key,
        pvk=pvk,
        scope=scope,
    )

    print(json.dumps(metadata, indent=2))
    print()
    print("JWT:")
    print(token)
    print()
    print("settings.py line:")
    print(f'AUTH_TOKEN = "{token}"')


if __name__ == "__main__":
    main()