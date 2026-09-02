#!/usr/bin/env python3
"""
Local smoke test for the LibertyAIR integration.

Validates, against the real endpoint, that:
  1. The Azure AD client-credentials grant returns a token (credentials OK).
  2. The Bedrock Converse route accepts a request (the `bedrock access role`
     has been granted to the client) and returns a completion.

Run:
    python scripts/smoke_libertyair.py
    python scripts/smoke_libertyair.py --tools   # also exercise a tool call

Config comes from env vars (same names the deployment injects), with the
test-environment defaults baked in:
    SECRET__GEM_LIBERTYAIR_CLIENT__ID       (client id)
    SECRET__GEM_LIBERTYAIR_CLIENT__SECRET   (client secret)
    LIBERTY_AIR_URL        default https://test-libertyair.lmig.com
    LIBERTY_CLIENT_SCOPE   default 87d1c382-6128-4150-aacf-bb624a9f2748/.default
    AZURE_TENANT_ID        default 08a83339-90e7-49bf-9075-957ccd561bf1
    LIBERTY_TROUX_ID       default B7135EEA-F02B-44BC-B19E-70D195A9C6E1
    LLM_MODEL_ID           default us.anthropic.claude-sonnet-4-6
    CERTIFICATE_PATH       default certs/lm-ca-bundle.crt
"""

import argparse
import json
import os
import sys

# Import the LibertyAIR modules directly (as top-level modules) so we do not
# trigger mongomcp/__init__.py, which pulls in the full server stack.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mongomcp"))

from libertyair_converse import LibertyAIRConverseClient  # noqa: E402
from libertyair_token import LibertyAIRTokenProvider  # noqa: E402

DEFAULTS = {
    "LIBERTY_AIR_URL": "https://test-libertyair.lmig.com",
    "LIBERTY_CLIENT_SCOPE": "87d1c382-6128-4150-aacf-bb624a9f2748/.default",
    "AZURE_TENANT_ID": "08a83339-90e7-49bf-9075-957ccd561bf1",
    "LIBERTY_TROUX_ID": "B7135EEA-F02B-44BC-B19E-70D195A9C6E1",
    "LLM_MODEL_ID": "us.anthropic.claude-sonnet-4-6",
    "CERTIFICATE_PATH": "certs/lm-ca-bundle.crt",
}


def load_env_file() -> None:
    """Load KEY=VALUE pairs from a local `.env.libertyair` (gitignored) so the
    smoke test can be run without exporting secrets into the shell.

    Looked up in the repo root and in scripts/. Existing env vars win, so an
    explicit `export` still overrides the file.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, ".env.libertyair"),
        os.path.join(repo_root, "scripts", ".env.libertyair"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ.setdefault(key, value)
        print(f"(loaded env from {path})")


def cfg(name: str) -> str:
    return os.getenv(name, DEFAULTS.get(name, ""))


def mask(secret: str) -> str:
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "****"
    return f"{secret[:4]}…{secret[-4:]} (len={len(secret)})"


def main() -> int:
    parser = argparse.ArgumentParser(description="LibertyAIR smoke test")
    parser.add_argument("--tools", action="store_true", help="also exercise a tool call")
    args = parser.parse_args()

    load_env_file()

    client_id = os.getenv("SECRET__GEM_LIBERTYAIR_CLIENT__ID", "")
    client_secret = os.getenv("SECRET__GEM_LIBERTYAIR_CLIENT__SECRET", "")
    cert = cfg("CERTIFICATE_PATH")
    cert = cert if cert and os.path.exists(cert) else None

    print("=== LibertyAIR config ===")
    print(f"  base_url : {cfg('LIBERTY_AIR_URL')}")
    print(f"  tenant   : {cfg('AZURE_TENANT_ID')}")
    print(f"  scope    : {cfg('LIBERTY_CLIENT_SCOPE')}")
    print(f"  troux_id : {cfg('LIBERTY_TROUX_ID')}")
    print(f"  model    : {cfg('LLM_MODEL_ID')}")
    print(f"  cert     : {cert or '<system default>'}")
    print(f"  client_id: {mask(client_id)}")
    print(f"  secret   : {mask(client_secret)}")

    if not client_id or not client_secret:
        print(
            "\nERROR: client id/secret not found in env. Set "
            "SECRET__GEM_LIBERTYAIR_CLIENT__ID and SECRET__GEM_LIBERTYAIR_CLIENT__SECRET "
            "(deployment injects these).",
            file=sys.stderr,
        )
        return 2

    # ── Step 1: token ────────────────────────────────────────────────────────
    print("\n=== Step 1: Azure AD client-credentials token ===")
    provider = LibertyAIRTokenProvider(
        tenant_id=cfg("AZURE_TENANT_ID"),
        client_id=client_id,
        client_secret=client_secret,
        scope=cfg("LIBERTY_CLIENT_SCOPE"),
        verify=cert,
    )
    try:
        token = provider.get_token()
    except Exception as e:  # noqa: BLE001 - smoke test wants the full reason
        print(f"  FAILED to fetch token: {e}", file=sys.stderr)
        return 1
    print(f"  OK — token {mask(token)}")

    # ── Step 2: Converse call ─────────────────────────────────────────────────
    print("\n=== Step 2: Bedrock Converse call ===")
    converse = LibertyAIRConverseClient(
        base_url=cfg("LIBERTY_AIR_URL"),
        token_provider=provider,
        troux_id=cfg("LIBERTY_TROUX_ID"),
        verify=cert,
    )
    try:
        resp = converse.converse(
            modelId=cfg("LLM_MODEL_ID"),
            messages=[{"role": "user", "content": [{"text": "Reply with exactly: LibertyAIR OK"}]}],
            inferenceConfig={"maxTokens": 64, "temperature": 0.0},
        )
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED Converse call: {e}", file=sys.stderr)
        print(
            "  Hint: a 403/permission error usually means the `bedrock access role` "
            "has not been granted to this client — contact the LibertyAIR team.",
            file=sys.stderr,
        )
        return 1

    text = "".join(
        b.get("text", "") for b in resp.get("output", {}).get("message", {}).get("content", [])
    )
    print(f"  OK — model said: {text!r}")
    print(f"  usage: {json.dumps(resp.get('usage', {}))}  stopReason: {resp.get('stopReason')}")

    # ── Step 3: optional tool-calling check ───────────────────────────────────
    if args.tools:
        print("\n=== Step 3: tool-calling (toolConfig) ===")
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            }
                        },
                    }
                }
            ]
        }
        try:
            resp = converse.converse(
                modelId=cfg("LLM_MODEL_ID"),
                messages=[{"role": "user", "content": [{"text": "What is the weather in Boston?"}]}],
                toolConfig=tool_config,
                inferenceConfig={"maxTokens": 256, "temperature": 0.0},
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED tool call: {e}", file=sys.stderr)
            return 1
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        tool_uses = [b["toolUse"] for b in blocks if "toolUse" in b]
        print(f"  stopReason: {resp.get('stopReason')}  toolUse blocks: {len(tool_uses)}")
        for tu in tool_uses:
            print(f"    -> {tu.get('name')}({json.dumps(tu.get('input'))})")
        if not tool_uses:
            print("  WARNING: expected a toolUse block but got none.")

    print("\nAll checks passed ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
