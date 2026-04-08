#! python

"""Initialize MongoDB config database and required collections for MongoMCP."""

import argparse
import base64
import hashlib
import hmac
import json
import os
import uuid
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongomcp.mongodb_client import MongoDBClient


AIR_BNB_VECTOR_SEARCH_INDEX_CONFIG = {	
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    {
      "path": "address.country_code",
      "type": "filter"
    },
    {
      "path": "address.market",
      "type": "filter"
    },
    {
      "path": "beds",
      "type": "filter"
    },
    {
      "path": "bedrooms",
      "type": "filter"
    },
    {
      "path": "address.suburb",
      "type": "filter"
    },
    {
      "path": "property_type",
      "type": "filter"
    }
  ]
}

AIR_BNB_DB_NAME = "sample_airbnb"
AIR_BNB_COLLECTION_NAME = "listingsAndReviews"
AIR_BNB_VECTOR_SEARCH_INDEX_NAME = "listing_vector_index"



def _load_settings(use_aws: bool):
	if use_aws:
		from AWS_settings import settings  # pylint: disable=import-outside-toplevel
	else:
		from local_settings import settings  # pylint: disable=import-outside-toplevel
	return settings


def _get_settings_mongo_url(settings) -> str:
	mongo_url_value = getattr(settings, "mongo_url", None)
	if callable(mongo_url_value):
		return mongo_url_value()
	if isinstance(mongo_url_value, str) and mongo_url_value:
		return mongo_url_value
	raise ValueError("Could not resolve mongo URL from settings.mongo_url")


def _b64url(data: bytes) -> str:
	return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode_pvk(pvk: str) -> bytes:
	padding = "=" * (-len(pvk) % 4)
	return base64.b64decode(pvk + padding, validate=True)


def _parse_scope(value: str) -> list[str]:
	if not value:
		return ["read", "write", "llm:invoke"]
	return [item.strip() for item in value.split(",") if item.strip()]


def _generate_jwt(agent_name: str, agent_key: str, pvk: str) -> str:
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


def create_mcp_config_collections(settings) -> None:
	"""Create mcp_config database collections if they do not already exist."""
	settings.mcp_config_db = "mcp_config"
	settings.mcp_config_col = "mcp_tools"

	mongo_client = MongoDBClient(settings=settings)
	mongo_client.sync_connect_to_mongodb()

	db = mongo_client.db
	required_collections = ["agent_identities", "mcp_cache", "mcp_tools", "mcp_patterns", "llm_history"]
	existing_collections = set(db.list_collection_names())

	print("Connected to database: mcp_config")
	for collection_name in required_collections:
		if collection_name in existing_collections:
			print(f"Collection already exists: {collection_name}")
			continue
		db.create_collection(collection_name)
		print(f"Created collection: {collection_name}")


def load_and_insert_mcp_tools(settings, mongo_client: MongoDBClient) -> None:
	"""Read mcp_tools config JSON, rewrite module_info.url, and upsert into mcp_config.mcp_tools."""
	config_path = os.path.join(os.path.dirname(__file__), "mcp_config.mcp_tools.json")
	with open(config_path, "r", encoding="utf-8") as infile:
		tool_docs = json.load(infile)

	mongo_url = _get_settings_mongo_url(settings)
	for tool_doc in tool_docs:
		module_info = tool_doc.get("module_info", {})
		module_info["url"] = mongo_url
		tool_doc["module_info"] = module_info

	collection = mongo_client.db["mcp_tools"]
	inserted = 0
	updated = 0
	for tool_doc in tool_docs:
		name = tool_doc.get("Name")
		if not name:
			raise ValueError("Each tool document must include 'Name'.")

		result = collection.replace_one({"Name": name}, tool_doc, upsert=True)
		if result.upserted_id is not None:
			inserted += 1
		elif result.modified_count > 0:
			updated += 1

	print(f"mcp_tools sync complete. inserted={inserted}, updated={updated}, total={len(tool_docs)}")


def create_airbnb_vector_search_index(mongo_client: MongoDBClient) -> None:
	"""Create the Airbnb vector search index if it does not already exist."""
	collection = mongo_client.client[AIR_BNB_DB_NAME][AIR_BNB_COLLECTION_NAME]

	existing_indexes = {
		index_doc.get("name")
		for index_doc in collection.list_search_indexes()
		if index_doc.get("name")
	}

	if AIR_BNB_VECTOR_SEARCH_INDEX_NAME in existing_indexes:
		print(
			f"Vector search index already exists: "
			f"{AIR_BNB_DB_NAME}.{AIR_BNB_COLLECTION_NAME}.{AIR_BNB_VECTOR_SEARCH_INDEX_NAME}"
		)
		return

	collection.create_search_index(
		name=AIR_BNB_VECTOR_SEARCH_INDEX_NAME,
		definition=AIR_BNB_VECTOR_SEARCH_INDEX_CONFIG,
	)
	print(
		f"Created vector search index: "
		f"{AIR_BNB_DB_NAME}.{AIR_BNB_COLLECTION_NAME}.{AIR_BNB_VECTOR_SEARCH_INDEX_NAME}"
	)


def create_and_insert_agent_identity(
	settings,
	mongo_client: MongoDBClient,
	agent_name: str = "webui_chatuser",
	agent_key: str | None = None,
	pvk: str | None = None,
	scope_csv: str = "read,write,llm:invoke",
) -> tuple[dict, str]:
	"""Generate JWT metadata, upsert into mcp_config.agent_identities, and print AUTH_TOKEN line."""
	resolved_agent_key = agent_key or str(uuid.uuid4())
	resolved_pvk = pvk or base64.b64encode(os.urandom(32)).decode("ascii")
	scope = _parse_scope(scope_csv)

	# Validate the provided/generated key is valid base64.
	_decode_pvk(resolved_pvk)

	token = _generate_jwt(
		agent_name=agent_name,
		agent_key=resolved_agent_key,
		pvk=resolved_pvk,
	)
	metadata = {
		"pvk": resolved_pvk,
		"agent_name": agent_name,
		"agent_key": resolved_agent_key,
		"scope": scope,
	}

	collection = mongo_client.db["agent_identities"]
	collection.replace_one({"agent_name": agent_name}, metadata, upsert=True)

	print(json.dumps(metadata, indent=2))
	print()
	print("JWT:")
	print(token)
	print()
	print("settings.py line:")
	print(f'AUTH_TOKEN = "{token}"')

	return metadata, token


def run_setup(	
	seed_agent_identity: bool = True,
	agent_name: str = "webui_chatuser",
) -> None:
	settings = _load_settings(use_aws=False)
	create_mcp_config_collections(settings)

	mongo_client = MongoDBClient(settings=settings)
	mongo_client.sync_connect_to_mongodb()
	load_and_insert_mcp_tools(settings, mongo_client)
	create_airbnb_vector_search_index(mongo_client)
	if seed_agent_identity:
		create_and_insert_agent_identity(
			settings=settings,
			mongo_client=mongo_client,
			agent_name=agent_name,
		)

	mongo_client.client.close()
	print("MongoDB setup complete.")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Create MongoMCP config database and collections"
	)
	parser.add_argument(
		"--seed-agent-identity",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Generate JWT metadata and upsert into mcp_config.agent_identities (default: enabled)",
	)
	parser.add_argument(
		"--agent-name",
		default="webui_chatuser",
		help="Agent name used when --seed-agent-identity is provided",
	)
	args = parser.parse_args()

	run_setup(		
		seed_agent_identity=args.seed_agent_identity,
		agent_name=args.agent_name,
	)


if __name__ == "__main__":
	main()
