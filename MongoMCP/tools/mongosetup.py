#! python

"""Initialize MongoDB config database and required collections for MongoMCP."""

import argparse
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongomcp.mongodb_client import MongoDBClient
from pymongo.operations import SearchIndexModel
from tools.generate_jwt_token import get_or_create_agent_identity_and_token


AIR_BNB_VECTOR_SEARCH_INDEX_CONFIG = {	
  "fields": [
    {
      "type": "vector",
      "path": "embedding", # "voyage_embedding", # 
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
AIR_BNB_VECTOR_SEARCH_INDEX_NAME = "listing_vector_index" # "listing_voyage_index" # 


# ---------------------------------------------------------------------------
# Memory layer — index definitions (must match mongomcp/memory/memservice.py)
# ---------------------------------------------------------------------------
MEMORY_DB_NAME = "mcp_config"  # overridden by MEMORY_DB env var at runtime

MEMORY_EPISODIC_VECTOR_INDEX_CONFIG = {
	"fields": [
		{"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "dotProduct"},
		{"type": "filter", "path": "agent_id"},
		{"type": "filter", "path": "username"},
		{"type": "filter", "path": "session_id"},
		{"type": "filter", "path": "memory_type"},
		{"type": "filter", "path": "tags"},
		{"type": "filter", "path": "entities"},
		{"type": "filter", "path": "importance"},
		{"type": "filter", "path": "expires_at"},
		{"type": "filter", "path": "is_isolated"},
	]
}

MEMORY_SEMANTIC_VECTOR_INDEX_CONFIG = {
	"fields": [
		{"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "dotProduct"},
		{"type": "filter", "path": "agent_id"},
		{"type": "filter", "path": "username"},
		{"type": "filter", "path": "memory_type"},
		{"type": "filter", "path": "tags"},
		{"type": "filter", "path": "entities"},
		{"type": "filter", "path": "session_id"},
		{"type": "filter", "path": "is_isolated"},
	]
}

# Atlas Search (BM25 fulltext) index for the $rankFusion text leg on memory_semantic.
MEMORY_SEMANTIC_FULLTEXT_INDEX_CONFIG = {
	"mappings": {
		"dynamic": False,
		"fields": {
			"content":      {"type": "string"},
			"tags":         {"type": "string"},
			"memory_type":  {"type": "string"},
		}
	}
}
MEMORY_SEMANTIC_FULLTEXT_INDEX_NAME = "memory_semantic_fulltext_index"

MEMORY_COLLECTIONS = ["memory_episodic", "memory_semantic"]

MEMORY_EPISODIC_VECTOR_INDEX_NAME = "memory_episodic_vector_index"
MEMORY_SEMANTIC_VECTOR_INDEX_NAME = "memory_semantic_vector_index"


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


def create_mcp_config_collections(settings) -> None:
	"""Create mcp_config database collections if they do not already exist."""
	settings.mcp_config_db = "mcp_config"
	settings.mcp_config_col = "mcp_tools"

	mongo_client = MongoDBClient(settings=settings)
	mongo_client.sync_connect_to_mongodb()

	db = mongo_client.db
	required_collections = ["agent_identities", "mcp_cache", "mcp_tools", "mcp_patterns", "llm_history"] + MEMORY_COLLECTIONS
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

	search_index_model = SearchIndexModel(
		definition={
			"fields": AIR_BNB_VECTOR_SEARCH_INDEX_CONFIG["fields"]
		},
		name=AIR_BNB_VECTOR_SEARCH_INDEX_NAME,
		type="vectorSearch",
	)	
	collection.create_search_index(model=search_index_model)

	print(
		f"Created vector search index: "
		f"{AIR_BNB_DB_NAME}.{AIR_BNB_COLLECTION_NAME}.{AIR_BNB_VECTOR_SEARCH_INDEX_NAME}"
	)


def create_memory_vector_search_indexes(mongo_client: MongoDBClient) -> None:
	"""Create memory vector search indexes if they do not already exist."""
	targets = [
		("memory_episodic",  MEMORY_EPISODIC_VECTOR_INDEX_NAME,  MEMORY_EPISODIC_VECTOR_INDEX_CONFIG),
		("memory_semantic",  MEMORY_SEMANTIC_VECTOR_INDEX_NAME,  MEMORY_SEMANTIC_VECTOR_INDEX_CONFIG),
	]
	for collection_name, index_name, index_config in targets:
		collection = mongo_client.client[MEMORY_DB_NAME][collection_name]
		existing_indexes = {
			idx.get("name")
			for idx in collection.list_search_indexes()
			if idx.get("name")
		}
		if index_name in existing_indexes:
			print(f"Vector search index already exists: {MEMORY_DB_NAME}.{collection_name}.{index_name}")
			continue
		search_index_model = SearchIndexModel(
			definition={"fields": index_config["fields"]},
			name=index_name,
			type="vectorSearch",
		)
		collection.create_search_index(model=search_index_model)
		print(f"Created vector search index: {MEMORY_DB_NAME}.{collection_name}.{index_name}")


def create_memory_semantic_fulltext_index(mongo_client: MongoDBClient) -> None:
	"""Create the Atlas Search fulltext index on memory_semantic for the $rankFusion BM25 text leg."""
	collection = mongo_client.client[MEMORY_DB_NAME]["memory_semantic"]
	existing_indexes = {
		idx.get("name")
		for idx in collection.list_search_indexes()
		if idx.get("name")
	}
	if MEMORY_SEMANTIC_FULLTEXT_INDEX_NAME in existing_indexes:
		print(f"Fulltext index already exists: {MEMORY_DB_NAME}.memory_semantic.{MEMORY_SEMANTIC_FULLTEXT_INDEX_NAME}")
		return
	search_index_model = SearchIndexModel(
		definition=MEMORY_SEMANTIC_FULLTEXT_INDEX_CONFIG,
		name=MEMORY_SEMANTIC_FULLTEXT_INDEX_NAME,
		type="search",
	)
	collection.create_search_index(model=search_index_model)
	print(f"Created fulltext index: {MEMORY_DB_NAME}.memory_semantic.{MEMORY_SEMANTIC_FULLTEXT_INDEX_NAME}")


def create_and_insert_agent_identity(
	settings,
	mongo_client: MongoDBClient,
	agent_name: str = "webui_chatuser",
	agent_key: str | None = None,
	pvk: str | None = None,
	scope_csv: str = "read,write,llm:invoke",
) -> tuple[dict, str]:
	"""Get or create an agent identity, then print metadata + token."""
	metadata, token, was_created = get_or_create_agent_identity_and_token(
		mongo_client=mongo_client,
		agent_name=agent_name,
		agent_key=agent_key,
		pvk=pvk,
		scope_csv=scope_csv,
	)
	status = "Created new" if was_created else "Found existing"
	print(f"{status} agent: {agent_name}")
	print(json.dumps(metadata, indent=2, default=str))
	print("[AWS | local]_settings.py line:")
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
	create_memory_vector_search_indexes(mongo_client)
	create_memory_semantic_fulltext_index(mongo_client)
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
