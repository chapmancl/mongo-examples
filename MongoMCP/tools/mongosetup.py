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
		{"type": "filter", "path": "scope"},
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
		{"type": "filter", "path": "scope"},
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

# Compound regular index for efficient scope-based queries.
MEMORY_SCOPE_INDEX_NAME = "memory_scope_compound_idx"
MEMORY_SCOPE_INDEX_SPEC = [("scope", 1), ("agent_id", 1), ("username", 1), ("session_id", 1)]


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


def create_mcp_cache_indexes(mongo_client: MongoDBClient) -> None:
	"""Create indexes on mcp_cache for both embedded and per-document cache modes.

	Embedded mode (tool_discovery): one doc per (username, session_id) — already
	unique-indexed by the MongoSessionCache itself on first use.

	Per-document mode (tool_response): one doc per cache entry — requires a
	compound unique index and a TTL index for automatic expiry.
	"""
	collection = mongo_client.client["mcp_config"]["mcp_cache"]
	existing = {idx["name"] for idx in collection.list_indexes()}

	# Drop the old (username, session_id) unique index if it still exists — it conflicts
	# with per-document mode entries that share the same (username, session_id).
	emb_old_idx = "mcp_cache_username_session_id_unique"
	if emb_old_idx in existing:
		collection.drop_index(emb_old_idx)
		print(f"Dropped old index: mcp_config.mcp_cache.{emb_old_idx}")

	# Embedded mode: unique per (username, session_id, cache_object_name) for docs with doc_type="embedded".
	# Partial filter uses doc_type equality — Atlas supports $eq in partial filter expressions.
	emb_idx = "mcp_cache_embedded_unique"
	if emb_idx not in existing:
		collection.create_index(
			[("username", 1), ("session_id", 1), ("cache_object_name", 1)],
			unique=True,
			partialFilterExpression={"doc_type": {"$eq": "embedded"}},
			name=emb_idx,
		)
		print(f"Created index: mcp_config.mcp_cache.{emb_idx}")
	else:
		print(f"Index already exists: mcp_config.mcp_cache.{emb_idx}")

	# Per-document mode: unique per cache entry.
	entry_idx = "mcp_cache_entry_unique"
	if entry_idx not in existing:
		collection.create_index(
			[("username", 1), ("session_id", 1), ("cache_object_name", 1), ("cache_key", 1)],
			unique=True,
			sparse=True,  # docs without cache_key (embedded mode) are excluded
			name=entry_idx,
		)
		print(f"Created index: mcp_config.mcp_cache.{entry_idx}")
	else:
		print(f"Index already exists: mcp_config.mcp_cache.{entry_idx}")

	# TTL index — MongoDB auto-deletes per-document entries when expires_at is reached.
	ttl_idx = "mcp_cache_entry_ttl"
	if ttl_idx not in existing:
		collection.create_index(
			[("expires_at", 1)],
			expireAfterSeconds=0,
			sparse=True,  # embedded-mode docs have no expires_at
			name=ttl_idx,
		)
		print(f"Created TTL index: mcp_config.mcp_cache.{ttl_idx}")
	else:
		print(f"TTL index already exists: mcp_config.mcp_cache.{ttl_idx}")


def create_memory_scope_compound_index(mongo_client: MongoDBClient) -> None:
	"""Create a compound regular index on {scope, agent_id, username, session_id} for both memory collections.

	Also runs a migration to backfill scope=0 (SCOPE_SHARED) on any document that
	has no scope field yet, preserving backwards compatibility with legacy docs.
	"""
	from pymongo import ASCENDING, IndexModel as PyIndexModel

	for coll_name in MEMORY_COLLECTIONS:
		collection = mongo_client.client[MEMORY_DB_NAME][coll_name]

		# Check for the index by key spec — it may exist under an auto-generated name.
		existing_indexes = list(collection.list_indexes())
		existing_names = {idx["name"] for idx in existing_indexes}
		target_key = dict(MEMORY_SCOPE_INDEX_SPEC)
		scope_index_exists = any(
			dict(idx.get("key", {})) == target_key
			for idx in existing_indexes
		)

		if not scope_index_exists:
			collection.create_index(MEMORY_SCOPE_INDEX_SPEC, name=MEMORY_SCOPE_INDEX_NAME)
			print(f"Created compound scope index: {MEMORY_DB_NAME}.{coll_name}.{MEMORY_SCOPE_INDEX_NAME}")
		else:
			print(f"Compound scope index already exists: {MEMORY_DB_NAME}.{coll_name} (skipping)")

		# Migration: set scope=0 on legacy docs without the field.
		result = collection.update_many(
			{"scope": {"$exists": False}},
			{"$set": {"scope": 0}},
		)
		if result.modified_count:
			print(f"  Migrated {result.modified_count} legacy docs to scope=0 in {coll_name}")
		else:
			print(f"  No legacy docs to migrate in {coll_name}")


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
	load_tools: bool = False,
	use_aws: bool = False,
) -> None:
	settings = _load_settings(use_aws=use_aws)
	create_mcp_config_collections(settings)

	mongo_client = MongoDBClient(settings=settings)
	mongo_client.sync_connect_to_mongodb()
	if load_tools:
		load_and_insert_mcp_tools(settings, mongo_client)
	else:
		print("Skipping mcp_tools load (use --load-tools to overwrite)")
	create_airbnb_vector_search_index(mongo_client)
	create_mcp_cache_indexes(mongo_client)
	create_memory_vector_search_indexes(mongo_client)
	create_memory_semantic_fulltext_index(mongo_client)
	create_memory_scope_compound_index(mongo_client)
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
	parser.add_argument(
		"--load-tools",
		action="store_true",
		default=False,
		help="Overwrite mcp_config.mcp_tools from mcp_config.mcp_tools.json (destructive — off by default)",
	)
	parser.add_argument(
		"--aws",
		action="store_true",
		default=False,
		help="Use AWS_settings.py credentials instead of local_settings.py",
	)
	args = parser.parse_args()

	run_setup(
		seed_agent_identity=args.seed_agent_identity,
		agent_name=args.agent_name,
		load_tools=args.load_tools,
		use_aws=args.aws,
	)


if __name__ == "__main__":
	main()
