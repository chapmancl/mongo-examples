"""
Low-level MongoDB helpers for the memory subpackage.

Provides stateless utility functions used by MemoryService.
No business logic lives here.
"""

import re
import time
import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection


def now_ms() -> int:
    """Return current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


def to_ms(created_at) -> int:
    """Coerce created_at to milliseconds int, handling both int (Python intake)
    and datetime.datetime (Go/BSON intake) values."""
    if isinstance(created_at, (int, float)):
        return int(created_at)
    if isinstance(created_at, datetime.datetime):
        return int(created_at.timestamp() * 1000)
    # Fallback: treat as current time so age=0
    return now_ms()


def format_object_id(oid) -> str:
    """Format a bson ObjectId as the Go-compatible string 'ObjectID(\"...\")'."""
    return f'ObjectID("{str(oid)}")'


def get_collection(motor_client, db_name: str, collection_name: str) -> AsyncIOMotorCollection:
    """Return a motor AsyncIOMotorCollection from the given client, database, and collection name."""
    db = motor_client[db_name]
    return db[collection_name]


def strip_embedding(doc: dict, collection_name: str) -> dict:
    """Remove the 'embedding' field from a document and reformat _id for API responses."""
    result = {k: v for k, v in doc.items() if k != "embedding"}
    if "_id" in result:
        result["id"] = format_object_id(result.pop("_id"))
    result["collection"] = collection_name
    return result


def split_regex_filters(filter_dict: dict) -> tuple:
    """
    Separate $regex conditions from a filter dict.

    Returns (mongo_safe_filter, [(field, compiled_regex), ...]).
    Atlas vectorSearch pre-filters do not support $regex; callers should
    apply the returned regex list as a post-filter in Python after the query.
    """
    safe: dict = {}
    regex_filters: list = []
    for field, value in filter_dict.items():
        if isinstance(value, dict) and "$regex" in value:
            pattern = value["$regex"]
            options = value.get("$options", "")
            flags = re.IGNORECASE if "i" in options.lower() else 0
            regex_filters.append((field, re.compile(pattern, flags)))
        else:
            safe[field] = value
    return safe, regex_filters


def apply_regex_post_filters(docs: list, regex_filters: list) -> list:
    """Filter a list of documents using compiled regex conditions extracted by split_regex_filters."""
    if not regex_filters:
        return docs
    result = []
    for doc in docs:
        match = True
        for field, pattern in regex_filters:
            val = doc.get(field)
            if not isinstance(val, str) or not pattern.search(val):
                match = False
                break
        if match:
            result.append(doc)
    return result
