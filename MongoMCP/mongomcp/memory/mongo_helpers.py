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


def normalize_oid_str(s) -> str:
    """
    Normalise any ObjectId representation to a plain 24-char hex string.

    Accepts:
    - bson ObjectId instance              → str(oid)
    - Go-compat wrapper  ObjectID("hex")  → strip and return hex
    - Plain hex string   "69e77..."       → returned as-is
    Raises ValueError for unrecognised formats so callers can catch cleanly.
    """
    from bson import ObjectId as _OID
    if isinstance(s, _OID):
        return str(s)
    raw = str(s).strip()
    if len(raw) > 12 and raw[:10].upper() == 'OBJECTID("' and raw[-2:] == '")':
        raw = raw[10:-2]
    # Validate it looks like a 24-char hex ObjectId.
    if len(raw) != 24 or not all(c in "0123456789abcdefABCDEF" for c in raw):
        raise ValueError(f"Cannot parse ObjectId from: {s!r}")
    return raw


def convert_objectids(data):
    """
    Recursively convert any bson ObjectId instances inside *data* to plain hex
    strings.  datetime.datetime objects are converted to ISO-8601 strings so
    the response is always JSON-serializable.  Other types are returned unchanged.
    """
    from bson import ObjectId as _OID
    if isinstance(data, _OID):
        return str(data)
    if isinstance(data, datetime.datetime):
        return data.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if isinstance(data, dict):
        return {k: convert_objectids(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_objectids(item) for item in data]
    return data


def get_collection(motor_client, db_name: str, collection_name: str) -> AsyncIOMotorCollection:
    """Return a motor AsyncIOMotorCollection from the given client, database, and collection name."""
    db = motor_client[db_name]
    return db[collection_name]


# Date fields that may be written as int-ms (Python) or datetime (Go/BSON).
# Epoch (≤ 1 s from Unix epoch) means "not set" and is returned as None.
_DATE_FIELDS = frozenset({"created_at", "last_accessed", "promoted_at", "expires_at"})
_EPOCH_THRESHOLD_MS = 1_000  # 1 second — anything ≤ this is treated as unset


def format_date_field(val) -> "str | None":
    """
    Normalize a date value to a human-readable ISO-8601 string.

    Accepts:
    - datetime.datetime  (Go/BSON Date read via Motor)
    - int or float       (milliseconds since Unix epoch, Python-written)
    - str                (already formatted — returned as-is)
    - None               → None

    Returns None for epoch/zero values (sentinel for "not set").
    """
    if val is None:
        return None
    if isinstance(val, str):
        return val  # already formatted
    if isinstance(val, datetime.datetime):
        ts_ms = int(val.timestamp() * 1000)
    elif isinstance(val, (int, float)):
        ts_ms = int(val)
    else:
        return None
    if ts_ms <= _EPOCH_THRESHOLD_MS:
        return None
    dt = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
    # millisecond precision, UTC, matches Go output: "2026-04-14T12:06:22.929Z"
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def strip_embedding(doc: dict, collection_name: str) -> dict:
    """Remove the 'embedding' field from a document and reformat _id for API responses.

    Also recursively converts any remaining bson ObjectId objects (e.g. inside
    linked_ids or related_docs arrays) to plain hex strings so the response is
    always JSON-serializable without bson-specific encoders.

    Known date fields (created_at, last_accessed, promoted_at, expires_at) are
    normalized to human-readable ISO-8601 strings regardless of whether they
    were written by Python (int-ms) or Go (BSON Date / datetime).
    Epoch/zero values are returned as None.
    """
    result = {k: v for k, v in doc.items() if k != "embedding"}
    if "_id" in result:
        result["id"] = format_object_id(result.pop("_id"))
    result["collection"] = collection_name
    result = convert_objectids(result)
    # Normalize known date fields after objectid conversion (values may be
    # datetime objects from Go/BSON or int-ms from Python).
    for field in _DATE_FIELDS:
        if field in result:
            result[field] = format_date_field(result[field])
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
