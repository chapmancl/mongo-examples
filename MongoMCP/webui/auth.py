"""Cookie-session login for the web UI. Users live in mcp_config.webui_users with hashed passwords."""
import logging
import os
import secrets
import time
from typing import Optional

from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)

USERS_COLLECTION = "webui_users"
PROTECTED_PATHS = ("/query", "/reset", "/pattern/save", "/feedback")
_DUMMY_HASH = generate_password_hash(secrets.token_hex(8))
_FAIL_WINDOW = 300
_FAIL_LIMIT = 10
MIN_PASSWORD_LENGTH = 8
_failures: dict = {}

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")
_collection = None
_settings = None


def get_users_collection(settings):
    from mongomcp.mongodb_client import MongoDBClient
    client = MongoDBClient(settings=settings)
    client.sync_connect_to_mongodb()
    coll = client.db[USERS_COLLECTION]
    coll.create_index("username", unique=True)
    return coll


def create_or_update_user(coll, username: str, password: str, must_change_password: bool = True) -> None:
    coll.update_one(
        {"username": username},
        {"$set": {"password_hash": generate_password_hash(password),
                  "must_change_password": must_change_password,
                  "updated_at": time.time()},
         "$setOnInsert": {"created_at": time.time()}},
        upsert=True,
    )


def _users():
    global _collection
    if _collection is None:
        _collection = get_users_collection(_settings)
    return _collection


def _too_many_failures(key: str) -> bool:
    now = time.time()
    recent = [t for t in _failures.get(key, []) if now - t < _FAIL_WINDOW]
    _failures[key] = recent
    return len(recent) >= _FAIL_LIMIT


def current_username() -> Optional[str]:
    return session.get("username")


@auth_bp.post("/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    key = f"{request.remote_addr}:{username.lower()}"
    if _too_many_failures(key):
        return jsonify({"error": "Too many failed attempts. Try again later."}), 429
    try:
        user = _users().find_one({"username": username}) if username else None
    except Exception:
        logger.exception("User lookup failed")
        return jsonify({"error": "Login service unavailable"}), 503
    # Always run a hash check so response time doesn't reveal whether the user exists.
    ok = check_password_hash(user["password_hash"] if user else _DUMMY_HASH, password)
    if not (user and ok):
        _failures.setdefault(key, []).append(time.time())
        return jsonify({"error": "Invalid username or password"}), 401
    _failures.pop(key, None)
    session.clear()
    session["username"] = username
    session["must_change_password"] = bool(user.get("must_change_password"))
    session.permanent = True
    return jsonify({"username": username, "must_change_password": session["must_change_password"]})


@auth_bp.post("/change-password")
def change_password():
    username = current_username()
    if not username:
        return jsonify({"error": "Authentication required"}), 401
    payload = request.get_json(silent=True) or {}
    current = str(payload.get("current_password", ""))
    new = str(payload.get("new_password", ""))
    if len(new) < MIN_PASSWORD_LENGTH:
        return jsonify({"error": f"New password must be at least {MIN_PASSWORD_LENGTH} characters"}), 400
    if new == current:
        return jsonify({"error": "New password must differ from the current password"}), 400
    key = f"{request.remote_addr}:{username.lower()}"
    if _too_many_failures(key):
        return jsonify({"error": "Too many failed attempts. Try again later."}), 429
    user = _users().find_one({"username": username})
    if not user or not check_password_hash(user["password_hash"], current):
        _failures.setdefault(key, []).append(time.time())
        return jsonify({"error": "Current password is incorrect"}), 401
    _users().update_one(
        {"username": username},
        {"$set": {"password_hash": generate_password_hash(new),
                  "must_change_password": False, "updated_at": time.time()}},
    )
    session["must_change_password"] = False
    return jsonify({"username": username, "must_change_password": False})


@auth_bp.post("/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@auth_bp.get("/me")
def me():
    username = current_username()
    if not username:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify({"username": username, "must_change_password": bool(session.get("must_change_password"))})


def init_auth(app, settings) -> None:
    global _settings
    _settings = settings
    key = os.getenv("WEBUI_SECRET_KEY")
    if not key:
        key = secrets.token_hex(32)
        logger.warning("WEBUI_SECRET_KEY not set; using a random key. Sessions reset on restart "
                       "and will not work across multiple gunicorn workers.")
    app.config.update(
        SECRET_KEY=key,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("WEBUI_COOKIE_SECURE", "false").lower() == "true",
        PERMANENT_SESSION_LIFETIME=int(os.getenv("WEBUI_SESSION_HOURS", "12")) * 3600,
    )
    app.register_blueprint(auth_bp)

    @app.before_request
    def require_login():
        if request.method == "OPTIONS":
            return None
        if any(request.path == p or request.path.startswith(p + "/") for p in PROTECTED_PATHS):
            if not current_username():
                return jsonify({"error": "Authentication required"}), 401
            if session.get("must_change_password"):
                return jsonify({"error": "Password change required", "password_change_required": True}), 403
        return None
