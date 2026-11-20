"""Create or reset a web UI user.  Usage (from MongoMCP/webui): python manage_users.py <username> [--no-force-change]

By default the user must change this password at first login."""
import getpass
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

MONGOMCP_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MONGOMCP_DIR))
# Must run before settings are imported so they see MONGO_* / USE_LOCAL_MODE.
load_dotenv(dotenv_path=MONGOMCP_DIR / ".env.local")

if os.getenv("USE_LOCAL_MODE", "false").lower() == "true":
    from local_settings import settings
else:
    from AWS_settings import settings

from auth import create_or_update_user, get_users_collection


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--no-force-change"]
    if len(args) != 1:
        sys.exit(__doc__)
    username = args[0].strip()
    force_change = "--no-force-change" not in sys.argv
    password = getpass.getpass("Password: ")
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")
    if password != getpass.getpass("Confirm password: "):
        sys.exit("Passwords do not match.")
    create_or_update_user(get_users_collection(settings), username, password, must_change_password=force_change)
    print(f"User '{username}' saved." + (" Password change required at next login." if force_change else ""))


if __name__ == "__main__":
    main()
