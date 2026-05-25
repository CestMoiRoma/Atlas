# SPDX-License-Identifier: AGPL-3.0-or-later
"""
scripts/edit_user.py
====================
Edit individual profile fields for an existing Atlas user.

Usage::

    python scripts/edit_user.py --name "Roma" --profession "CTO"
    python scripts/edit_user.py --name "Roma" --age 29 --preferred-address "chef"
    python scripts/edit_user.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from atlas.config import Config, ConfigError
from atlas.db.user_db import get_all_users, get_user_by_name, init_db, upsert_user


def _list_users(db) -> None:
    users = get_all_users(db)
    if not users:
        print("No registered users.")
        return
    print(f"\n{'Name':<20} {'Tag':<20} {'Age':<5} {'Profession':<25} {'Address'}")
    print("─" * 80)
    for u in users:
        emb = "✓" if u.embedding is not None else "✗"
        print(
            f"{u.name:<20} {u.user_tag:<20} {str(u.age or ''):<5} "
            f"{str(u.profession or ''):<25} {u.preferred_address}  [emb:{emb}]"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edit an existing Atlas user profile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", default=None, help="User to edit")
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--gender", default=None)
    parser.add_argument("--profession", default=None)
    parser.add_argument("--preferred-address", dest="preferred_address", default=None)
    parser.add_argument("--list", action="store_true", help="List all registered users")
    args = parser.parse_args()

    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"\n[Error] {exc}\n", file=sys.stderr)
        sys.exit(1)

    db = init_db(config.speaker_db_path)

    if args.list:
        _list_users(db)
        return

    if not args.name:
        parser.error("--name is required unless --list is used")

    user = get_user_by_name(db, args.name)
    if user is None:
        print(f"\n[Error] User {args.name!r} not found. Use register_user.py to create.", file=sys.stderr)
        sys.exit(1)

    upsert_user(
        db,
        name=user.name,
        user_tag=user.user_tag,
        age=args.age if args.age is not None else user.age,
        gender=args.gender or user.gender,
        profession=args.profession or user.profession,
        preferred_address=args.preferred_address or user.preferred_address,
    )
    print(f"\n  \033[32m✓\033[0m  {args.name!r} updated successfully.\n")


if __name__ == "__main__":
    main()
