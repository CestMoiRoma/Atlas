# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/db/user_db.py
===================
SQLite persistence layer for Atlas user profiles and voice embeddings.

Schema
------
``users`` table — one row per registered user:

    id              INTEGER  PRIMARY KEY AUTOINCREMENT
    name            TEXT     NOT NULL UNIQUE          -- display name
    user_tag        TEXT     NOT NULL UNIQUE          -- Obsidian tag (#name_lowercased)
    age             INTEGER                           -- optional
    gender          TEXT                              -- optional
    profession      TEXT                              -- optional
    preferred_address TEXT   NOT NULL DEFAULT ''      -- primary nickname
    other_addresses TEXT     NOT NULL DEFAULT '[]'    -- JSON list of alt nicknames
    embedding       BLOB                              -- numpy float32 array, serialised

Usage
-----
::

    from atlas.db.user_db import init_db, get_all_users, upsert_user, update_embedding
    conn = init_db(Path("atlas_users.db"))
    users = get_all_users(conn)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from atlas.core.models import User

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT    NOT NULL UNIQUE,
    user_tag            TEXT    NOT NULL UNIQUE,
    age                 INTEGER,
    gender              TEXT,
    profession          TEXT,
    preferred_address   TEXT    NOT NULL DEFAULT '',
    other_addresses     TEXT    NOT NULL DEFAULT '[]',
    embedding           BLOB
);
"""


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def init_db(path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the schema is up to date.

    Args:
        path: Filesystem path for the ``.db`` file.

    Returns:
        An open :class:`sqlite3.Connection` with ``row_factory`` set to
        :class:`sqlite3.Row` for dict-like column access.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(_DDL)
    conn.commit()
    logger.debug("Database ready → %s", path.resolve())
    return conn


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Serialise a float32 numpy array to raw bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialise a BLOB back to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# ── Queries ───────────────────────────────────────────────────────────────────

def get_all_users(conn: sqlite3.Connection) -> list["User"]:
    """Return all registered users, ordered by ``id``.

    Rows with no embedding stored yet are included; their ``embedding`` field
    will be ``None`` inside the returned :class:`~atlas.core.models.User`.
    """
    # Import here to avoid a circular import (models → db would not exist)
    from atlas.core.models import User  # noqa: PLC0415

    rows = conn.execute(
        "SELECT id, name, user_tag, age, gender, profession, "
        "preferred_address, other_addresses, embedding FROM users ORDER BY id"
    ).fetchall()

    users: list[User] = []
    for row in rows:
        embedding = _blob_to_embedding(row["embedding"]) if row["embedding"] else None
        raw_addr = row["other_addresses"] or ""
        if raw_addr:
            try:
                other = json.loads(raw_addr)
            except json.JSONDecodeError:
                # Legacy CSV format from older schemas (e.g. "Roma,Noctisse")
                other = [a.strip() for a in raw_addr.split(",") if a.strip()]
        else:
            other = []
        users.append(User(
            id=row["id"],
            name=row["name"],
            user_tag=row["user_tag"],
            age=row["age"],
            gender=row["gender"],
            profession=row["profession"],
            preferred_address=row["preferred_address"] or row["name"],
            other_addresses=other,
            embedding=embedding,
        ))
    return users


def get_user_by_name(conn: sqlite3.Connection, name: str) -> "User | None":
    """Return a single user by display name, or ``None`` if not found."""
    from atlas.core.models import User  # noqa: PLC0415

    row = conn.execute(
        "SELECT id, name, user_tag, age, gender, profession, "
        "preferred_address, other_addresses, embedding FROM users WHERE name = ?",
        (name,),
    ).fetchone()
    if row is None:
        return None

    embedding = _blob_to_embedding(row["embedding"]) if row["embedding"] else None
    raw_addr = row["other_addresses"] or ""
    if raw_addr:
        try:
            other = json.loads(raw_addr)
        except json.JSONDecodeError:
            # Legacy CSV format from older schemas (e.g. "Roma,Noctisse")
            other = [a.strip() for a in raw_addr.split(",") if a.strip()]
    else:
        other = []
    return User(
        id=row["id"],
        name=row["name"],
        user_tag=row["user_tag"],
        age=row["age"],
        gender=row["gender"],
        profession=row["profession"],
        preferred_address=row["preferred_address"] or row["name"],
        other_addresses=other,
        embedding=embedding,
    )


def upsert_user(
    conn: sqlite3.Connection,
    name: str,
    user_tag: str,
    age: int | None = None,
    gender: str | None = None,
    profession: str | None = None,
    preferred_address: str = "",
    other_addresses: list[str] | None = None,
) -> int:
    """Insert or update a user profile (does not touch the voice embedding).

    Returns:
        The ``id`` of the inserted or updated row.
    """
    other_json = json.dumps(other_addresses or [])
    preferred = preferred_address or name

    existing = conn.execute(
        "SELECT id FROM users WHERE name = ?", (name,)
    ).fetchone()

    if existing:
        conn.execute(
            """
            UPDATE users SET
                user_tag          = ?,
                age               = COALESCE(?, age),
                gender            = COALESCE(?, gender),
                profession        = COALESCE(?, profession),
                preferred_address = ?,
                other_addresses   = ?
            WHERE name = ?
            """,
            (user_tag, age, gender, profession, preferred, other_json, name),
        )
        user_id: int = existing["id"]
    else:
        conn.execute(
            """
            INSERT INTO users (name, user_tag, age, gender, profession,
                               preferred_address, other_addresses)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, user_tag, age, gender, profession, preferred, other_json),
        )
        row = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()
        user_id = row["id"]

    conn.commit()
    logger.info("User upserted → id=%d  name=%r  tag=%r", user_id, name, user_tag)
    return user_id


def update_embedding(
    conn: sqlite3.Connection,
    user_id: int,
    embedding: np.ndarray,
) -> None:
    """Persist a new (or re-averaged) voice embedding for the given user."""
    conn.execute(
        "UPDATE users SET embedding = ? WHERE id = ?",
        (_embedding_to_blob(embedding), user_id),
    )
    conn.commit()
    logger.debug("Embedding updated → user_id=%d  shape=%s", user_id, embedding.shape)


def delete_user(conn: sqlite3.Connection, user_id: int) -> None:
    """Permanently delete a user profile and their voice embedding."""
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    logger.info("User deleted → id=%d", user_id)
