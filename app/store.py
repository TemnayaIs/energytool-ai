"""Хранилище диалогов и заявок на SQLite (stdlib, без ORM)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'Новый диалог',
                state TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                meta TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                company TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            CREATE INDEX IF NOT EXISTS idx_leads_session ON leads(session_id);
            """
        )


def create_session(title: str = "Новый диалог") -> dict:
    session_id = uuid.uuid4().hex
    now = _now()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO sessions (id, title, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, json.dumps({}, ensure_ascii=False), now, now),
        )
    return {"id": session_id, "title": title, "state": {}, "created_at": now, "updated_at": now}


def get_session(session_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    try:
        data["state"] = json.loads(data.get("state") or "{}")
    except json.JSONDecodeError:
        data["state"] = {}
    return data


def save_state(session_id: str, state: dict, title: str | None = None) -> None:
    with _connect() as connection:
        if title:
            connection.execute(
                "UPDATE sessions SET state = ?, title = ?, updated_at = ? WHERE id = ?",
                (json.dumps(state, ensure_ascii=False), title, _now(), session_id),
            )
        else:
            connection.execute(
                "UPDATE sessions SET state = ?, updated_at = ? WHERE id = ?",
                (json.dumps(state, ensure_ascii=False), _now(), session_id),
            )


def add_message(session_id: str, role: str, content: str, meta: dict | None = None) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(meta or {}, ensure_ascii=False), _now()),
        )
        connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))


def get_messages(session_id: str) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT role, content, meta, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["meta"] = json.loads(item.get("meta") or "{}")
        except json.JSONDecodeError:
            item["meta"] = {}
        result.append(item)
    return result


def list_sessions(limit: int = 20) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count,
                   (SELECT COUNT(*) FROM leads l WHERE l.session_id = s.id) AS has_lead
            FROM sessions s
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_lead(session_id: str, name: str, phone: str, company: str, summary: str) -> int:
    with _connect() as connection:
        cursor = connection.execute(
            "INSERT INTO leads (session_id, name, phone, company, summary, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, name, phone, company, summary, _now()),
        )
        return int(cursor.lastrowid)


def get_lead(session_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM leads WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def get_session_bundle(session_id: str) -> dict | None:
    session = get_session(session_id)
    if not session:
        return None
    session["messages"] = get_messages(session_id)
    session["lead"] = get_lead(session_id)
    return session


def stats() -> dict[str, Any]:
    with _connect() as connection:
        sessions = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        leads = connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    return {"sessions": sessions, "messages": messages, "leads": leads}
