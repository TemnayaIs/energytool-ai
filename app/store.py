"""Хранилище диалогов и заявок на SQLite (stdlib, без ORM)."""

from __future__ import annotations

import json
import sqlite3
import uuid
import hashlib
import secrets
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
                title TEXT NOT NULL DEFAULT 'Новая сессия',
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
                status TEXT NOT NULL DEFAULT 'open',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            CREATE INDEX IF NOT EXISTS idx_leads_session ON leads(session_id);

            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                name TEXT NOT NULL DEFAULT '',
                phone TEXT NOT NULL DEFAULT '',
                company TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )
        # Миграция для старых БД: добавляем колонку status, если её нет
        try:
            connection.execute("ALTER TABLE leads ADD COLUMN status TEXT NOT NULL DEFAULT 'open'")
        except Exception:
            pass


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


def clear_all_sessions() -> int:
    """Удаляет все сессии, сообщения и заявки. Возвращает количество удалённых сессий."""
    with _connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        connection.execute("DELETE FROM leads")
        connection.execute("DELETE FROM messages")
        connection.execute("DELETE FROM sessions")
    return count


def add_lead(session_id: str, name: str, phone: str, company: str, summary: str) -> int:
    with _connect() as connection:
        cursor = connection.execute(
            "INSERT INTO leads (session_id, name, phone, company, summary, created_at, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, name, phone, company, summary, _now(), "open"),
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


def _hash_password(password: str, salt: str = "") -> str:
    if not salt:
        salt = secrets.token_hex(16)
    return salt + ":" + hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _verify_password(password: str, stored: str) -> bool:
    if ":" not in stored:
        return False
    salt, hashed = stored.split(":", 1)
    return _hash_password(password, salt) == stored


def create_user(username: str, password: str, role: str = "user", name: str = "", phone: str = "", company: str = "") -> dict:
    username = username.strip().lower()
    if not username or not password:
        raise ValueError("username and password required")
    user_id = uuid.uuid4().hex
    now = _now()
    with _connect() as connection:
        try:
            connection.execute(
                "INSERT INTO users (id, username, password_hash, role, name, phone, company, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, username, _hash_password(password), role, name, phone, company, now),
            )
        except sqlite3.IntegrityError:
            raise ValueError("username already exists")
    return {"id": user_id, "username": username, "role": role, "name": name, "phone": phone, "company": company, "created_at": now}


def verify_password(password, stored):
    return _verify_password(password, stored)


def get_user_by_username(username: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
    return dict(row) if row else None


def get_user(user_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute("SELECT id, username, role, name, phone, company, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def update_user(user_id: str, name: str = None, phone: str = None, company: str = None) -> dict | None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if phone is not None:
        fields.append("phone = ?")
        values.append(phone)
    if company is not None:
        fields.append("company = ?")
        values.append(company)
    if not fields:
        return get_user(user_id)
    values.append(user_id)
    with _connect() as connection:
        connection.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", values)
    return get_user(user_id)


def create_token(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with _connect() as connection:
        connection.execute("INSERT INTO tokens (token, user_id, created_at) VALUES (?, ?, ?)", (token, user_id, _now()))
    return token


def get_user_by_token(token: str) -> dict | None:
    if not token:
        return None
    with _connect() as connection:
        row = connection.execute(
            "SELECT u.id, u.username, u.role, u.name, u.phone, u.company, u.created_at FROM users u JOIN tokens t ON t.user_id = u.id WHERE t.token = ?",
            (token,),
        ).fetchone()
    return dict(row) if row else None


def revoke_token(token: str) -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM tokens WHERE token = ?", (token,))


def list_all_leads(limit: int = 100) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM leads ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def list_all_sessions(limit: int = 50) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at, s.state, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count, "
            "(SELECT COUNT(*) FROM leads l WHERE l.session_id = s.id) AS has_lead "
            "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_lead_status(lead_id: int, status: str) -> bool:
    if status not in ("open", "closed"):
        return False
    with _connect() as connection:
        cur = connection.execute("UPDATE leads SET status = ? WHERE id = ?", (status, lead_id))
        return cur.rowcount > 0


def stats() -> dict[str, Any]:
    with _connect() as connection:
        sessions = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        leads = connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    return {"sessions": sessions, "messages": messages, "leads": leads}
