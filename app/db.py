"""SQLite storage: FAQs, documents, embedded chunks and interaction logs."""
import json
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS faqs (
    id       INTEGER PRIMARY KEY,
    intent   TEXT UNIQUE NOT NULL,
    patterns TEXT NOT NULL,          -- JSON list of example phrasings
    answer   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INTEGER,
    section     TEXT,                 -- nearest heading, used in citations
    position    INTEGER,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL         -- float32 vector, L2-normalised
);
CREATE TABLE IF NOT EXISTS logs (
    id           INTEGER PRIMARY KEY,
    session_id   TEXT,
    user_message TEXT,
    bot_reply    TEXT,
    route        TEXT,                -- faq | rag | passage | clarify | offtopic | fallback
    intent       TEXT,
    confidence   REAL,
    sources      TEXT,                -- JSON list of citations
    resolved     INTEGER DEFAULT 0,   -- 1 once an admin has taught the bot this answer
    escalated    INTEGER DEFAULT 0,   -- 1 if the customer seemed frustrated and was offered a human
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tickets (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT,
    question    TEXT NOT NULL,
    email       TEXT NOT NULL,
    status      TEXT DEFAULT 'open',  -- open | closed
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # upgrade databases created by older versions of the app
        columns = {r[1] for r in conn.execute("PRAGMA table_info(logs)")}
        if "escalated" not in columns:
            conn.execute("ALTER TABLE logs ADD COLUMN escalated INTEGER DEFAULT 0")
        columns = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
        if "checksum" not in columns:  # lets start-up notice edited help-centre files
            conn.execute("ALTER TABLE documents ADD COLUMN checksum TEXT")


def upsert_faq(intent: str, patterns: list[str], answer: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO faqs (intent, patterns, answer) VALUES (?, ?, ?) "
            "ON CONFLICT(intent) DO UPDATE SET patterns = excluded.patterns, answer = excluded.answer",
            (intent, json.dumps(patterns), answer),
        )


def list_faqs() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT intent, patterns, answer FROM faqs ORDER BY intent").fetchall()
    return [{"intent": r["intent"], "patterns": json.loads(r["patterns"]), "answer": r["answer"]} for r in rows]


def log_interaction(session_id, user_message, bot_reply, route, intent, confidence, sources, escalated=False):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO logs (session_id, user_message, bot_reply, route, intent, confidence, sources, escalated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, user_message, bot_reply, route, intent, confidence, json.dumps(sources), int(escalated)),
        )


def recent_logs(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "sources": json.loads(r["sources"] or "[]")} for r in rows]


GAP_ROUTES = ("fallback", "passage")  # not answered / only partly answered


def unresolved_questions() -> list[tuple[str, str]]:
    """(question, route) for every question the bot couldn't fully answer and nobody has fixed yet."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT user_message, route FROM logs WHERE route IN (?, ?) AND resolved = 0", GAP_ROUTES
        ).fetchall()
    return [(r["user_message"], r["route"]) for r in rows]


def mark_resolved(questions: list[str]):
    with get_conn() as conn:
        conn.executemany(
            "UPDATE logs SET resolved = 1 WHERE route IN (?, ?) AND lower(user_message) = lower(?)",
            [(*GAP_ROUTES, q) for q in questions],
        )


def create_ticket(session_id: str | None, question: str, email: str) -> int:
    with get_conn() as conn:
        return conn.execute(
            "INSERT INTO tickets (session_id, question, email) VALUES (?, ?, ?)",
            (session_id, question, email),
        ).lastrowid


def list_tickets() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM tickets ORDER BY status = 'closed', id DESC").fetchall()
    return [dict(r) for r in rows]


def close_ticket(ticket_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE tickets SET status = 'closed' WHERE id = ?", (ticket_id,))


def stats() -> dict:
    with get_conn() as conn:
        by_route = conn.execute("SELECT route, COUNT(*) AS n FROM logs GROUP BY route").fetchall()
        unanswered = conn.execute(
            "SELECT user_message, COUNT(*) AS n FROM logs WHERE route = 'fallback' AND resolved = 0 "
            "GROUP BY lower(user_message) ORDER BY n DESC LIMIT 10"
        ).fetchall()
        open_tickets = conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'open'").fetchone()[0]
        escalations = conn.execute("SELECT COUNT(*) FROM logs WHERE escalated = 1").fetchone()[0]
    return {
        "by_route": {r["route"]: r["n"] for r in by_route},
        "top_unanswered": [dict(r) for r in unanswered],
        "open_tickets": open_tickets,
        "escalations": escalations,
    }
