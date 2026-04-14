"""
Jarvis Persistent Memory — Level 1
──────────────────────────────────────────────────────────────────────────────
SQLite-backed memory that survives restarts. Stores:
  • facts          — key/value pairs Jarvis "knows" about Nicholas
  • conversations  — full turn-by-turn history across sessions

Uses only Python's built-in sqlite3 — no new dependencies.
"""

import sqlite3
import datetime
from pathlib import Path

DB_PATH = Path("/Users/nicholascoppola/Documents/Coding_Projects/Jarvis/jarvis_memory.db")

_REMEMBER_TRIGGERS = (
    "remember that",
    "don't forget",
    "do not forget",
    "make a note that",
    "keep in mind that",
)


class MemoryManager:
    def __init__(self) -> None:
        try:
            self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    value TEXT,
                    updated_at TIMESTAMP
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME
                )
                """
            )
            self._conn.commit()
        except Exception as e:
            print(f"[Memory] Init error: {e}")

    # ── Facts ────────────────────────────────────────────────────────────────
    def seed_initial_facts(self) -> None:
        try:
            cur = self._conn.execute("SELECT COUNT(*) FROM facts")
            (count,) = cur.fetchone()
            if count == 0:
                self.save_fact("user_name", "Nicholas")
                self.save_fact(
                    "user_preference_mornings",
                    "Nicholas wakes up early and values morning productivity",
                )
                print("[Memory] Initial facts seeded.")
        except Exception as e:
            print(f"[Memory] seed_initial_facts error: {e}")

    def save_fact(self, key: str, value: str) -> None:
        try:
            now = datetime.datetime.utcnow().isoformat()
            self._conn.execute(
                "INSERT OR REPLACE INTO facts (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now),
            )
            self._conn.commit()
        except Exception as e:
            print(f"[Memory] save_fact error: {e}")

    def get_all_facts(self) -> list:
        try:
            cur = self._conn.execute("SELECT key, value FROM facts ORDER BY id ASC")
            return cur.fetchall()
        except Exception as e:
            print(f"[Memory] get_all_facts error: {e}")
            return []

    def format_facts_for_prompt(self) -> str:
        try:
            facts = self.get_all_facts()
            if not facts:
                return ""
            parts = [f"{k}={v}" for k, v in facts]
            return "What I know about you: " + " | ".join(parts)
        except Exception as e:
            print(f"[Memory] format_facts_for_prompt error: {e}")
            return ""

    # ── Conversations ────────────────────────────────────────────────────────
    def save_exchange(self, user_message: str, assistant_reply: str) -> None:
        try:
            now = datetime.datetime.utcnow().isoformat()
            self._conn.execute(
                "INSERT INTO conversations (role, content, timestamp) VALUES (?, ?, ?)",
                ("user", user_message, now),
            )
            self._conn.execute(
                "INSERT INTO conversations (role, content, timestamp) VALUES (?, ?, ?)",
                ("assistant", assistant_reply, now),
            )
            self._conn.commit()
        except Exception as e:
            print(f"[Memory] save_exchange error: {e}")

    def get_recent_exchanges(self, n: int = 10) -> list:
        try:
            cur = self._conn.execute(
                "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
                (n * 2,),
            )
            rows = cur.fetchall()
            rows.reverse()
            return [{"role": r, "content": c} for r, c in rows]
        except Exception as e:
            print(f"[Memory] get_recent_exchanges error: {e}")
            return []

    # ── Command detection ────────────────────────────────────────────────────
    def detect_remember_command(self, text: str):
        try:
            if not text:
                return None
            lowered = text.lower()
            for trigger in _REMEMBER_TRIGGERS:
                idx = lowered.find(trigger)
                if idx != -1:
                    extracted = text[idx + len(trigger):].strip()
                    extracted = extracted.lstrip(",:. ").strip()
                    if extracted:
                        return extracted
            return None
        except Exception as e:
            print(f"[Memory] detect_remember_command error: {e}")
            return None
