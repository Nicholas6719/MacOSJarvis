"""
Jarvis Persistent Memory — Level 1
──────────────────────────────────────────────────────────────────────────────
SQLite-backed memory that survives restarts. Stores:
  • facts          — key/value pairs Jarvis "knows" about Nicholas
  • conversations  — full turn-by-turn history across sessions

Uses only Python's built-in sqlite3 — no new dependencies.
"""

import calendar
import sqlite3
import datetime
import json as _json
import re
import time
from pathlib import Path

DB_PATH = Path("/Users/nicholascoppola/Documents/Coding_Projects/Jarvis/jarvis_memory.db")

_REMEMBER_TRIGGERS = (
    "remember that",
    "don't forget",
    "do not forget",
    "make a note that",
    "keep in mind that",
)

_SEARCH_STOP_WORDS = frozenset({
    "i", "me", "my", "the", "a", "an", "is", "was", "were", "did", "do",
    "you", "about", "when", "what", "that", "it", "in", "on", "at", "to",
    "of", "and", "or", "for", "with", "this", "we",
})

_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)

_FORGET_TRIGGERS = (
    "forget that",
    "don't remember",
    "do not remember",
    "remove the fact that",
    "delete the fact that",
    "you don't need to know",
    "you do not need to know",
    "forget",
)

_VERB_3P = {
    "like": "likes", "love": "loves", "hate": "hates",
    "prefer": "prefers", "enjoy": "enjoys", "live": "lives",
    "work": "works", "go": "goes", "grew": "grew", "grow": "grows",
    "am": "is", "drink": "drinks", "eat": "eats", "wake": "wakes",
    "sleep": "sleeps", "run": "runs", "walk": "walks", "drive": "drives",
    "read": "reads", "play": "plays", "watch": "watches", "listen": "listens",
    "use": "uses", "have": "has", "take": "takes", "make": "makes",
    "do": "does", "start": "starts", "finish": "finishes", "study": "studies",
    "cook": "cooks", "clean": "cleans", "write": "writes", "code": "codes",
    "exercise": "exercises", "meditate": "meditates", "travel": "travels",
}


def _to_third_person(verb: str) -> str:
    v = verb.lower()
    if v in _VERB_3P:
        return _VERB_3P[v]
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    if v.endswith(("s", "sh", "ch", "x", "z", "o")):
        return v + "es"
    return v + "s"


def _strip_tail(s: str) -> str:
    return s.strip().rstrip(".!?,;:").strip()


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
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_text TEXT,
                    date_from DATETIME,
                    date_to DATETIME,
                    conversation_ids TEXT,
                    created_at DATETIME
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_text TEXT,
                    covers_from DATETIME,
                    covers_to DATETIME,
                    source_summary_ids TEXT,
                    created_at DATETIME
                )
                """
            )
            # Level 4: FTS5 search indexes — virtual tables, not data stores.
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                USING fts5(content, role, content='conversations', content_rowid='id')
                """
            )
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts
                USING fts5(summary_text, content='summaries', content_rowid='id')
                """
            )
            self._conn.commit()
            # Always keep indexes current on startup.
            self.rebuild_fts_indexes()
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

    # ── Level 2: Dedup, auto-detect, forget, readback ────────────────────────
    def facts_are_similar(self, new_value: str, existing_values: list) -> bool:
        try:
            if not new_value:
                return False
            new_s = new_value.strip().lower()
            if not new_s:
                return False
            for existing in existing_values:
                if not existing:
                    continue
                ex_s = existing.strip().lower()
                if not ex_s:
                    continue
                if new_s == ex_s:
                    return True
                if new_s in ex_s or ex_s in new_s:
                    return True
                new_words = set(w for w in new_s.split() if len(w) > 2)
                ex_words = set(w for w in ex_s.split() if len(w) > 2)
                if not new_words or not ex_words:
                    continue
                overlap = new_words & ex_words
                ratio = len(overlap) / max(len(new_words), len(ex_words))
                if ratio >= 0.6:
                    return True
            return False
        except Exception as e:
            print(f"[Memory] facts_are_similar error: {e}")
            return False

    def auto_detect_fact(self, text: str):
        try:
            if not text:
                return None
            t = text.strip()
            if not t or t.endswith("?"):
                return None
            lowered = t.lower()

            # Reject hypotheticals, commands, vague statements, anything about Jarvis
            rejects = (
                "i would like", "i'd like",
                "i want you to", "i need you to",
                "i think", "i feel like", "i'm not sure", "i am not sure",
                "i guess", "i suppose", "i wonder",
                "jarvis", "you are", "you're",
                "can you", "could you", "would you", "please",
            )
            for r in rejects:
                if r in lowered:
                    return None

            ts = str(int(time.time()))

            # PREFERENCES — "I like/love/hate/prefer/enjoy X"
            m = re.search(r"\bi (like|love|hate|prefer|enjoy) (.+)", t, re.IGNORECASE)
            if m:
                verb = m.group(1)
                obj = _strip_tail(m.group(2))
                if obj and obj not in ("you", "it", "that", "this", "them"):
                    return (f"pref_{ts}", f"Nicholas {_to_third_person(verb)} {obj}")

            # PREFERENCES — "I can't stand X"
            m = re.search(r"\bi can'?t stand (.+)", t, re.IGNORECASE)
            if m:
                obj = _strip_tail(m.group(1))
                if obj:
                    return (f"pref_{ts}", f"Nicholas can't stand {obj}")

            # PREFERENCES — "I don't like X"
            m = re.search(r"\bi don'?t like (.+)", t, re.IGNORECASE)
            if m:
                obj = _strip_tail(m.group(1))
                if obj and obj not in ("you", "it", "that", "this"):
                    return (f"pref_{ts}", f"Nicholas doesn't like {obj}")

            # PREFERENCES — "my favorite X is Y"
            m = re.search(r"\bmy favorite (\w+(?:\s+\w+)?) is (.+)", t, re.IGNORECASE)
            if m:
                cat = m.group(1).strip()
                val = _strip_tail(m.group(2))
                if val:
                    return (f"pref_{ts}", f"Nicholas's favorite {cat} is {val}")

            # ROUTINES — "every [morning/day/monday...] I X"
            m = re.search(
                r"\bevery (morning|evening|night|afternoon|day|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekend)s? i (\w+)(.*)",
                t, re.IGNORECASE,
            )
            if m:
                when = m.group(1)
                verb = m.group(2)
                rest = _strip_tail(m.group(3))
                phrase = f"Every {when} Nicholas {_to_third_person(verb)}"
                if rest:
                    phrase += " " + rest
                return (f"routine_{ts}", phrase.strip())

            # ROUTINES — "I work from home [day]s"
            m = re.search(
                r"\bi work from home (mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?|on \w+)",
                t, re.IGNORECASE,
            )
            if m:
                day = _strip_tail(m.group(1))
                return (f"routine_{ts}", f"Nicholas works from home {day}")

            # ROUTINES — "I always/usually/never X"
            m = re.search(r"\bi (always|usually|never|rarely|often) (\w+)(.*)", t, re.IGNORECASE)
            if m:
                freq = m.group(1)
                verb = m.group(2)
                rest = _strip_tail(m.group(3))
                phrase = f"Nicholas {freq} {_to_third_person(verb)}"
                if rest:
                    phrase += " " + rest
                return (f"routine_{ts}", phrase.strip())

            # NAMES OF THINGS — "my [category] is/is named/named X"
            categories = (
                "dog|cat|pet|wife|husband|partner|girlfriend|boyfriend|"
                "son|daughter|kid|child|boss|manager|car|truck"
            )
            m = re.search(
                rf"\bmy ({categories})(?:'s name)? (?:is named|is called|is|named|'s) (.+)",
                t, re.IGNORECASE,
            )
            if m:
                cat = m.group(1)
                name = _strip_tail(m.group(2))
                if name and name not in ("a", "an", "the"):
                    return (f"personal_{cat}_{ts}", f"Nicholas's {cat} is named {name}")

            # STATED FACTS — "I live in X"
            m = re.search(r"\bi live in (.+)", t, re.IGNORECASE)
            if m:
                place = _strip_tail(m.group(1))
                if place:
                    return (f"fact_{ts}", f"Nicholas lives in {place}")

            # STATED FACTS — "I work at/in/from X"
            m = re.search(r"\bi work (at|in|from) (.+)", t, re.IGNORECASE)
            if m:
                prep = m.group(1)
                place = _strip_tail(m.group(2))
                if place and place != "home":
                    return (f"fact_{ts}", f"Nicholas works {prep} {place}")

            # STATED FACTS — "I go to X"
            m = re.search(r"\bi go to (\w+(?:\s+\w+){0,4})", t, re.IGNORECASE)
            if m:
                place = _strip_tail(m.group(1))
                if place and place not in ("bed", "sleep", "the store", "the gym"):
                    return (f"fact_{ts}", f"Nicholas goes to {place}")

            # STATED FACTS — "I'm from X"
            m = re.search(r"\bi(?:'m| am) from (.+)", t, re.IGNORECASE)
            if m:
                place = _strip_tail(m.group(1))
                if place:
                    return (f"fact_{ts}", f"Nicholas is from {place}")

            # STATED FACTS — "I grew up in X"
            m = re.search(r"\bi grew up in (.+)", t, re.IGNORECASE)
            if m:
                place = _strip_tail(m.group(1))
                if place:
                    return (f"fact_{ts}", f"Nicholas grew up in {place}")

            # STATED FACTS — "I'm a X" / "I am a X" (occupation)
            m = re.search(r"\bi(?:'m| am) an? ([a-z][a-z ]{2,30})", t, re.IGNORECASE)
            if m:
                role = _strip_tail(m.group(1))
                bad = (
                    "bit", "little", "lot", "bunch", "mess", "tired", "hungry",
                    "happy", "sad", "busy", "sick", "bored", "ready", "fan of",
                    "big fan", "huge fan",
                )
                first = role.split()[0] if role else ""
                if role and first not in bad and not role.startswith("bit "):
                    # Keep only first 4 words to stay role-ish
                    words = role.split()[:4]
                    role_clean = " ".join(words)
                    return (f"fact_{ts}", f"Nicholas is a {role_clean}")

            return None
        except Exception as e:
            print(f"[Memory] auto_detect_fact error: {e}")
            return None

    def detect_forget_command(self, text: str):
        try:
            if not text:
                return None
            lowered = text.lower()
            for trigger in _FORGET_TRIGGERS:
                idx = lowered.find(trigger)
                if idx != -1:
                    extracted = text[idx + len(trigger):].strip()
                    extracted = extracted.lstrip(",:. ").rstrip(".!?,;:").strip()
                    # Strip leading "that" if present
                    if extracted.lower().startswith("that "):
                        extracted = extracted[5:].strip()
                    if extracted:
                        return extracted
            return None
        except Exception as e:
            print(f"[Memory] detect_forget_command error: {e}")
            return None

    def delete_matching_facts(self, search_term: str) -> int:
        try:
            if not search_term:
                return 0
            like = f"%{search_term.lower()}%"
            cur = self._conn.execute(
                "SELECT id FROM facts WHERE LOWER(value) LIKE ?", (like,)
            )
            ids = [row[0] for row in cur.fetchall()]
            if not ids:
                return 0
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM facts WHERE id IN ({placeholders})", ids
            )
            self._conn.commit()
            return len(ids)
        except Exception as e:
            print(f"[Memory] delete_matching_facts error: {e}")
            return 0

    def get_facts_for_readback(self) -> list:
        try:
            cur = self._conn.execute(
                "SELECT key, value FROM facts WHERE key NOT LIKE 'user_name%' ORDER BY id ASC"
            )
            return [v for _, v in cur.fetchall()]
        except Exception as e:
            print(f"[Memory] get_facts_for_readback error: {e}")
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

    # ── Level 3: Summarization ───────────────────────────────────────────────
    def get_unsummarized_exchanges(self) -> list:
        """Return conversation rows not yet covered by any summary and outside
        the active 40-row (20-exchange) window."""
        try:
            # Collect every conversation_id already covered by a summary.
            cur = self._conn.execute("SELECT conversation_ids FROM summaries")
            covered = set()
            for (ids_json,) in cur.fetchall():
                if not ids_json:
                    continue
                try:
                    ids = _json.loads(ids_json)
                    for i in ids:
                        covered.add(int(i))
                except Exception:
                    continue

            # Determine the cutoff id: everything newer than this is part of
            # the active window and should be skipped.
            cur = self._conn.execute(
                "SELECT id FROM conversations ORDER BY id DESC LIMIT 40"
            )
            active_ids = {row[0] for row in cur.fetchall()}

            # Fetch every conversation row and filter in Python — simple and
            # correct even when covered/active sets overlap strangely.
            cur = self._conn.execute(
                "SELECT id, role, content, timestamp FROM conversations ORDER BY id ASC"
            )
            out = []
            for cid, role, content, ts in cur.fetchall():
                if cid in covered:
                    continue
                if cid in active_ids:
                    continue
                out.append(
                    {"id": cid, "role": role, "content": content, "timestamp": ts}
                )
            return out
        except Exception as e:
            print(f"[Memory] get_unsummarized_exchanges error: {e}")
            return []

    def save_summary(
        self,
        summary_text: str,
        conversation_ids: list,
        date_from: str,
        date_to: str,
    ) -> None:
        try:
            ids_json = _json.dumps(list(conversation_ids))
            now = datetime.datetime.utcnow().isoformat()
            self._conn.execute(
                "INSERT INTO summaries (summary_text, date_from, date_to, conversation_ids, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (summary_text, date_from, date_to, ids_json, now),
            )
            self._conn.commit()
        except Exception as e:
            print(f"[Memory] save_summary error: {e}")

    def get_recent_summaries(self, days: int = 30) -> list:
        try:
            cutoff = (
                datetime.datetime.utcnow() - datetime.timedelta(days=days)
            ).isoformat()
            cur = self._conn.execute(
                "SELECT summary_text FROM summaries WHERE created_at >= ? "
                "ORDER BY created_at ASC",
                (cutoff,),
            )
            return [row[0] for row in cur.fetchall() if row[0]]
        except Exception as e:
            print(f"[Memory] get_recent_summaries error: {e}")
            return []

    def format_summaries_for_prompt(self) -> str:
        try:
            summaries = self.get_recent_summaries()
            if not summaries:
                return ""
            bullets = "\n".join(f"- {s}" for s in summaries)
            return f"What I recall from older conversations:\n{bullets}"
        except Exception as e:
            print(f"[Memory] format_summaries_for_prompt error: {e}")
            return ""

    def batch_conversations_for_summary(
        self, conversations: list, batch_size: int = 10
    ) -> list:
        try:
            if not conversations:
                return []
            return [
                conversations[i : i + batch_size]
                for i in range(0, len(conversations), batch_size)
            ]
        except Exception as e:
            print(f"[Memory] batch_conversations_for_summary error: {e}")
            return []

    # ── Level 3+: Cleanup of covered raw conversations ──────────────────────
    def cleanup_summarized_exchanges(self) -> int:
        """Delete conversation rows that are already covered by a summary AND
        outside the most-recent 40-row active window. Never touches facts or
        summaries tables. Never deletes anything not covered. Never deletes
        anything inside the active window. Returns count of deleted rows."""
        try:
            # Build the set of IDs covered by any existing summary.
            cur = self._conn.execute("SELECT conversation_ids FROM summaries")
            covered = set()
            for (ids_json,) in cur.fetchall():
                if not ids_json:
                    continue
                try:
                    ids = _json.loads(ids_json)
                    for i in ids:
                        covered.add(int(i))
                except Exception:
                    continue

            if not covered:
                return 0

            # Lock the most recent 40 rows — the active window is untouchable.
            cur = self._conn.execute(
                "SELECT id FROM conversations ORDER BY id DESC LIMIT 40"
            )
            active_ids = {row[0] for row in cur.fetchall()}

            # Deletable = covered AND NOT active.
            deletable = [cid for cid in covered if cid not in active_ids]
            if not deletable:
                return 0

            # Delete in chunks to keep the SQL parameter list sane.
            deleted_total = 0
            chunk = 500
            for i in range(0, len(deletable), chunk):
                piece = deletable[i : i + chunk]
                placeholders = ",".join("?" * len(piece))
                cur = self._conn.execute(
                    f"DELETE FROM conversations WHERE id IN ({placeholders})",
                    piece,
                )
                deleted_total += cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(piece)
            self._conn.commit()

            if deleted_total > 0:
                print(f"[Memory] Cleaned up {deleted_total} covered conversation rows.")
            return deleted_total
        except Exception as e:
            print(f"[Memory] cleanup_summarized_exchanges error: {e}")
            return 0

    # ── Level 3+: Meta-summarization of old summaries ───────────────────────
    def get_summaries_needing_meta(self, days: int = 30) -> list:
        """Return summary rows older than `days` days whose id is NOT already
        covered by any existing meta_summary's source_summary_ids."""
        try:
            # Collect already-covered summary IDs from existing meta summaries.
            cur = self._conn.execute("SELECT source_summary_ids FROM meta_summaries")
            covered = set()
            for (ids_json,) in cur.fetchall():
                if not ids_json:
                    continue
                try:
                    for i in _json.loads(ids_json):
                        covered.add(int(i))
                except Exception:
                    continue

            cutoff = (
                datetime.datetime.utcnow() - datetime.timedelta(days=days)
            ).isoformat()
            cur = self._conn.execute(
                "SELECT id, summary_text, date_from, date_to "
                "FROM summaries WHERE created_at < ? ORDER BY created_at ASC",
                (cutoff,),
            )
            out = []
            for sid, text, dfrom, dto in cur.fetchall():
                if sid in covered:
                    continue
                out.append(
                    {
                        "id": sid,
                        "summary_text": text or "",
                        "date_from": dfrom or "",
                        "date_to": dto or "",
                    }
                )
            return out
        except Exception as e:
            print(f"[Memory] get_summaries_needing_meta error: {e}")
            return []

    def save_meta_summary(
        self,
        summary_text: str,
        source_ids: list,
        covers_from: str,
        covers_to: str,
    ) -> None:
        try:
            ids_json = _json.dumps(list(source_ids))
            now = datetime.datetime.utcnow().isoformat()
            self._conn.execute(
                "INSERT INTO meta_summaries "
                "(summary_text, covers_from, covers_to, source_summary_ids, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (summary_text, covers_from, covers_to, ids_json, now),
            )
            self._conn.commit()
        except Exception as e:
            print(f"[Memory] save_meta_summary error: {e}")

    def get_latest_meta_summary(self):
        try:
            cur = self._conn.execute(
                "SELECT summary_text FROM meta_summaries "
                "ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
            return None
        except Exception as e:
            print(f"[Memory] get_latest_meta_summary error: {e}")
            return None

    def format_meta_summary_for_prompt(self) -> str:
        try:
            latest = self.get_latest_meta_summary()
            if not latest:
                return ""
            return f"Long-range memory: {latest}"
        except Exception as e:
            print(f"[Memory] format_meta_summary_for_prompt error: {e}")
            return ""

    # ── Level 4: Searchable memory (FTS5) ───────────────────────────────────
    def rebuild_fts_indexes(self) -> None:
        """Sync the FTS5 virtual tables with their source tables. Safe to
        call repeatedly — FTS5's 'rebuild' command is idempotent."""
        try:
            self._conn.execute(
                "INSERT INTO conversations_fts(conversations_fts) VALUES('rebuild')"
            )
            self._conn.execute(
                "INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')"
            )
            self._conn.commit()
        except Exception as e:
            print(f"[Memory] rebuild_fts_indexes error: {e}")

    def _clean_search_query(self, query: str) -> str:
        """Strip punctuation, lowercase, drop stop words. Returns the cleaned
        term with words joined by spaces, ready for an FTS5 MATCH clause."""
        if not query:
            return ""
        # Keep only alphanumerics and spaces; this also escapes FTS5 operators.
        cleaned = re.sub(r"[^a-z0-9\s]", " ", query.lower())
        words = [w for w in cleaned.split() if w and w not in _SEARCH_STOP_WORDS]
        return " ".join(words)

    def search_memory(
        self,
        query: str = "",
        max_results: int = 5,
        date_range=None,
    ) -> list:
        """Full-text search across conversations and summaries, with optional
        timestamp-range filtering (Level 4+).

        query      — free text (cleaned to FTS5 MATCH terms). May be empty
                     when the caller is doing a pure date-range lookup.
        date_range — optional (start_iso, end_iso) tuple. When present, both
                     FTS queries are restricted to rows inside the window; if
                     `query` is empty, a plain SELECT against the window is
                     used and results are ordered by timestamp ASC.

        Returns a combined list of result dicts (max `max_results` total).
        Returns [] on any error."""
        try:
            cleaned = self._clean_search_query(query) if query else ""
            has_terms = bool(cleaned) and len(cleaned) >= 3
            has_range = (
                date_range is not None
                and isinstance(date_range, tuple)
                and len(date_range) == 2
                and date_range[0]
                and date_range[1]
            )

            # Nothing to search on — bail early.
            if not has_terms and not has_range:
                return []

            results: list = []

            if has_terms:
                # FTS5 MATCH + optional date-range filter.
                try:
                    if has_range:
                        cur = self._conn.execute(
                            "SELECT c.id, c.role, c.content, c.timestamp "
                            "FROM conversations_fts f "
                            "JOIN conversations c ON c.id = f.rowid "
                            "WHERE conversations_fts MATCH ? "
                            "AND c.timestamp BETWEEN ? AND ? "
                            "ORDER BY rank LIMIT ?",
                            (cleaned, date_range[0], date_range[1], max_results),
                        )
                    else:
                        cur = self._conn.execute(
                            "SELECT c.id, c.role, c.content, c.timestamp "
                            "FROM conversations_fts f "
                            "JOIN conversations c ON c.id = f.rowid "
                            "WHERE conversations_fts MATCH ? "
                            "ORDER BY rank LIMIT ?",
                            (cleaned, max_results),
                        )
                    for cid, role, content, ts in cur.fetchall():
                        results.append(
                            {
                                "source": "conversation",
                                "id": cid,
                                "role": role,
                                "content": content or "",
                                "timestamp": ts or "",
                            }
                        )
                except Exception as e:
                    print(f"[Memory] conversations_fts MATCH error: {e}")

                try:
                    if has_range:
                        # Summaries "overlap" the window when the summary's
                        # date_from falls inside it. Cheap overlap check.
                        cur = self._conn.execute(
                            "SELECT s.id, s.summary_text, s.date_from, s.date_to "
                            "FROM summaries_fts f "
                            "JOIN summaries s ON s.id = f.rowid "
                            "WHERE summaries_fts MATCH ? "
                            "AND s.date_from BETWEEN ? AND ? "
                            "ORDER BY rank LIMIT ?",
                            (cleaned, date_range[0], date_range[1], max_results),
                        )
                    else:
                        cur = self._conn.execute(
                            "SELECT s.id, s.summary_text, s.date_from, s.date_to "
                            "FROM summaries_fts f "
                            "JOIN summaries s ON s.id = f.rowid "
                            "WHERE summaries_fts MATCH ? "
                            "ORDER BY rank LIMIT ?",
                            (cleaned, max_results),
                        )
                    for sid, text, dfrom, dto in cur.fetchall():
                        results.append(
                            {
                                "source": "summary",
                                "id": sid,
                                "summary_text": text or "",
                                "date_from": dfrom or "",
                                "date_to": dto or "",
                            }
                        )
                except Exception as e:
                    print(f"[Memory] summaries_fts MATCH error: {e}")

            else:
                # Date-only lookup — no FTS, just a plain window slice.
                try:
                    cur = self._conn.execute(
                        "SELECT id, role, content, timestamp "
                        "FROM conversations "
                        "WHERE timestamp BETWEEN ? AND ? "
                        "ORDER BY timestamp ASC LIMIT ?",
                        (date_range[0], date_range[1], max_results),
                    )
                    for cid, role, content, ts in cur.fetchall():
                        results.append(
                            {
                                "source": "conversation",
                                "id": cid,
                                "role": role,
                                "content": content or "",
                                "timestamp": ts or "",
                            }
                        )
                except Exception as e:
                    print(f"[Memory] conversations range lookup error: {e}")

                try:
                    cur = self._conn.execute(
                        "SELECT id, summary_text, date_from, date_to "
                        "FROM summaries "
                        "WHERE date_from BETWEEN ? AND ? "
                        "ORDER BY date_from ASC LIMIT ?",
                        (date_range[0], date_range[1], max_results),
                    )
                    for sid, text, dfrom, dto in cur.fetchall():
                        results.append(
                            {
                                "source": "summary",
                                "id": sid,
                                "summary_text": text or "",
                                "date_from": dfrom or "",
                                "date_to": dto or "",
                            }
                        )
                except Exception as e:
                    print(f"[Memory] summaries range lookup error: {e}")

            # Cap total at max_results, preferring summaries first (they're
            # higher signal per row), then falling back to conversation hits.
            summaries_first = [r for r in results if r["source"] == "summary"]
            convs = [r for r in results if r["source"] == "conversation"]
            combined = (summaries_first + convs)[:max_results]
            return combined
        except Exception as e:
            print(f"[Memory] search_memory error: {e}")
            return []

    def _parse_date_range_from_text(self, text: str):
        """Extract a (start_iso, end_iso) UTC window from a user utterance if
        the text references a month (optionally with year/day) OR a supported
        relative phrase like 'yesterday', 'last week', 'this morning'.

        Returns a tuple (start, end) or None on failure.

        Absolute shapes:
          - 'April 2026'            → whole month of April 2026
          - 'April'                 → whole month of April in the current year
          - 'April 14'              → single day, current year
          - 'April 14th 2026'       → single day
          - 'on April 14th'         → single day

        Relative shapes (all evaluated against datetime.utcnow()):
          - 'this morning'          → today 00:00–11:59:59 UTC
          - 'this afternoon'        → today 12:00–17:59:59 UTC
          - 'this evening' / 'tonight' → today 18:00–23:59:59 UTC
          - 'today'                 → today 00:00–23:59:59 UTC
          - 'yesterday'             → yesterday 00:00–23:59:59 UTC
          - 'this week'             → Monday 00:00 → today 23:59:59 UTC
          - 'last week'             → previous Mon–Sun 00:00–23:59:59 UTC
          - 'this month'            → month's 1st 00:00 → today 23:59:59 UTC
          - 'last month'            → previous month 1st–last 00:00–23:59:59
          - 'this year'             → Jan 1 00:00 → today 23:59:59 UTC
          - 'last year'             → previous Jan 1 – Dec 31 00:00–23:59:59
          - 'recently' / 'lately'   → 7 days ago 00:00 → today 23:59:59 UTC
        """
        try:
            if not text:
                return None
            lowered = text.lower()
            now = datetime.datetime.utcnow()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)

            def _iso(start: datetime.datetime, end: datetime.datetime):
                return (start.isoformat(), end.isoformat())

            # ── Relative phrases (checked FIRST) ─────────────────────────
            # Order matters: longer / more specific phrases before shorter
            # ones so e.g. "this morning" wins over "this" fragments.
            if "this morning" in lowered:
                return _iso(today, today.replace(hour=11, minute=59, second=59))
            if "this afternoon" in lowered:
                return _iso(
                    today.replace(hour=12),
                    today.replace(hour=17, minute=59, second=59),
                )
            if "this evening" in lowered or "tonight" in lowered:
                return _iso(
                    today.replace(hour=18),
                    today.replace(hour=23, minute=59, second=59),
                )
            if "yesterday" in lowered:
                y = today - datetime.timedelta(days=1)
                return _iso(y, y.replace(hour=23, minute=59, second=59))
            if "last week" in lowered:
                # Previous calendar week: Mon 00:00 through Sun 23:59:59.
                this_mon = today - datetime.timedelta(days=today.weekday())
                last_mon = this_mon - datetime.timedelta(days=7)
                last_sun = last_mon + datetime.timedelta(days=6)
                return _iso(
                    last_mon, last_sun.replace(hour=23, minute=59, second=59)
                )
            if "this week" in lowered:
                this_mon = today - datetime.timedelta(days=today.weekday())
                end_today = today.replace(hour=23, minute=59, second=59)
                return _iso(this_mon, end_today)
            if "last month" in lowered:
                # First day of previous month → last day of previous month.
                first_this = today.replace(day=1)
                last_prev = first_this - datetime.timedelta(days=1)
                first_prev = last_prev.replace(day=1)
                return _iso(
                    first_prev,
                    last_prev.replace(hour=23, minute=59, second=59),
                )
            if "this month" in lowered:
                first_this = today.replace(day=1)
                end_today = today.replace(hour=23, minute=59, second=59)
                return _iso(first_this, end_today)
            if "last year" in lowered:
                y = today.year - 1
                start = datetime.datetime(y, 1, 1, 0, 0, 0)
                end = datetime.datetime(y, 12, 31, 23, 59, 59)
                return _iso(start, end)
            if "this year" in lowered:
                start = datetime.datetime(today.year, 1, 1, 0, 0, 0)
                end_today = today.replace(hour=23, minute=59, second=59)
                return _iso(start, end_today)
            if "recently" in lowered or "lately" in lowered:
                start = today - datetime.timedelta(days=7)
                end_today = today.replace(hour=23, minute=59, second=59)
                return _iso(start, end_today)
            if "today" in lowered:
                return _iso(today, today.replace(hour=23, minute=59, second=59))

            # ── Absolute month-based parsing (existing logic) ────────────
            month_names = list(_MONTHS)
            month_num = None
            month_match = re.search(
                rf"\b({'|'.join(month_names)})\b", lowered
            )
            if not month_match:
                return None
            month_num = month_names.index(month_match.group(1)) + 1

            # Optional 4-digit year anywhere in the sentence.
            year = None
            year_match = re.search(r"\b(19|20)\d{2}\b", lowered)
            if year_match:
                year = int(year_match.group(0))
            else:
                year = datetime.datetime.utcnow().year

            # Optional day-of-month following the month name, e.g. "april 14",
            # "april 14th", "april 2nd".
            day = None
            day_match = re.search(
                rf"\b{month_match.group(1)}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
                lowered,
            )
            if day_match:
                d = int(day_match.group(1))
                if 1 <= d <= 31:
                    day = d

            # Guard against impossible dates like Feb 30.
            last_day = calendar.monthrange(year, month_num)[1]
            if day is not None and day > last_day:
                day = last_day

            if day is not None:
                start = datetime.datetime(year, month_num, day, 0, 0, 0)
                end = datetime.datetime(year, month_num, day, 23, 59, 59)
            else:
                start = datetime.datetime(year, month_num, 1, 0, 0, 0)
                end = datetime.datetime(
                    year, month_num, last_day, 23, 59, 59
                )

            return (start.isoformat(), end.isoformat())
        except Exception as e:
            print(f"[Memory] _parse_date_range_from_text error: {e}")
            return None

    def detect_memory_search_query(self, text: str):
        """Detect whether a user utterance is a memory-retrieval question.

        Returns a dict with two keys:
            {
                "terms":      "<cleaned FTS keyword string, possibly empty>",
                "date_range": (start_iso, end_iso) or None,
            }
        or None if the utterance is not a memory-search question at all.

        Never overlaps with remember/forget/readback command handlers.
        Date triggers are checked BEFORE phrase triggers so that questions
        like 'in April 2026 what were we talking about' produce a proper
        (keywords, date_range) pair rather than a stale keyword lookup."""
        try:
            if not text:
                return None
            t = text.strip()
            if len(t.split()) < 4:
                return None

            lowered = t.lower()

            # Explicit exclusions — these belong to other handlers.
            if "what do you know about me" in lowered:
                return None
            if "what have you remembered" in lowered:
                return None
            if "what do you remember about me" in lowered:
                return None
            if "tell me what you know about me" in lowered:
                return None
            if "what facts do you have about me" in lowered:
                return None
            for rt in _REMEMBER_TRIGGERS:
                if rt in lowered:
                    return None
            for ft in _FORGET_TRIGGERS:
                if ft in lowered:
                    return None

            months_group = "|".join(_MONTHS)
            matched_trigger = None
            is_date_trigger = False

            # ── Date triggers (checked FIRST) ────────────────────────────
            # "back in <month>" — always a trigger.
            m = re.search(rf"\bback in ({months_group})\b", lowered)
            if m:
                matched_trigger = m.group(0)
                is_date_trigger = True

            # "in <month>" when followed by 4-digit year or "when i"/"that i".
            if matched_trigger is None:
                m = re.search(
                    rf"\bin ({months_group})(?:\s+(\d{{4}}|when i|that i))",
                    lowered,
                )
                if m:
                    matched_trigger = m.group(0)
                    is_date_trigger = True

            # "in <month> <year>" — same but with a raw 4-digit year.
            if matched_trigger is None:
                m = re.search(
                    rf"\bin ({months_group})\s+\d{{4}}\b", lowered
                )
                if m:
                    matched_trigger = m.group(0)
                    is_date_trigger = True

            # "on <month> <day>" e.g. "on April 14th".
            if matched_trigger is None:
                m = re.search(
                    rf"\bon ({months_group})\s+\d+", lowered
                )
                if m:
                    matched_trigger = m.group(0)
                    is_date_trigger = True

            # Relative date phrases (checked after absolute dates so that an
            # explicit "in April 2026" still wins over a stray "recently").
            # Longer phrases listed first so "this morning" beats "today".
            relative_phrases = (
                "this morning",
                "this afternoon",
                "this evening",
                "tonight",
                "this week",
                "last week",
                "this month",
                "last month",
                "this year",
                "last year",
                "yesterday",
                "recently",
                "lately",
                "today",
            )
            if matched_trigger is None:
                for rp in relative_phrases:
                    if rp in lowered:
                        matched_trigger = rp
                        is_date_trigger = True
                        break

            # ── Phrase triggers (checked SECOND) ─────────────────────────
            simple_triggers = (
                "do you remember when",
                "do you remember what",
                "what were we talking about",
                "what was i working on",
                "what did i say about",
                "what did we talk about",
                "what have i told you about",
                "remember when i",
                "do you recall",
            )
            if matched_trigger is None:
                for trig in simple_triggers:
                    if trig in lowered:
                        matched_trigger = trig
                        break

            # "what do i usually X" — require topic word(s) after "usually".
            if matched_trigger is None:
                m = re.search(r"\bwhat do i usually (\w+(?:\s+\w+)+)", lowered)
                if m:
                    matched_trigger = "what do i usually"

            if matched_trigger is None:
                return None

            # Parse a date range from the WHOLE original text — we want to
            # catch dates even when the primary trigger was a phrase trigger
            # like "what were we talking about in April 2026".
            date_range = self._parse_date_range_from_text(t)

            # Strip the primary trigger from the text and clean what remains
            # into FTS terms.
            idx = lowered.find(matched_trigger)
            remainder = (
                lowered[:idx] + lowered[idx + len(matched_trigger):]
            ).strip()
            # Also strip any OTHER simple phrase triggers that slipped into
            # the remainder. Prevents filler verbs like "talking" / "working"
            # from leaking through when a date trigger was primary (e.g.
            # "in April 2026 what were we talking about").
            for trig in simple_triggers:
                if trig in remainder:
                    remainder = remainder.replace(trig, " ")
            # Same treatment for relative phrases — so combined questions like
            # "what were we talking about this morning" don't leave "morning"
            # or "today" in the FTS term list.
            for rp in relative_phrases:
                if rp in remainder:
                    remainder = remainder.replace(rp, " ")
            remainder = remainder.rstrip("?.!,").strip()
            cleaned = self._clean_search_query(remainder)

            # Whenever we successfully parsed a date range, strip month names,
            # years, day numbers, and bare ordinal suffixes from the FTS terms
            # so the keyword lookup doesn't get polluted with date tokens.
            if date_range is not None and cleaned:
                _ordinal_tokens = {"st", "nd", "rd", "th"}
                cleaned_tokens = [
                    w for w in cleaned.split()
                    if w not in _MONTHS
                    and w not in _ordinal_tokens
                    and not re.fullmatch(r"\d{4}", w)
                    and not re.fullmatch(r"\d{1,2}(st|nd|rd|th)?", w)
                ]
                cleaned = " ".join(cleaned_tokens)

            # Require at least ONE useful signal.
            has_terms = bool(cleaned) and len(cleaned) >= 3
            has_range = date_range is not None
            if not has_terms and not has_range:
                return None

            return {
                "terms": cleaned if has_terms else "",
                "date_range": date_range,
            }
        except Exception as e:
            print(f"[Memory] detect_memory_search_query error: {e}")
            return None
