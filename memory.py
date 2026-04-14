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

    def search_memory(self, query: str, max_results: int = 5) -> list:
        """Full-text search across conversations and summaries. Returns a
        combined, ranked list of result dicts (max `max_results` total).
        Returns [] on any error or if the cleaned query is too short."""
        try:
            cleaned = self._clean_search_query(query)
            if not cleaned or len(cleaned) < 3:
                return []

            # FTS5 MATCH accepts space-separated terms (implicit AND/OR per
            # FTS5 grammar). Use them as-is since _clean_search_query already
            # stripped all special characters.
            results = []

            try:
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

            # Cap total at max_results, preferring summaries first (they're
            # higher signal per row), then falling back to conversation hits.
            summaries_first = [r for r in results if r["source"] == "summary"]
            convs = [r for r in results if r["source"] == "conversation"]
            combined = (summaries_first + convs)[:max_results]
            return combined
        except Exception as e:
            print(f"[Memory] search_memory error: {e}")
            return []

    def detect_memory_search_query(self, text: str):
        """Return a cleaned search term if the user's question is clearly
        about retrieving past memory. Return None otherwise. Never overlaps
        with remember/forget/readback command handlers."""
        try:
            if not text:
                return None
            t = text.strip()
            # Must be substantial — at least 4 words total.
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
            # Remember command patterns (save, not search).
            for rt in _REMEMBER_TRIGGERS:
                if rt in lowered:
                    return None
            # Forget command patterns (delete, not search).
            for ft in _FORGET_TRIGGERS:
                if ft in lowered:
                    return None

            # Direct phrase triggers — simple substring tests first.
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
            matched_trigger = None
            for trig in simple_triggers:
                if trig in lowered:
                    matched_trigger = trig
                    break

            if matched_trigger is None:
                # "back in <month>" — always a trigger.
                m = re.search(
                    rf"\bback in ({'|'.join(_MONTHS)})\b", lowered
                )
                if m:
                    matched_trigger = m.group(0)

            if matched_trigger is None:
                # "in <month>" ONLY when followed by a 4-digit year or
                # "when i" / "that i".
                m = re.search(
                    rf"\bin ({'|'.join(_MONTHS)})(?:\s+(\d{{4}}|when i|that i))",
                    lowered,
                )
                if m:
                    matched_trigger = m.group(0)

            if matched_trigger is None:
                # "on <month> <day>" e.g. "on April 14th".
                m = re.search(
                    rf"\bon ({'|'.join(_MONTHS)})\s+\d+",
                    lowered,
                )
                if m:
                    matched_trigger = m.group(0)

            if matched_trigger is None:
                # "what do i usually X" — require at least one topic word
                # after "usually".
                m = re.search(r"\bwhat do i usually (\w+(?:\s+\w+)+)", lowered)
                if m:
                    matched_trigger = "what do i usually"

            if matched_trigger is None:
                return None

            # Strip the trigger from the text and clean what remains.
            idx = lowered.find(matched_trigger)
            remainder = (
                lowered[:idx] + lowered[idx + len(matched_trigger):]
            ).strip()
            remainder = remainder.rstrip("?.!,").strip()

            cleaned = self._clean_search_query(remainder)
            if not cleaned or len(cleaned) < 3:
                return None
            return cleaned
        except Exception as e:
            print(f"[Memory] detect_memory_search_query error: {e}")
            return None
