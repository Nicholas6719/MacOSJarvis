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
