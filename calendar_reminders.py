"""
Apple Calendar & Reminders — pure function module.
─────────────────────────────────────────────────────────────────────────────
Every public function here calls `osascript` via subprocess. None of them
run at import time — importing this module is zero-cost and has no side
effects on macOS permissions, Calendar.app, or Reminders.app.

All timeouts are short by design: if macOS is waiting on a first-run
permission dialog, we'd rather fail fast with a speakable error than
freeze Jarvis's main loop for 30+ seconds.

IMPORTANT: every public function may raise RuntimeError. Callers must
handle the exception and keep Jarvis running — never let a calendar
failure propagate into the main voice pipeline.
"""

import datetime
import re
import subprocess
import threading
import time
from typing import Optional

# EventKit via PyObjC is Apple's native Calendar/Reminders API — dramatically
# faster than AppleScript for reads. A Reminders fetch with AppleScript's
# `whose completed is false` takes 24+ seconds against a list with 1700+
# historical items; the same fetch via EventKit completes in ~0.1 seconds.
# We only use it for `get_all_reminders` where AppleScript timed out — the
# calendar reads are already fast enough with the per-calendar `tell` scope
# + skip list approach.
try:
    import EventKit as _EK
    _EK_AVAILABLE = True
except ImportError:
    _EK_AVAILABLE = False


# Delimiters used inside AppleScript output so we can parse free-form text
# back into structured Python dicts. These are unlikely to appear in real
# calendar entries.
_REC = "~~~"
_FIELD = "|||"

# Default subprocess timeout for osascript calls. Short enough that a
# stuck permission dialog can't hang Jarvis indefinitely. Bumped a bit
# from 10s because the first invocation in a session may need to cold-
# launch Calendar.app, which takes ~2-3s before events can be queried.
_DEFAULT_TIMEOUT_S = 20

# Calendars to SKIP during read operations.
#
# The AppleScript `whose start date ≥ X and start date < Y` filter on
# Calendar.app events is catastrophically slow on certain calendars:
#
#   Scheduled Reminders        70+ seconds   (synthetic: aggregates Reminders
#                                             with due dates — Apple has never
#                                             made this efficient to query)
#   Holidays in United States  10+ seconds
#   US Holidays                 7+ seconds
#   Birthdays                   2+ seconds   (auto-generated from Contacts)
#   Siri Suggestions            <1s          (auto-detected events from Mail,
#                                             usually noise)
#
# The user rarely wants these in a "what's on my calendar this week?"
# summary — they're subscriptions, auto-generated, or already visible in
# other apps. Skipping them reduces a typical read from 99s to ~8s and
# keeps Jarvis's calendar responses snappy.
#
# CREATE operations target an explicit calendar name and are unaffected.
_READ_SKIP_CALENDAR_NAMES = (
    "Scheduled Reminders",
    "Siri Suggestions",
    "Birthdays",
    "Holidays in United States",
    "US Holidays",
)


# ── Target-app launch helper ─────────────────────────────────────────────────
#
# macOS normally auto-launches the target of `tell application "Foo"` if Foo
# isn't running. That auto-launch is silently blocked when osascript runs
# inside our LSUIElement (JarvisApp) → Python → osascript chain, and every
# query fails with error -600 "Application isn't running".
#
# AppleScript's own `launch` / `activate` commands get blocked the same way.
# The one mechanism that still works is the shell's `open -a Foo -j`, which
# launches the app hidden without going through AppleEvents, and that's what
# this helper uses.
#
# We check `pgrep` first so the helper is effectively free once Calendar or
# Reminders is already running — only the first query in a session pays the
# ~1 second cold-launch cost.

def _ensure_app_running(app_name: str, max_wait_s: float = 4.0) -> None:
    """Start the given GUI app (by `.app` bundle name) if it isn't already
    running, and wait briefly for its process to appear so subsequent Apple
    Event queries actually have a target."""
    try:
        check = subprocess.run(
            ["pgrep", "-x", app_name],
            capture_output=True, text=True, timeout=3,
        )
        if check.returncode == 0:
            return  # Already running — nothing to do.
    except Exception:
        pass  # Fall through to launch attempt.

    # `-j` = launch hidden (no window steals focus). Still appears briefly
    # in the Dock, which is fine — we just need the process alive.
    try:
        subprocess.run(
            ["open", "-a", app_name, "-j"],
            capture_output=True, timeout=10,
        )
    except Exception as e:
        print(f"[Calendar] open -a {app_name} failed: {e}")
        return

    # Poll for the process to appear, then give it a moment to be ready to
    # answer Apple Events. Calendar.app typically takes 0.5-1.5s cold.
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        time.sleep(0.2)
        try:
            check = subprocess.run(
                ["pgrep", "-x", app_name],
                capture_output=True, text=True, timeout=2,
            )
            if check.returncode == 0:
                # Small extra settle delay — process exists but may not yet
                # have its Apple Event handlers wired up.
                time.sleep(0.4)
                return
        except Exception:
            pass
    # Ran out of time — let the caller's osascript attempt fail with a clear
    # error rather than hanging here indefinitely.
    print(f"[Calendar] warning: {app_name} did not appear after {max_wait_s}s")


# ── Internals ────────────────────────────────────────────────────────────

def _run_osa(script: str, timeout: int = _DEFAULT_TIMEOUT_S) -> str:
    """Execute an AppleScript via osascript (fed on stdin) and return
    stdout stripped. Raises RuntimeError on any failure, including timeout."""
    try:
        result = subprocess.run(
            ["osascript"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"AppleScript timed out after {timeout}s — "
            f"Calendar or Reminders may be waiting on a permission dialog"
        ) from e
    except Exception as e:
        raise RuntimeError(f"osascript failed to launch: {e}") from e

    if result.returncode != 0:
        err = (result.stderr or "").strip() or "unknown AppleScript error"
        raise RuntimeError(err)
    return (result.stdout or "").strip()


def format_time_for_speech(dt: datetime.datetime) -> str:
    """Format just the time portion for TTS. Drops ':00' when minutes are
    zero so Jarvis says '7 PM' instead of '7:00 PM'."""
    if dt.minute == 0:
        return dt.strftime("%-I %p")
    return dt.strftime("%-I:%M %p")


def format_datetime_for_speech(dt: datetime.datetime) -> str:
    """Format a full day + time for TTS. Drops ':00' when minutes are
    zero — 'Saturday, April 18 at 7 PM' vs 'Saturday, April 18 at 6:30 PM'."""
    if dt.minute == 0:
        return dt.strftime("%A, %B %-d at %-I %p")
    return dt.strftime("%A, %B %-d at %-I:%M %p")


def _escape(s: Optional[str]) -> str:
    """Escape a Python string for safe embedding inside an AppleScript
    double-quoted literal."""
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _as_date_block(var: str, dt: datetime.datetime) -> str:
    """Emit AppleScript lines that build an AppleScript date object
    in `var` from a Python datetime. Sets day=1 first so the year/month
    assignments don't overflow (going from Jan 31 to Feb would otherwise
    roll into March)."""
    return (
        f'set {var} to current date\n'
        f'set day of {var} to 1\n'
        f'set year of {var} to {dt.year}\n'
        f'set month of {var} to {dt.month}\n'
        f'set day of {var} to {dt.day}\n'
        f'set hours of {var} to {dt.hour}\n'
        f'set minutes of {var} to {dt.minute}\n'
        f'set seconds of {var} to {dt.second}\n'
    )


def _parse_records(raw: str, fields: list) -> list:
    """Parse delimiter-separated AppleScript output into a list of dicts."""
    if not raw:
        return []
    out = []
    for chunk in raw.split(_REC):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(_FIELD)
        while len(parts) < len(fields):
            parts.append("")
        record = {fields[i]: parts[i].strip() for i in range(len(fields))}
        if record.get(fields[0]):
            out.append(record)
    return out


# ── Read operations ─────────────────────────────────────────────────────

_EVENT_FIELDS = ["title", "start", "end", "location", "notes", "calendar"]


def _read_events_script(start_dt: datetime.datetime, end_dt: datetime.datetime) -> str:
    """Build an AppleScript that dumps every event in [start_dt, end_dt)
    across every calendar, skipping known-slow/synthetic ones.
    Uses bulk property fetches (summary of theEvents, start date of theEvents)
    rather than per-event property loops — 10-100x faster on calendars with
    many events, essential for keeping Jarvis's background worker responsive."""
    # Build AppleScript list literal of calendars to skip: {"a", "b", ...}
    skip_list_as = (
        "{" + ", ".join(f'"{_escape(n)}"' for n in _READ_SKIP_CALENDAR_NAMES) + "}"
    )
    return (
        _as_date_block("theStart", start_dt)
        + _as_date_block("theEnd", end_dt)
        + f'set skipNames to {skip_list_as}\n'
        + 'set recSep to "' + _REC + '"\n'
        + 'set fldSep to "' + _FIELD + '"\n'
        + 'set outputText to ""\n'
        + 'tell application "Calendar"\n'
        + '  repeat with cal in calendars\n'
        + '    set calName to name of cal as string\n'
        + '    if calName is not in skipNames then\n'
        + '      try\n'
        # Switch into the calendar's tell scope before querying events.
        # Without this, `every event of cal whose ...` returns partial
        # references that error out (-1728) on bulk property fetches
        # like `summary of theEvents`. `tell cal` materializes them.
        + '        tell cal\n'
        + '          set theEvents to (every event whose start date is greater than or equal to theStart and start date is less than theEnd)\n'
        + '          if (count of theEvents) > 0 then\n'
        + '            repeat with evt in theEvents\n'
        + '              set evTitle to ""\n'
        + '              try\n'
        + '                set evTitle to summary of evt as string\n'
        + '              end try\n'
        + '              set evStart to ""\n'
        + '              try\n'
        + '                set evStart to (start date of evt) as string\n'
        + '              end try\n'
        + '              set evEnd to ""\n'
        + '              try\n'
        + '                set evEnd to (end date of evt) as string\n'
        + '              end try\n'
        + '              set evLoc to ""\n'
        + '              try\n'
        + '                set l to location of evt\n'
        + '                if l is not missing value then set evLoc to l as string\n'
        + '              end try\n'
        + '              set evNotes to ""\n'
        + '              try\n'
        + '                set n to description of evt\n'
        + '                if n is not missing value then set evNotes to n as string\n'
        + '              end try\n'
        + '              set outputText to outputText & evTitle & fldSep & evStart & fldSep & evEnd & fldSep & evLoc & fldSep & evNotes & fldSep & calName & recSep\n'
        + '            end repeat\n'
        + '          end if\n'
        + '        end tell\n'
        + '      end try\n'
        + '    end if\n'
        + '  end repeat\n'
        + 'end tell\n'
        + 'return outputText\n'
    )


def get_today_events() -> list:
    """Return every calendar event scheduled for today, across all calendars."""
    _ensure_app_running("Calendar")
    today = datetime.date.today()
    start = datetime.datetime.combine(today, datetime.time(0, 0, 0))
    end = start + datetime.timedelta(days=1)
    raw = _run_osa(_read_events_script(start, end), timeout=20)
    return _parse_records(raw, _EVENT_FIELDS)


def get_upcoming_events() -> list:
    """Return every event from today through the end of this coming Saturday.

    If today is Sunday, covers Sun through next Saturday (7 days).
    If today is Saturday, covers only today.
    Otherwise covers today through Saturday of the current week."""
    _ensure_app_running("Calendar")
    today = datetime.date.today()
    # Python weekday: Mon=0 ... Sun=6. We want to cover [today, midnight-after-Saturday).
    days_until_sat = (5 - today.weekday()) % 7
    start = datetime.datetime.combine(today, datetime.time(0, 0, 0))
    end = start + datetime.timedelta(days=days_until_sat + 1)
    raw = _run_osa(_read_events_script(start, end), timeout=25)
    return _parse_records(raw, _EVENT_FIELDS)


_REMINDER_FIELDS = ["title", "due", "notes"]


def get_all_reminders() -> list:
    """Return every incomplete reminder across all Reminders lists.

    Uses EventKit (PyObjC) when available — Apple's native API, which
    completes this query in ~0.1s even against a list with thousands of
    historical completed reminders. AppleScript's `whose completed is
    false` takes 20+ seconds on the same data and has been timing out.
    Falls back to the AppleScript implementation if EventKit isn't
    installed (e.g. dev env without pyobjc-framework-EventKit)."""
    if _EK_AVAILABLE:
        return _get_reminders_via_eventkit()
    return _get_reminders_via_applescript()


def _get_reminders_via_eventkit() -> list:
    """Fetch incomplete reminders via the native EventKit API, sorted by
    due date (earliest first; no-due-date items come last).
    Result shape matches the AppleScript path: dicts with title/due/notes."""
    store = _EK.EKEventStore.alloc().init()
    # Predicate for all incomplete reminders in all calendars.
    predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
        None, None, None
    )

    # fetchReminders... uses an async completion handler. Run it synchronously
    # by blocking on a threading.Event.
    result = {"reminders": None, "done": threading.Event()}

    def _completion(reminders):
        try:
            result["reminders"] = list(reminders) if reminders else []
        except Exception:
            result["reminders"] = []
        finally:
            result["done"].set()

    store.fetchRemindersMatchingPredicate_completion_(predicate, _completion)
    if not result["done"].wait(timeout=10):
        raise RuntimeError("EventKit reminder fetch timed out")

    # Collect with a parallel Python datetime for sorting, then strip it.
    rows = []
    for r in result["reminders"] or []:
        try:
            title = r.title() or ""
        except Exception:
            title = ""
        if not title:
            continue  # No title → skip (matches AppleScript behavior)

        # Due date, if any. dueDateComponents() returns NSDateComponents.
        due_str = ""
        due_dt: Optional[datetime.datetime] = None
        try:
            comps = r.dueDateComponents()
            if comps is not None:
                y = comps.year()
                mo = comps.month()
                d = comps.day()
                h = comps.hour()
                mi = comps.minute()
                if y and mo and d:
                    due_dt = datetime.datetime(
                        y, mo, d,
                        h if (h is not None and h >= 0) else 0,
                        mi if (mi is not None and mi >= 0) else 0,
                    )
                    # Format for speech. Drops ":00" when minutes are zero
                    # so Jarvis says "7 PM" instead of "7:00 PM".
                    due_str = format_datetime_for_speech(due_dt)
        except Exception:
            pass

        # Notes
        try:
            notes = r.notes() or ""
        except Exception:
            notes = ""

        rows.append((due_dt, {"title": title, "due": due_str, "notes": notes}))

    # Sort by due date ascending. Items with no due date sort LAST so
    # "what's due soon" comes first and untimed tasks trail.
    _max_dt = datetime.datetime.max
    rows.sort(key=lambda x: (x[0] is None, x[0] or _max_dt))
    return [row[1] for row in rows]


def _get_reminders_via_applescript() -> list:
    """AppleScript fallback for environments without PyObjC EventKit.
    Slow (20+ seconds on large reminder lists) but portable."""
    _ensure_app_running("Reminders")
    script = (
        'set recSep to "' + _REC + '"\n'
        + 'set fldSep to "' + _FIELD + '"\n'
        + 'set outputText to ""\n'
        + 'tell application "Reminders"\n'
        + '  repeat with lst in lists\n'
        + '    try\n'
        + '      tell lst\n'
        + '        set openReminders to (every reminder whose completed is false)\n'
        + '        repeat with r in openReminders\n'
        + '          set rName to ""\n'
        + '          try\n'
        + '            set rName to name of r as string\n'
        + '          end try\n'
        + '          set rDue to ""\n'
        + '          try\n'
        + '            set d to due date of r\n'
        + '            if d is not missing value then set rDue to d as string\n'
        + '          end try\n'
        + '          set rBody to ""\n'
        + '          try\n'
        + '            set b to body of r\n'
        + '            if b is not missing value then set rBody to b as string\n'
        + '          end try\n'
        + '          set outputText to outputText & rName & fldSep & rDue & fldSep & rBody & recSep\n'
        + '        end repeat\n'
        + '      end tell\n'
        + '    end try\n'
        + '  end repeat\n'
        + 'end tell\n'
        + 'return outputText\n'
    )
    raw = _run_osa(script, timeout=20)
    return _parse_records(raw, _REMINDER_FIELDS)


def get_calendar_names() -> list:
    """Return the names of every calendar in Apple Calendar."""
    _ensure_app_running("Calendar")
    script = (
        'set outputText to ""\n'
        + 'tell application "Calendar"\n'
        + '  repeat with cal in calendars\n'
        + '    try\n'
        + '      set outputText to outputText & (name of cal as string) & "' + _FIELD + '"\n'
        + '    end try\n'
        + '  end repeat\n'
        + 'end tell\n'
        + 'return outputText\n'
    )
    raw = _run_osa(script, timeout=10)
    return [n.strip() for n in raw.split(_FIELD) if n.strip()]


# ── Write operations ────────────────────────────────────────────────────

def create_calendar_event(
    title: str,
    calendar_name: str,
    start_datetime: datetime.datetime,
    end_datetime: Optional[datetime.datetime] = None,
    location: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Create an event in the specified Apple Calendar.
    If end_datetime is None, defaults to 1 hour after start_datetime."""
    _ensure_app_running("Calendar")
    if end_datetime is None:
        end_datetime = start_datetime + datetime.timedelta(hours=1)

    props = [
        f'summary:"{_escape(title)}"',
        'start date:theStart',
        'end date:theEnd',
    ]
    if location:
        props.append(f'location:"{_escape(location)}"')
    if notes:
        props.append(f'description:"{_escape(notes)}"')
    props_str = ", ".join(props)

    script = (
        _as_date_block("theStart", start_datetime)
        + _as_date_block("theEnd", end_datetime)
        + f'tell application "Calendar"\n'
        + f'  tell calendar "{_escape(calendar_name)}"\n'
        + f'    make new event with properties {{{props_str}}}\n'
        + '  end tell\n'
        + 'end tell\n'
    )
    _run_osa(script, timeout=20)


def create_reminder(
    title: str,
    due_datetime: Optional[datetime.datetime] = None,
    notes: Optional[str] = None,
) -> None:
    """Create a reminder in the default Reminders list."""
    _ensure_app_running("Reminders")
    props = [f'name:"{_escape(title)}"']
    date_block = ""
    if due_datetime is not None:
        date_block = _as_date_block("theDue", due_datetime)
        props.append('due date:theDue')
    if notes:
        props.append(f'body:"{_escape(notes)}"')
    props_str = ", ".join(props)

    script = (
        date_block
        + 'tell application "Reminders"\n'
        + '  tell default list\n'
        + f'    make new reminder with properties {{{props_str}}}\n'
        + '  end tell\n'
        + 'end tell\n'
    )
    _run_osa(script, timeout=20)


# ── Classification ─────────────────────────────────────────────────────

_WORK_KEYWORDS = (
    "working", "shift", "meeting", "office", "job", "client",
    "stand-up", "standup", "sprint", "scrum", "1:1", "one on one",
)
_FAMILY_KEYWORDS = (
    "mom", "dad", "family", "kids", "dinner with", "birthday",
    "brother", "sister", "grandma", "grandpa", "aunt", "uncle",
)
_HOME_KEYWORDS = (
    "appointment", "dentist", "doctor", "groceries",
    "haircut", "errand", "pickup",
)


def classify_calendar(user_text: str) -> str:
    """Pick the most appropriate calendar name based on keywords in the
    user's utterance. Falls back to 'Home' by default. If the classified
    calendar does not exist in the user's actual calendar list, falls back
    to the first available calendar instead."""
    t = (user_text or "").lower()
    chosen = "Home"
    if any(k in t for k in _WORK_KEYWORDS):
        chosen = "Work"
    elif any(k in t for k in _FAMILY_KEYWORDS):
        chosen = "Family"
    elif any(k in t for k in _HOME_KEYWORDS):
        chosen = "Home"

    try:
        available = get_calendar_names()
    except Exception:
        return chosen
    if not available:
        return chosen
    if chosen in available:
        return chosen
    return available[0]


# ── Edit / delete / complete operations (EventKit-backed) ─────────────────────
#
# All mutations return a 2-tuple: (success: bool, matched_title_or_error: str).
# The matched title (on success) is what the callers speak back to the user
# so they can confirm we acted on the right item. Failures return a short
# human-readable error string.
#
# Target items are found by fuzzy title match — users won't repeat exact
# titles out loud, so we take a title HINT ("Amazon smartwatch") and find
# the best match among open reminders / upcoming events ("Contact Amazon
# smart watch refund").


def _fuzzy_match_score(query: str, candidate: str) -> float:
    """Return 0.0 to 1.0 — fraction of significant query words that appear
    inside the candidate title. Space-insensitive: 'smartwatch' matches
    'smart watch'. Words shorter than 3 characters are ignored so filler
    like 'the', 'my', 'a' doesn't dilute the score."""
    if not query or not candidate:
        return 0.0
    q_words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if not q_words:
        return 0.0
    c_lower = candidate.lower()
    c_nospace = c_lower.replace(" ", "")
    matches = 0
    for w in q_words:
        if w in c_lower or w in c_nospace:
            matches += 1
    return matches / len(q_words)


def _find_best_reminder(title_hint: str, store=None):
    """Fetch incomplete reminders via EventKit, return the EKReminder whose
    title best matches title_hint. Returns None if nothing scores above 0.5."""
    if not _EK_AVAILABLE:
        return None
    if store is None:
        store = _EK.EKEventStore.alloc().init()

    predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
        None, None, None
    )
    result = {"reminders": None, "done": threading.Event()}

    def _completion(reminders):
        try:
            result["reminders"] = list(reminders) if reminders else []
        except Exception:
            result["reminders"] = []
        finally:
            result["done"].set()

    store.fetchRemindersMatchingPredicate_completion_(predicate, _completion)
    if not result["done"].wait(timeout=10):
        return None

    best_score = 0.0
    best = None
    for r in result["reminders"] or []:
        try:
            title = r.title() or ""
        except Exception:
            title = ""
        score = _fuzzy_match_score(title_hint, title)
        if score > best_score:
            best_score = score
            best = r
    return best if best_score >= 0.5 else None


def complete_reminder(title_hint: str) -> tuple:
    """Find and mark-complete the incomplete reminder best matching title_hint.
    Returns (True, matched_title) on success, (False, error_message) otherwise.
    EventKit-based — a full marker-complete operation takes <0.2 seconds."""
    if not _EK_AVAILABLE:
        return (False, "EventKit is not available")
    store = _EK.EKEventStore.alloc().init()
    reminder = _find_best_reminder(title_hint, store=store)
    if reminder is None:
        return (False, f"no open reminder matches {title_hint!r}")

    try:
        title = reminder.title() or title_hint
    except Exception:
        title = title_hint

    try:
        reminder.setCompleted_(True)
        success, err = store.saveReminder_commit_error_(reminder, True, None)
        if success:
            return (True, title)
        return (False, f"save failed: {err}")
    except Exception as e:
        return (False, f"error: {e}")


def delete_reminder(title_hint: str) -> tuple:
    """Find and permanently delete the reminder best matching title_hint.
    Returns (True, matched_title) on success, (False, error_message) otherwise."""
    if not _EK_AVAILABLE:
        return (False, "EventKit is not available")
    store = _EK.EKEventStore.alloc().init()
    reminder = _find_best_reminder(title_hint, store=store)
    if reminder is None:
        return (False, f"no open reminder matches {title_hint!r}")

    try:
        title = reminder.title() or title_hint
    except Exception:
        title = title_hint

    try:
        success, err = store.removeReminder_commit_error_(reminder, True, None)
        if success:
            return (True, title)
        return (False, f"delete failed: {err}")
    except Exception as e:
        return (False, f"error: {e}")


def update_reminder(
    title_hint: str,
    new_title: Optional[str] = None,
    new_due: Optional[datetime.datetime] = None,
    new_notes: Optional[str] = None,
) -> tuple:
    """Find the best-matching open reminder and update any non-None fields.
    Returns (True, matched_original_title) or (False, error)."""
    if not _EK_AVAILABLE:
        return (False, "EventKit is not available")
    store = _EK.EKEventStore.alloc().init()
    reminder = _find_best_reminder(title_hint, store=store)
    if reminder is None:
        return (False, f"no open reminder matches {title_hint!r}")

    try:
        original_title = reminder.title() or title_hint
    except Exception:
        original_title = title_hint

    try:
        import Foundation
        if new_title:
            reminder.setTitle_(new_title)
        if new_due is not None:
            # EKReminder expects NSDateComponents for its due date AND the
            # components need an explicit NSCalendar attached, otherwise
            # EventKit can't resolve them to a real date and silently
            # discards the new value — the save reports success but the
            # reminder keeps its old due time (the exact bug Nicholas hit).
            gregorian = Foundation.NSCalendar.alloc().initWithCalendarIdentifier_(
                Foundation.NSCalendarIdentifierGregorian
            )
            comps = Foundation.NSDateComponents.alloc().init()
            comps.setCalendar_(gregorian)
            comps.setYear_(new_due.year)
            comps.setMonth_(new_due.month)
            comps.setDay_(new_due.day)
            comps.setHour_(new_due.hour)
            comps.setMinute_(new_due.minute)
            reminder.setDueDateComponents_(comps)

            # Reminders created with a due date also carry an NSAlarm that
            # fires at the old time. Setting dueDateComponents does not
            # update the alarm, and the alarm's trigger date is what
            # Reminders.app sorts and displays by on macOS 14+ — so the
            # UI still shows the old time unless we also replace the
            # alarm. Remove any existing alarms and add a fresh one at
            # the new due date.
            existing = list(reminder.alarms() or [])
            for a in existing:
                try:
                    reminder.removeAlarm_(a)
                except Exception:
                    pass
            try:
                ns_due = Foundation.NSDate.dateWithTimeIntervalSince1970_(
                    new_due.timestamp()
                )
                new_alarm = _EK.EKAlarm.alarmWithAbsoluteDate_(ns_due)
                reminder.addAlarm_(new_alarm)
            except Exception as e:
                print(f"[Calendar] alarm update warning: {e}")
        if new_notes is not None:
            reminder.setNotes_(new_notes)
        success, err = store.saveReminder_commit_error_(reminder, True, None)
        if success:
            # Force a store refresh so subsequent reads see the change.
            # Without this, an immediate `get_all_reminders()` call may
            # still return the cached old value.
            try:
                store.refreshSourcesIfNecessary()
            except Exception:
                pass
            return (True, original_title)
        return (False, f"save failed: {err}")
    except Exception as e:
        return (False, f"error: {e}")


def _find_best_event(
    title_hint: str,
    date_hint: Optional[datetime.date] = None,
    window_days: int = 60,
    store=None,
):
    """Search events in a date window around date_hint (or today +/- window_days),
    return the EKEvent whose title best matches title_hint. None if no match."""
    if not _EK_AVAILABLE:
        return None
    if store is None:
        store = _EK.EKEventStore.alloc().init()

    # Define the search window
    if date_hint is not None:
        anchor = datetime.datetime.combine(date_hint, datetime.time(0, 0, 0))
        start_dt = anchor - datetime.timedelta(days=2)
        end_dt = anchor + datetime.timedelta(days=2)
    else:
        start_dt = datetime.datetime.now() - datetime.timedelta(days=2)
        end_dt = datetime.datetime.now() + datetime.timedelta(days=window_days)

    import Foundation
    ns_start = Foundation.NSDate.dateWithTimeIntervalSince1970_(start_dt.timestamp())
    ns_end = Foundation.NSDate.dateWithTimeIntervalSince1970_(end_dt.timestamp())

    predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
        ns_start, ns_end, None
    )
    events = store.eventsMatchingPredicate_(predicate) or []

    best_score = 0.0
    best = None
    for e in events:
        try:
            title = e.title() or ""
        except Exception:
            title = ""
        score = _fuzzy_match_score(title_hint, title)
        if score > best_score:
            best_score = score
            best = e
    return best if best_score >= 0.5 else None


def delete_calendar_event(
    title_hint: str,
    date_hint: Optional[datetime.date] = None,
) -> tuple:
    """Find and delete the calendar event best matching title_hint.
    An optional date_hint narrows the search window. Returns
    (True, matched_title) on success or (False, error)."""
    if not _EK_AVAILABLE:
        return (False, "EventKit is not available")
    store = _EK.EKEventStore.alloc().init()
    event = _find_best_event(title_hint, date_hint=date_hint, store=store)
    if event is None:
        return (False, f"no upcoming event matches {title_hint!r}")

    try:
        title = event.title() or title_hint
    except Exception:
        title = title_hint

    try:
        # EKSpanThisEvent = only this occurrence, not future repeats. Safer.
        success, err = store.removeEvent_span_commit_error_(
            event, _EK.EKSpanThisEvent, True, None
        )
        if success:
            return (True, title)
        return (False, f"delete failed: {err}")
    except Exception as e:
        return (False, f"error: {e}")
