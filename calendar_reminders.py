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
                    # Format the same way AppleScript does:
                    # "Saturday, April 18, 2026 at 10:30:00 AM"
                    due_str = due_dt.strftime("%A, %B %-d, %Y at %-I:%M:%S %p")
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
