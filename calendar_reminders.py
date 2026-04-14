"""
Apple Calendar & Reminders integration for Jarvis.
─────────────────────────────────────────────────────────────────────────────
All operations execute AppleScript via `osascript` subprocess — no third
party libraries. The first time these functions run, macOS will prompt the
user to grant automation access to Calendar / Reminders. This is expected
and must be done manually by the user in System Settings → Privacy & Security.
"""

import datetime
import subprocess
from typing import Optional

# Record & field delimiters used inside AppleScript output so we can parse
# free-form text back into structured Python dicts. These strings are very
# unlikely to appear in real calendar entries.
_REC = "~~~"
_FIELD = "|||"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _run_osa(script: str, timeout: int = 30) -> str:
    """Run an AppleScript via osascript (stdin), return stdout.
    Raises RuntimeError on any AppleScript failure."""
    try:
        result = subprocess.run(
            ["osascript"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"AppleScript timed out: {e}") from e
    except Exception as e:
        raise RuntimeError(f"AppleScript failed to launch: {e}") from e

    if result.returncode != 0:
        err = (result.stderr or "").strip() or "unknown AppleScript error"
        raise RuntimeError(err)
    return (result.stdout or "").strip()


def _escape(s: Optional[str]) -> str:
    """Escape a Python string for safe embedding in an AppleScript string literal."""
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _as_date_block(var: str, dt: datetime.datetime) -> str:
    """Emit AppleScript lines that build a date object in `var` from a
    Python datetime. day is set to 1 first to avoid month-overflow bugs
    (e.g. switching from Jan 31 to Feb would roll forward a day)."""
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
    """Parse the delimiter-separated output from an event/reminder query."""
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


# ── Read operations ─────────────────────────────────────────────────────────

_EVENT_FIELDS = ["title", "start", "end", "location", "notes", "calendar"]


def _read_events_script(start_dt: datetime.datetime, end_dt: datetime.datetime) -> str:
    """Build an AppleScript that dumps every event in [start_dt, end_dt)
    across every calendar, as ~~~-separated records with |||-separated fields."""
    return (
        _as_date_block("theStart", start_dt)
        + _as_date_block("theEnd", end_dt)
        + 'set recSep to "' + _REC + '"\n'
        + 'set fldSep to "' + _FIELD + '"\n'
        + 'set outputText to ""\n'
        + 'tell application "Calendar"\n'
        + '  repeat with cal in calendars\n'
        + '    try\n'
        + '      set calName to name of cal as string\n'
        + '      set theEvents to (every event of cal whose start date is greater than or equal to theStart and start date is less than theEnd)\n'
        + '      repeat with evt in theEvents\n'
        + '        set evTitle to ""\n'
        + '        try\n'
        + '          set evTitle to summary of evt as string\n'
        + '        end try\n'
        + '        set evStart to ""\n'
        + '        try\n'
        + '          set evStart to (start date of evt) as string\n'
        + '        end try\n'
        + '        set evEnd to ""\n'
        + '        try\n'
        + '          set evEnd to (end date of evt) as string\n'
        + '        end try\n'
        + '        set evLoc to ""\n'
        + '        try\n'
        + '          set l to location of evt\n'
        + '          if l is not missing value then set evLoc to l as string\n'
        + '        end try\n'
        + '        set evNotes to ""\n'
        + '        try\n'
        + '          set n to description of evt\n'
        + '          if n is not missing value then set evNotes to n as string\n'
        + '        end try\n'
        + '        set outputText to outputText & evTitle & fldSep & evStart & fldSep & evEnd & fldSep & evLoc & fldSep & evNotes & fldSep & calName & recSep\n'
        + '      end repeat\n'
        + '    end try\n'
        + '  end repeat\n'
        + 'end tell\n'
        + 'return outputText\n'
    )


def get_today_events() -> list:
    """Return every calendar event scheduled for today, across all calendars."""
    today = datetime.date.today()
    start = datetime.datetime.combine(today, datetime.time(0, 0, 0))
    end = start + datetime.timedelta(days=1)
    raw = _run_osa(_read_events_script(start, end), timeout=45)
    return _parse_records(raw, _EVENT_FIELDS)


def get_upcoming_events() -> list:
    """Return every event from today through the end of this coming Saturday.

    Always covers the rest of the current week through Saturday regardless of
    what day it is. If today is Sunday, that means through the following
    Saturday. If today is Saturday, that means just today."""
    today = datetime.date.today()
    # Python weekday: Mon=0, Sun=6. Want to cover [today, end-of-coming-Saturday].
    days_until_sat = (5 - today.weekday()) % 7
    start = datetime.datetime.combine(today, datetime.time(0, 0, 0))
    # end is exclusive: midnight after Saturday
    end = start + datetime.timedelta(days=days_until_sat + 1)
    raw = _run_osa(_read_events_script(start, end), timeout=60)
    return _parse_records(raw, _EVENT_FIELDS)


_REMINDER_FIELDS = ["title", "due", "notes"]


def get_all_reminders() -> list:
    """Return every incomplete reminder across all Reminders lists."""
    script = (
        'set recSep to "' + _REC + '"\n'
        + 'set fldSep to "' + _FIELD + '"\n'
        + 'set outputText to ""\n'
        + 'tell application "Reminders"\n'
        + '  set openReminders to (every reminder whose completed is false)\n'
        + '  repeat with r in openReminders\n'
        + '    set rName to ""\n'
        + '    try\n'
        + '      set rName to name of r as string\n'
        + '    end try\n'
        + '    set rDue to ""\n'
        + '    try\n'
        + '      set d to due date of r\n'
        + '      if d is not missing value then set rDue to d as string\n'
        + '    end try\n'
        + '    set rBody to ""\n'
        + '    try\n'
        + '      set b to body of r\n'
        + '      if b is not missing value then set rBody to b as string\n'
        + '    end try\n'
        + '    set outputText to outputText & rName & fldSep & rDue & fldSep & rBody & recSep\n'
        + '  end repeat\n'
        + 'end tell\n'
        + 'return outputText\n'
    )
    raw = _run_osa(script, timeout=45)
    return _parse_records(raw, _REMINDER_FIELDS)


def get_calendar_names() -> list:
    """Return the names of every calendar in Apple Calendar."""
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
    raw = _run_osa(script, timeout=20)
    return [n.strip() for n in raw.split(_FIELD) if n.strip()]


# ── Write operations ───────────────────────────────────────────────────────

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
    _run_osa(script, timeout=30)


def create_reminder(
    title: str,
    due_datetime: Optional[datetime.datetime] = None,
    notes: Optional[str] = None,
) -> None:
    """Create a reminder in the default Reminders list."""
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
    _run_osa(script, timeout=30)


# ── Classification ──────────────────────────────────────────────────────────

# Keyword buckets used to pick which calendar a new event should go into.
_WORK_KEYWORDS = (
    "working", "shift", "meeting", "office", "job", "client", "work",
    "stand-up", "standup", "sprint", "scrum", "1:1", "one on one",
)
_FAMILY_KEYWORDS = (
    "mom", "dad", "family", "kids", "dinner with", "birthday",
    "brother", "sister", "grandma", "grandpa", "aunt", "uncle",
)
_HOME_KEYWORDS = (
    "appointment", "dentist", "doctor", "groceries", "home",
    "haircut", "errand", "pickup",
)


def classify_calendar(user_text: str) -> str:
    """Return the most appropriate calendar name based on keywords in the
    user's utterance. Falls back to 'Home' by default. If the classified
    calendar does not exist in the user's actual calendar list, fall back
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
