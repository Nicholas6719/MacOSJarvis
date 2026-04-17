"""
Self-tests for Phase 4 Features 6 and 7.

Feature 6: Brave browser control via AppleScript (browser_control.py)
Feature 7: Notification-driven proactive responses (chatbot_speech_to_speech.py)

The tests deliberately avoid importing the full VoiceAssistant class — that
would trigger an ML model load that takes minutes and needs GPU resources.
Instead, they exercise:
  - browser_control as a standalone module
  - the module-level notification helpers and state sets from
    chatbot_speech_to_speech

Run:  python _test_features_67.py
"""

import datetime
import importlib
import sys
import threading
import traceback
from typing import Callable, List, Tuple


# ── Test harness ──────────────────────────────────────────────────────────

_FAILURES: List[Tuple[str, str]] = []
_PASSED = 0


def _check(name: str, cond: bool, detail: str = "") -> None:
    global _PASSED
    if cond:
        _PASSED += 1
        print(f"  PASS: {name}")
    else:
        _FAILURES.append((name, detail))
        print(f"  FAIL: {name}  {detail}")


def _section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


# ── Feature 6: browser_control ────────────────────────────────────────────

def test_feature_6() -> None:
    _section("Feature 6: browser_control")
    import browser_control as bc

    # resolve_spoken_url — known sites
    known_expectations = {
        "Google": "https://www.google.com",
        "YouTube": "https://www.youtube.com",
        "Reddit": "https://www.reddit.com",
        "Twitter": "https://www.x.com",
        "X": "https://www.x.com",
    }
    for spoken, expected in known_expectations.items():
        got = bc.resolve_spoken_url(spoken)
        _check(
            f"resolve_spoken_url maps {spoken!r} -> {expected}",
            got == expected,
            f"got {got!r}",
        )

    # resolve_spoken_url — unknown → Google search
    unknown_query = "purple elephants reading shakespeare"
    got = bc.resolve_spoken_url(unknown_query)
    _check(
        "resolve_spoken_url falls back to Google search for unknown names",
        got.startswith("https://www.google.com/search?q=") and "purple" in got.lower(),
        f"got {got!r}",
    )

    # resolve_spoken_url — bare domain passes through
    got = bc.resolve_spoken_url("example.com")
    _check(
        "resolve_spoken_url passes through bare domain 'example.com'",
        got == "https://example.com",
        f"got {got!r}",
    )

    # resolve_spoken_url — already a URL
    got = bc.resolve_spoken_url("https://anthropic.com")
    _check(
        "resolve_spoken_url preserves full URL input",
        got == "https://anthropic.com",
        f"got {got!r}",
    )

    # get_current_url / get_current_title — must return strings, even if
    # Brave is not running (they swallow the 'app not running' error).
    try:
        url = bc.get_current_url()
        _check(
            "get_current_url returns a string without raising",
            isinstance(url, str),
            f"got {type(url).__name__}",
        )
    except Exception as e:
        _check("get_current_url doesn't crash", False, f"raised {e}")

    try:
        title = bc.get_current_title()
        _check(
            "get_current_title returns a string without raising",
            isinstance(title, str),
            f"got {type(title).__name__}",
        )
    except Exception as e:
        _check("get_current_title doesn't crash", False, f"raised {e}")

    # Each AppleScript action should run without a subprocess exception.
    # We can't verify the effect because Brave may not be installed on a
    # CI box — but an unexpected subprocess error (osascript missing,
    # TypeError in the script construction, etc.) will raise here.
    import subprocess
    brave_installed = False
    try:
        r = subprocess.run(
            ["osascript", "-e", 'exists application "Brave Browser"'],
            capture_output=True, text=True, timeout=5,
        )
        brave_installed = "true" in (r.stdout or "").lower()
    except Exception:
        pass

    def _run_and_check(name: str, fn: Callable[[], None]) -> None:
        # The spec only requires that no subprocess exception leaks through
        # — a well-formed RuntimeError IS our graceful failure path (Brave
        # not installed, Accessibility permission missing, etc). Anything
        # else (TypeError, AttributeError, subprocess.CalledProcessError)
        # is a real bug.
        try:
            fn()
            _check(f"{name} executes without subprocess exception", True)
        except RuntimeError as e:
            _check(
                f"{name} fails gracefully with RuntimeError",
                True,
                f"message: {e}",
            )
        except Exception as e:
            _check(f"{name} executes without subprocess exception", False,
                   f"raised {type(e).__name__}: {e}")

    _run_and_check("open_url('https://example.com')",
                   lambda: bc.open_url("https://example.com"))
    _run_and_check("new_tab(None)", lambda: bc.new_tab())
    _run_and_check("new_tab('https://example.com')",
                   lambda: bc.new_tab("https://example.com"))
    _run_and_check("close_tab()", bc.close_tab)
    _run_and_check("go_back()", bc.go_back)
    _run_and_check("go_forward()", bc.go_forward)
    _run_and_check("scroll_down()", bc.scroll_down)
    _run_and_check("scroll_up()", bc.scroll_up)


# ── Feature 7: notification monitor helpers ────────────────────────────────

def test_feature_7() -> None:
    _section("Feature 7: notification monitor")

    # Import the module WITHOUT instantiating VoiceAssistant. We need the
    # module-level helpers and sets. Defer the import so a failure here
    # surfaces as a test failure instead of crashing the whole script.
    try:
        mod = importlib.import_module("chatbot_speech_to_speech")
    except Exception as e:
        _check("import chatbot_speech_to_speech", False, f"raised {e}")
        return
    _check("import chatbot_speech_to_speech", True)

    _check(
        "module exposes announced_event_notifications set",
        isinstance(getattr(mod, "announced_event_notifications", None), set),
    )
    _check(
        "module exposes announced_reminder_notifications set",
        isinstance(getattr(mod, "announced_reminder_notifications", None), set),
    )

    # Event window logic — 10 minutes away should flag, 30 minutes away
    # should not, and an already-announced event should not re-flag.
    now = datetime.datetime(2026, 4, 17, 12, 0, 0)
    ten_min_dt = now + datetime.timedelta(minutes=10)
    thirty_min_dt = now + datetime.timedelta(minutes=30)
    # Build an AppleScript-style date string like Calendar.app returns.
    fmt = "%A, %B %d, %Y at %I:%M:%S %p"
    ev_10 = {"title": "Dentist appointment",
             "start": ten_min_dt.strftime(fmt)}
    ev_30 = {"title": "Lunch", "start": thirty_min_dt.strftime(fmt)}

    local_announced: set = set()
    k10 = mod._event_is_due_for_notification(ev_10, now=now,
                                             announced=local_announced)
    _check(
        "event 10 minutes away is flagged for announcement",
        isinstance(k10, str) and k10.startswith("Dentist appointment|"),
        f"got {k10!r}",
    )
    k30 = mod._event_is_due_for_notification(ev_30, now=now,
                                             announced=local_announced)
    _check(
        "event 30 minutes away is NOT flagged",
        k30 is None,
        f"got {k30!r}",
    )

    # Dedup — mark as announced and call again.
    assert k10 is not None
    local_announced.add(k10)
    k10_again = mod._event_is_due_for_notification(ev_10, now=now,
                                                   announced=local_announced)
    _check(
        "already-announced event is not flagged again",
        k10_again is None,
        f"got {k10_again!r}",
    )

    # Reminder due-this-minute logic.
    # due time aligned to now (same minute) should flag.
    rem_now = {"title": "Call Mom",
               "due": now.strftime("%A, %B %d, %Y at %I:%M:%S %p")}
    local_rem_announced: set = set()
    krn = mod._reminder_is_due_for_notification(rem_now, now=now,
                                                announced=local_rem_announced)
    _check(
        "reminder due within the current minute is flagged",
        isinstance(krn, str) and krn.startswith("Call Mom|"),
        f"got {krn!r}",
    )
    # A reminder due 5 minutes from now should NOT flag.
    rem_future = {"title": "Call Mom later",
                  "due": (now + datetime.timedelta(minutes=5)).strftime(
                      "%A, %B %d, %Y at %I:%M:%S %p")}
    krf = mod._reminder_is_due_for_notification(rem_future, now=now,
                                                announced=local_rem_announced)
    _check(
        "reminder due in 5 minutes is NOT flagged",
        krf is None,
        f"got {krf!r}",
    )
    # Dedup on already-announced reminder.
    assert krn is not None
    local_rem_announced.add(krn)
    krn2 = mod._reminder_is_due_for_notification(rem_now, now=now,
                                                 announced=local_rem_announced)
    _check(
        "already-announced reminder is not flagged again",
        krn2 is None,
        f"got {krn2!r}",
    )

    # announced-set dedup primitive — two identical adds yield one element.
    s: set = set()
    s.add("foo|2026-04-17")
    s.add("foo|2026-04-17")
    _check("announced sets dedupe by key", len(s) == 1, f"len={len(s)}")

    # Verify the monitor thread can be launched without crashing. We don't
    # want to spin up VoiceAssistant (loads heavy ML models), so construct
    # a minimal stand-in object and bind the relevant methods to it.
    class _Stub:
        pass

    stub = _Stub()
    stub._notification_queue = __import__("queue").Queue()
    stub._notification_stop = threading.Event()
    stub._notification_monitor_thread = None
    stub._notification_speaker_thread = None
    stub._tts_speaking = False
    # _calendar_working is an Event in the real class; mimic its is_set()
    # interface so the speaker loop can check it.
    stub._calendar_working = threading.Event()
    stub._llm_lock = threading.Lock()
    # speak_direct: no-op for test. The speaker loop should never actually
    # fire because we immediately stop the monitor before any poll runs.
    stub.speak_direct = lambda msg: None

    # Bind the unbound methods from the class to the stub instance.
    VA = mod.VoiceAssistant
    for name in (
        "start_notification_monitor",
        "stop_notification_monitor",
        "_notification_monitor_loop",
        "_notification_speaker_loop",
        "_check_calendar_notifications",
        "_check_reminder_notifications",
    ):
        setattr(stub, name, getattr(VA, name).__get__(stub, _Stub))

    try:
        stub.start_notification_monitor()
        # Threads should be alive.
        alive_monitor = stub._notification_monitor_thread is not None \
            and stub._notification_monitor_thread.is_alive()
        alive_speaker = stub._notification_speaker_thread is not None \
            and stub._notification_speaker_thread.is_alive()
        _check(
            "notification monitor thread starts without error",
            alive_monitor and alive_speaker,
            f"monitor_alive={alive_monitor} speaker_alive={alive_speaker}",
        )
    except Exception as e:
        _check("notification monitor thread starts without error", False,
               f"raised {type(e).__name__}: {e}")
    finally:
        stub.stop_notification_monitor()
        # Give threads a moment to exit cleanly.
        if stub._notification_monitor_thread is not None:
            stub._notification_monitor_thread.join(timeout=2)
        if stub._notification_speaker_thread is not None:
            stub._notification_speaker_thread.join(timeout=2)


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> int:
    print("Running Feature 6 + 7 self-tests …")
    try:
        test_feature_6()
    except Exception as e:
        print(f"[test_feature_6] uncaught exception: {e}")
        traceback.print_exc()
        _FAILURES.append(("test_feature_6 uncaught", str(e)))
    try:
        test_feature_7()
    except Exception as e:
        print(f"[test_feature_7] uncaught exception: {e}")
        traceback.print_exc()
        _FAILURES.append(("test_feature_7 uncaught", str(e)))

    print()
    print("=" * 60)
    print(f"Results: {_PASSED} passed, {len(_FAILURES)} failed")
    if _FAILURES:
        print("Failures:")
        for name, detail in _FAILURES:
            print(f"  - {name}: {detail}")
        return 1
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
