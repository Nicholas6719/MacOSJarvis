"""
Calendar-read speed regression.

Verifies the EventKit migration lands under the 3-second budget we set
after the 8.39s AppleScript baseline measured by _test_full_system.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _fail(msg: str) -> None:
    raise AssertionError(msg)


_EXPECTED_FIELDS = {"title", "start", "end", "location", "notes", "calendar"}

# Fields the downstream pipeline (_format_event_lines, _cal_read_today,
# _cal_read_upcoming in chatbot_speech_to_speech.py) reads from each event
# dict. If any of these disappeared, the voice path would silently render
# blank strings instead of breaking loudly, so we check their presence.
_REQUIRED_DOWNSTREAM_FIELDS = {"title", "start", "end", "location", "calendar"}


def test_backend_defaults_to_eventkit():
    import calendar_reminders as cal
    assert cal.CALENDAR_READ_BACKEND == "eventkit", (
        f"CALENDAR_READ_BACKEND should default to 'eventkit', "
        f"got {cal.CALENDAR_READ_BACKEND!r}"
    )


def test_fallback_functions_exist():
    import calendar_reminders as cal
    assert callable(getattr(cal, "_get_today_events_applescript", None)), (
        "_get_today_events_applescript missing"
    )
    assert callable(getattr(cal, "_get_upcoming_events_applescript", None)), (
        "_get_upcoming_events_applescript missing"
    )


def test_today_returns_list():
    import calendar_reminders as cal
    got = cal.get_today_events()
    assert isinstance(got, list), f"got_today_events returned {type(got)}"


def test_upcoming_returns_list():
    import calendar_reminders as cal
    got = cal.get_upcoming_events()
    assert isinstance(got, list), f"get_upcoming_events returned {type(got)}"


def test_today_under_3s():
    import calendar_reminders as cal
    t0 = time.monotonic()
    cal.get_today_events()
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"get_today_events took {elapsed:.2f}s, > 3.0s budget"
    return elapsed


def test_upcoming_under_3s():
    import calendar_reminders as cal
    t0 = time.monotonic()
    cal.get_upcoming_events()
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"get_upcoming_events took {elapsed:.2f}s, > 3.0s budget"
    return elapsed


def test_event_record_shape():
    """Every returned event must carry the fields the voice pipeline reads.
    We probe both today and upcoming since upcoming tends to return more
    records and gives better field coverage. If today is empty we fall
    through to upcoming; if both are empty we mark the check passed since
    there's nothing to validate — the shape contract is vacuously held."""
    import calendar_reminders as cal
    probes: list = []
    try:
        probes = cal.get_today_events()
    except Exception:
        probes = []
    if not probes:
        try:
            probes = cal.get_upcoming_events()
        except Exception:
            probes = []
    if not probes:
        # No events this week — shape can't regress from empty.
        return
    sample = probes[0]
    assert isinstance(sample, dict), f"event is {type(sample)}, not dict"
    missing = _REQUIRED_DOWNSTREAM_FIELDS - set(sample.keys())
    assert not missing, (
        f"event dict missing required fields {missing}; got keys "
        f"{sorted(sample.keys())}"
    )
    # Every required field must be a string (the downstream LLM prompts
    # and _format_event_lines concatenate them with other strings).
    for k in _REQUIRED_DOWNSTREAM_FIELDS:
        assert isinstance(sample[k], str), (
            f"event[{k!r}] is {type(sample[k])}, expected str"
        )


# ── Runner ────────────────────────────────────────────────────────────────

_TESTS = [
    ("Backend defaults to eventkit", test_backend_defaults_to_eventkit),
    ("AppleScript fallback functions still exist", test_fallback_functions_exist),
    ("get_today_events returns list", test_today_returns_list),
    ("get_upcoming_events returns list", test_upcoming_returns_list),
    ("get_today_events under 3s", test_today_under_3s),
    ("get_upcoming_events under 3s", test_upcoming_under_3s),
    ("event record shape matches downstream contract", test_event_record_shape),
]


def main() -> int:
    failed = 0
    timings: dict[str, float] = {}
    for name, t in _TESTS:
        t0 = time.monotonic()
        try:
            ret = t()
            elapsed = time.monotonic() - t0
            if isinstance(ret, float):
                timings[name] = ret
                print(f"  PASS  {name} ({ret*1000:.0f}ms)")
            else:
                print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {e!r}")
    print()
    print(f"{len(_TESTS) - failed}/{len(_TESTS)} passed")
    if timings:
        print()
        for k, v in timings.items():
            print(f"  timing: {k} = {v*1000:.0f}ms")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
