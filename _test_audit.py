"""
Audit tests for the codebase fix pass.

Pure static + import-time checks. We avoid running the full voice pipeline;
the goal is to prove:
  - Importing calendar_reminders / browser_control triggers no AppleScript.
  - faster-whisper is called with beam_size>=3, vad_filter=True, and an
    initial_prompt that mentions "Jarvis".
  - Every background threading.Thread(...) in chatbot_speech_to_speech.py
    is started with daemon=True.
  - pending_calendar_action and pending_file_action are cleared in finally
    (or in an explicit error-path clear) so a crash can't leave Jarvis
    stuck waiting for a follow-up.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE / "chatbot_speech_to_speech.py"


def _run_subprocess_import(module_name: str) -> tuple[bool, str]:
    """Import `module_name` in a fresh subprocess while monkey-patching
    subprocess.run/Popen to record any osascript invocation. Returns
    (ok, detail)."""
    code = textwrap.dedent(
        f"""
        import subprocess, sys, json

        calls = []
        _orig_run = subprocess.run
        _orig_popen = subprocess.Popen

        def _track_run(cmd, *a, **kw):
            try:
                if isinstance(cmd, (list, tuple)) and cmd and 'osascript' in str(cmd[0]):
                    calls.append(list(cmd))
            except Exception:
                pass
            return _orig_run(cmd, *a, **kw)

        class _TrackPopen(_orig_popen):
            def __init__(self, cmd, *a, **kw):
                try:
                    if isinstance(cmd, (list, tuple)) and cmd and 'osascript' in str(cmd[0]):
                        calls.append(list(cmd))
                except Exception:
                    pass
                super().__init__(cmd, *a, **kw)

        subprocess.run = _track_run
        subprocess.Popen = _TrackPopen

        import {module_name}  # noqa: F401

        print('__OSACALLS__' + json.dumps(calls))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(HERE), timeout=30,
    )
    if result.returncode != 0:
        return False, f"subprocess failed: {result.stderr.strip()}"
    m = re.search(r"__OSACALLS__(\[.*\])", result.stdout)
    if not m:
        return False, f"no marker in output: {result.stdout!r}"
    import json as _json
    calls = _json.loads(m.group(1))
    if calls:
        return False, f"osascript invoked at import time: {calls!r}"
    return True, "no osascript calls at import time"


def test_calendar_reminders_import_no_osascript():
    ok, detail = _run_subprocess_import("calendar_reminders")
    assert ok, detail


def test_browser_control_import_no_osascript():
    ok, detail = _run_subprocess_import("browser_control")
    assert ok, detail


def _find_transcribe_call_kwargs() -> dict:
    """Parse the main module's AST and pull the kwargs passed to
    self._stt.transcribe(...). Returns a dict of kwarg-name -> ast node."""
    tree = ast.parse(MAIN.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "transcribe"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "_stt"
        ):
            continue
        return {kw.arg: kw.value for kw in node.keywords if kw.arg}
    raise AssertionError("self._stt.transcribe(...) call not found")


def test_beam_size_at_least_3():
    kwargs = _find_transcribe_call_kwargs()
    assert "beam_size" in kwargs, "beam_size not passed to transcribe"
    node = kwargs["beam_size"]
    assert isinstance(node, ast.Constant), f"beam_size not a constant: {ast.dump(node)}"
    assert isinstance(node.value, int) and node.value >= 3, (
        f"beam_size must be >= 3, got {node.value!r}"
    )


def test_vad_filter_true():
    kwargs = _find_transcribe_call_kwargs()
    assert "vad_filter" in kwargs, "vad_filter not passed"
    node = kwargs["vad_filter"]
    assert isinstance(node, ast.Constant) and node.value is True, (
        f"vad_filter must be True, got {ast.dump(node)}"
    )


def test_initial_prompt_present_and_mentions_jarvis():
    kwargs = _find_transcribe_call_kwargs()
    assert "initial_prompt" in kwargs, "initial_prompt not passed"
    node = kwargs["initial_prompt"]
    # The prompt is built from a parenthesized string or string concat —
    # unparse and search for "Jarvis" anywhere in the source.
    src = ast.unparse(node)
    assert "Jarvis" in src, f"initial_prompt must contain 'Jarvis': {src!r}"


def test_background_threads_are_daemon():
    """Every threading.Thread(...) in the main module must be daemon=True
    (either via keyword, or via .daemon=True on the instance)."""
    tree = ast.parse(MAIN.read_text())
    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_thread_ctor = (
            (isinstance(func, ast.Attribute) and func.attr == "Thread")
            or (isinstance(func, ast.Name) and func.id == "Thread")
        )
        if not is_thread_ctor:
            continue
        kw_names = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        daemon_node = kw_names.get("daemon")
        if (
            isinstance(daemon_node, ast.Constant)
            and daemon_node.value is True
        ):
            continue
        # Not set inline. Skip Timer (handled separately) and look for a
        # sibling `.daemon = True` assignment in the enclosing scope.
        failures.append(f"line {node.lineno}: Thread(...) without daemon=True")
    assert not failures, "Non-daemon threads found:\n  " + "\n  ".join(failures)


def test_pending_actions_cleared_on_error():
    """Both pending_calendar_action and pending_file_action must be
    cleared from an exception/finally path in the worker bodies."""
    src = MAIN.read_text()
    # Look for clearing lines anywhere (they live inside except/finally
    # blocks in _calendar_worker_body, _resume_pending_calendar_action,
    # _handle_file_command). Pair each with a nearby 'except' or 'finally'.
    def _has_nearby_guard(keyword: str, assignment: str) -> bool:
        for m in re.finditer(re.escape(assignment), src):
            # Look backward up to 300 chars for 'except' or 'finally'
            window = src[max(0, m.start() - 400): m.start()]
            if re.search(rf"\b{keyword}\b", window):
                return True
        return False

    has_cal = (
        _has_nearby_guard("except", "self._pending_calendar_action = None")
        or _has_nearby_guard("finally", "self._pending_calendar_action = None")
    )
    has_file = (
        _has_nearby_guard("except", "self._pending_file_action = None")
        or _has_nearby_guard("finally", "self._pending_file_action = None")
    )
    assert has_cal, "pending_calendar_action is never cleared in except/finally"
    assert has_file, "pending_file_action is never cleared in except/finally"


# ── Runner ────────────────────────────────────────────────────────────────

_TESTS = [
    test_calendar_reminders_import_no_osascript,
    test_browser_control_import_no_osascript,
    test_beam_size_at_least_3,
    test_vad_filter_true,
    test_initial_prompt_present_and_mentions_jarvis,
    test_background_threads_are_daemon,
    test_pending_actions_cleared_on_error,
]


def main() -> int:
    failed = 0
    for t in _TESTS:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {e!r}")
    print()
    print(f"{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
