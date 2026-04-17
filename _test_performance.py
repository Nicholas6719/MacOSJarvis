"""
Performance-pass audit tests.

Static checks only — we don't load any ML models or run the pipeline.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
MAIN = HERE / "chatbot_speech_to_speech.py"


def _load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def test_n_gpu_layers_is_minus_1():
    cfg = _load_config()
    v = cfg["llm"].get("n_gpu_layers")
    assert v == -1, f"expected n_gpu_layers=-1, got {v!r}"


def test_n_threads_is_6():
    cfg = _load_config()
    v = cfg["llm"].get("n_threads")
    assert v == 6, f"expected n_threads=6, got {v!r}"


def test_n_batch_is_512():
    cfg = _load_config()
    v = cfg["llm"].get("n_batch")
    assert v == 512, f"expected n_batch=512, got {v!r}"


def test_f16_kv_is_true():
    cfg = _load_config()
    v = cfg["llm"].get("f16_kv")
    assert v is True, f"expected f16_kv=True, got {v!r}"


def test_legacy_text_only_removed():
    assert not (HERE / "chatbot_text_only.py").exists(), (
        "chatbot_text_only.py should have been deleted"
    )


def test_legacy_text_to_speech_removed():
    assert not (HERE / "chatbot_text_to_speech.py").exists(), (
        "chatbot_text_to_speech.py should have been deleted"
    )


# Phrases that indicate Jarvis is narrating its own memory work out loud.
# These must never appear as TTS input — memory operations are silent.
_BANNED_PHRASE_FRAGMENTS = (
    "updating my memory",
    "memory loaded",
    "loading memory",
    "saving fact",
    "summarizing our conversation",
    "remembering that",
    "initializing memory",
)

# Phrases from the old _speak_memory_ack rotation. If any of these reach
# speak_direct, auto-save is still audibly acknowledging.
_BANNED_ACK_PHRASES = (
    "noted, sir",
    "understood, sir",
    "duly noted",
    "i'll keep that in mind",
)


def _iter_string_args_to(func_names: set[str], module_src: str) -> list[tuple[int, str]]:
    """Return (lineno, literal_string) for every call node whose target
    attr is in func_names and whose first positional arg is a string
    constant. Concatenations and f-strings we can statically unparse are
    also flattened to a string for inspection."""
    tree = ast.parse(module_src)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = None
        if isinstance(fn, ast.Attribute):
            name = fn.attr
        elif isinstance(fn, ast.Name):
            name = fn.id
        if name not in func_names:
            continue
        if not node.args:
            continue
        arg = node.args[0]
        try:
            src = ast.unparse(arg)
        except Exception:
            continue
        hits.append((node.lineno, src))
    return hits


def test_no_spoken_memory_narration():
    src = MAIN.read_text()
    calls = _iter_string_args_to({"speak_direct", "_safe_speak"}, src)
    offenders: list[str] = []
    for lineno, s in calls:
        low = s.lower()
        for frag in _BANNED_PHRASE_FRAGMENTS + _BANNED_ACK_PHRASES:
            if frag in low:
                offenders.append(f"line {lineno}: {s}")
                break
    assert not offenders, (
        "Spoken memory narration still present:\n  " + "\n  ".join(offenders)
    )


def test_memory_ack_is_no_op():
    """The auto-save ack method must not call speak_direct or _safe_speak."""
    src = MAIN.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_speak_memory_ack"
        ):
            body_src = ast.unparse(node)
            assert "speak_direct(" not in body_src and "_safe_speak(" not in body_src, (
                "_speak_memory_ack must be a no-op — it still speaks"
            )
            return
    raise AssertionError("_speak_memory_ack method not found")


def test_notification_monitor_startup_delay_is_90s():
    """The loop that holds the monitor dormant for 90 seconds must still
    be a `range(900)` with a 0.1-second sleep per tick."""
    src = MAIN.read_text()
    # Search for the specific construct around _notification_monitor_loop.
    m = re.search(
        r"def _notification_monitor_loop[\s\S]{0,1500}?for _ in range\((\d+)\):"
        r"\s*\n\s*if self\._notification_stop\.is_set\(\):\s*\n\s*return"
        r"\s*\n\s*time\.sleep\(([\d.]+)\)",
        src,
    )
    assert m, "notification monitor startup delay loop not found in expected shape"
    iters = int(m.group(1))
    sleep_s = float(m.group(2))
    total = iters * sleep_s
    assert abs(total - 90.0) < 0.01, (
        f"expected 90s startup delay, got {total}s "
        f"({iters} iterations * {sleep_s}s)"
    )


def test_moondream_warmup_is_daemon_thread():
    """screen_awareness.warm_up_in_background must start a daemon Thread."""
    src = (HERE / "screen_awareness.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "warm_up_in_background"):
            continue
        # Find a Thread(...) call inside this function with daemon=True.
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            fn = inner.func
            is_thread = (
                (isinstance(fn, ast.Attribute) and fn.attr == "Thread")
                or (isinstance(fn, ast.Name) and fn.id == "Thread")
            )
            if not is_thread:
                continue
            for kw in inner.keywords:
                if kw.arg == "daemon" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    return
        raise AssertionError(
            "warm_up_in_background exists but has no Thread(..., daemon=True) call"
        )
    raise AssertionError("warm_up_in_background function not found")


# ── Runner ────────────────────────────────────────────────────────────────

_TESTS = [
    test_n_gpu_layers_is_minus_1,
    test_n_threads_is_6,
    test_n_batch_is_512,
    test_f16_kv_is_true,
    test_legacy_text_only_removed,
    test_legacy_text_to_speech_removed,
    test_no_spoken_memory_narration,
    test_memory_ack_is_no_op,
    test_notification_monitor_startup_delay_is_90s,
    test_moondream_warmup_is_daemon_thread,
]


def main() -> int:
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {e!r}")
    print()
    print(f"{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
