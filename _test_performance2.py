"""
Performance-pass-2 audit tests. Static checks only — no ML model loading.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE / "chatbot_speech_to_speech.py"
CONFIG = HERE / "config.json"


def _load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def _read_main() -> str:
    return MAIN.read_text()


def _module_constant(name: str):
    """Grab a top-level int/float constant from the main module by AST —
    avoids importing the whole module (which would load models)."""
    tree = ast.parse(_read_main())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    if isinstance(node.value, ast.Constant):
                        return node.value.value
    raise AssertionError(f"constant {name!r} not found at module level")


def test_silence_cutoff_short_in_range():
    v = _module_constant("SILENCE_CUTOFF_SHORT_MS")
    assert 650 <= v <= 750, f"SILENCE_CUTOFF_SHORT_MS {v} outside 650-750"


def test_silence_cutoff_long_in_range():
    v = _module_constant("SILENCE_CUTOFF_LONG_MS")
    assert 700 <= v <= 800, f"SILENCE_CUTOFF_LONG_MS {v} outside 700-800"


def test_n_ctx_is_2048():
    cfg = _load_config()
    v = cfg["llm"].get("n_ctx")
    assert v == 2048, f"n_ctx must be 2048, got {v!r}"


def test_use_mmap_passed_to_llama():
    """Find the Llama(...) constructor call in _load_llm and confirm
    use_mmap is passed as a keyword argument with a truthy value."""
    tree = ast.parse(_read_main())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_llama = (
            (isinstance(fn, ast.Name) and fn.id == "Llama")
            or (isinstance(fn, ast.Attribute) and fn.attr == "Llama")
        )
        if not is_llama:
            continue
        for kw in node.keywords:
            if kw.arg == "use_mmap":
                # Accept either a literal True or a config lookup defaulting to True.
                src = ast.unparse(kw.value)
                if "True" in src:
                    return
                raise AssertionError(
                    f"use_mmap kwarg present but doesn't default to True: {src!r}"
                )
        raise AssertionError("Llama(...) call found but no use_mmap kwarg")
    raise AssertionError("no Llama(...) call found in module")


def test_history_capped_at_10():
    """_messages must clamp history_pairs at 10 somewhere in its body,
    so a stale config with history_turns=20 can't exceed the cap."""
    src = _read_main()
    # Look for an explicit clamp (min(...,10) or > 10: raise/clip).
    assert re.search(r"min\([^)]*?,\s*10\)", src) or re.search(
        r"history_pairs\s*=\s*min\(", src
    ), "no min(..., 10) clamp on history in _messages"


def test_simple_conversational_helpers():
    """Import just the helper function without loading the VoiceAssistant
    class (which would trigger ML model loading). We do this by parsing
    the module and exec'ing only the needed constants + function."""
    src = _read_main()
    tree = ast.parse(src)
    wanted_names = {"_FAST_PATH_BANNED", "_FAST_PATH_MAX_WORDS",
                    "is_simple_conversational_turn"}
    subset_nodes: list[ast.stmt] = [
        ast.Import(names=[ast.alias(name="re", asname=None)])
    ]
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id in wanted_names
                   for t in node.targets):
                subset_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_names:
            subset_nodes.append(node)
    module = ast.Module(body=subset_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict = {}
    exec(compile(module, str(MAIN), "exec"), ns)
    f = ns["is_simple_conversational_turn"]
    assert f("How are you") is True, "expected True for 'How are you'"
    assert f("remind me to call mom") is False, "expected False for reminder"
    assert f("what's on my calendar") is False, "expected False for calendar"
    assert f("move my resume to the desktop") is False, "expected False for move"


def test_model_load_timing_logs():
    """The parallel-load block must emit per-model timing logs."""
    src = _read_main()
    # Per-iteration log — each timing entry gets formatted with '{elapsed:.2f}s'.
    assert re.search(
        r"logger\.info\(f[\"']\{label\} loaded in \{elapsed:\.2f\}s[\"']\)",
        src,
    ), "per-model 'loaded in Xs' logger.info line missing"
    # And the Moondream timing in screen_awareness.py.
    sa_src = (HERE / "screen_awareness.py").read_text()
    assert "Moondream loaded in" in sa_src, (
        "Moondream timing log missing from screen_awareness.py"
    )


# ── Runner ────────────────────────────────────────────────────────────────

_TESTS = [
    test_silence_cutoff_short_in_range,
    test_silence_cutoff_long_in_range,
    test_n_ctx_is_2048,
    test_use_mmap_passed_to_llama,
    test_history_capped_at_10,
    test_simple_conversational_helpers,
    test_model_load_timing_logs,
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
