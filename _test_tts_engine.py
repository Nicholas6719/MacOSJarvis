"""
Self-tests for tts_engine.py.

Covers routing logic, exhaustion handling, and a live Fish Audio call.
All Kokoro paths go through a lightweight fake so we don't need to load
the full ONNX model just to exercise the router.
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np

import tts_engine

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"


class FakeKokoro:
    """Stand-in for the real Kokoro model — returns a short silence
    clip at TTS_SAMPLE_RATE so we can verify routing without loading
    the ONNX file."""

    def create(self, text, voice="am_fenrir", speed=1.0, lang="en-us"):
        return np.zeros(int(tts_engine.TTS_SAMPLE_RATE * 0.1), dtype=np.float32), tts_engine.TTS_SAMPLE_RATE


def _load_cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _reset(backend: str = "fishaudio") -> dict:
    cfg = _load_cfg()
    cfg["tts"]["backend"] = backend
    tts_engine._reset_exhaustion_for_testing()
    tts_engine.configure(cfg)
    tts_engine.register_kokoro(FakeKokoro(), "am_fenrir", 1.0)
    return cfg


results: list[tuple[str, bool, str]] = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, False, tb))
        print(f"  FAIL  {name}: {e}")


# ── Tests ─────────────────────────────────────────────────────────────

def test_import():
    assert tts_engine.TTS_SAMPLE_RATE == 24000


def test_active_backend_fishaudio():
    _reset("fishaudio")
    assert tts_engine.get_active_backend() == "fishaudio", tts_engine.get_active_backend()


def test_fresh_not_exhausted():
    _reset("fishaudio")
    assert tts_engine.is_fishaudio_exhausted() is False


def test_quota_error_marks_exhausted_and_routes_kokoro():
    _reset("fishaudio")

    class QuotaErr(Exception):
        pass

    def boom(text):
        raise QuotaErr("Insufficient credits for this request")

    original = tts_engine._synthesize_fishaudio
    tts_engine._synthesize_fishaudio = boom
    try:
        out = tts_engine.synthesize("Hello")
    finally:
        tts_engine._synthesize_fishaudio = original

    assert tts_engine.is_fishaudio_exhausted() is True
    assert isinstance(out, np.ndarray) and out.size > 0
    # After exhaustion, router should report kokoro
    assert tts_engine.get_active_backend() == "kokoro"
    # And another call should skip Fish Audio entirely
    out2 = tts_engine.synthesize("Again")
    assert isinstance(out2, np.ndarray) and out2.size > 0


def test_transient_error_does_not_mark_exhausted():
    _reset("fishaudio")

    class NetErr(Exception):
        pass

    def boom(text):
        raise NetErr("connection reset by peer")

    original = tts_engine._synthesize_fishaudio
    tts_engine._synthesize_fishaudio = boom
    try:
        out = tts_engine.synthesize("Hello")
    finally:
        tts_engine._synthesize_fishaudio = original

    assert tts_engine.is_fishaudio_exhausted() is False
    assert isinstance(out, np.ndarray) and out.size > 0


def test_kokoro_backend_setting():
    _reset("kokoro")
    assert tts_engine.get_active_backend() == "kokoro"
    out = tts_engine.synthesize("Hello")
    assert isinstance(out, np.ndarray) and out.size > 0


def test_live_fishaudio_hello():
    """Attempts a real Fish Audio call. Two acceptable outcomes:

    1. Credits available: we get back a real audio clip in the exact
       format SeamlessPlayer expects (float32 mono @ 24 kHz).
    2. Credits exhausted: Fish Audio returns 402, the engine correctly
       flips _fish_audio_exhausted, and we fall back to Kokoro silently.
       The API key is still confirmed valid (no 401)."""
    _reset("fishaudio")
    if tts_engine._fish_client is None:
        raise RuntimeError("Fish Audio client not initialized — check API key in config")

    # Call the raw Fish Audio path first so we can report exactly what
    # the API did on a fresh run.
    raw_ok = False
    raw_exc: Exception | None = None
    try:
        raw_audio = tts_engine._synthesize_fishaudio("Hello")
        raw_ok = True
    except Exception as e:
        raw_exc = e

    if raw_ok:
        assert isinstance(raw_audio, np.ndarray)
        assert raw_audio.dtype == np.float32
        assert raw_audio.ndim == 1
        assert raw_audio.size > tts_engine.TTS_SAMPLE_RATE * 0.1, (
            f"audio too short: {raw_audio.size} samples"
        )
        assert float(np.max(np.abs(raw_audio))) <= 1.0 + 1e-3
        print(f"    LIVE Fish Audio OK: {raw_audio.size} samples "
              f"({raw_audio.size / tts_engine.TTS_SAMPLE_RATE:.2f}s)")
        return

    status = getattr(raw_exc, "status", None)
    # 401 = bad key, that's a real failure of this test. Any other
    # exhaustion-style error (402 / 429 / quota) proves the key/voice
    # are accepted and the fallback path activates.
    assert status != 401, f"Fish Audio rejected the API key: {raw_exc}"
    assert tts_engine._is_exhaustion_error(raw_exc), (
        f"Unexpected non-exhaustion error from Fish Audio: {type(raw_exc).__name__}: {raw_exc}"
    )
    print(f"    LIVE Fish Audio returned {type(raw_exc).__name__} "
          f"(status={status}) — credits exhausted, key valid, fallback engages.")

    # Now drive the public synthesize() path and confirm fallback works.
    _reset("fishaudio")
    original = tts_engine._synthesize_fishaudio
    def _raise_real(text, _e=raw_exc):
        raise _e
    tts_engine._synthesize_fishaudio = _raise_real
    try:
        out = tts_engine.synthesize("Hello")
    finally:
        tts_engine._synthesize_fishaudio = original
    assert isinstance(out, np.ndarray) and out.size > 0
    assert tts_engine.is_fishaudio_exhausted() is True


if __name__ == "__main__":
    print("tts_engine self-tests")
    print("─" * 60)
    check("imports_ok", test_import)
    check("active_backend_fishaudio", test_active_backend_fishaudio)
    check("fresh_not_exhausted", test_fresh_not_exhausted)
    check("quota_error_marks_exhausted", test_quota_error_marks_exhausted_and_routes_kokoro)
    check("transient_error_preserves_flag", test_transient_error_does_not_mark_exhausted)
    check("kokoro_backend_setting", test_kokoro_backend_setting)
    check("live_fishaudio_hello", test_live_fishaudio_hello)
    print("─" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"{passed}/{total} passed")
    if passed != total:
        sys.exit(1)
