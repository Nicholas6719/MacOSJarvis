"""
TTS synthesis abstraction layer.

Routes every synthesis call in Jarvis through a single `synthesize(text)`
entry point. Primary backend is Fish Audio cloud TTS; fallback is the
local Kokoro ONNX model. When Fish Audio returns a credit-exhaustion
style error, all subsequent calls in this process route straight to
Kokoro — no further API attempts until restart.

Returns a float32 mono numpy array at TTS_SAMPLE_RATE (24 kHz), matching
what Kokoro already produces so the existing SeamlessPlayer pipeline
consumes the output unchanged.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("jarvis")

TTS_SAMPLE_RATE = 24_000

_EXHAUSTION_KEYWORDS = (
    "quota", "credit", "exhausted", "balance", "insufficient", "402",
)
_EXHAUSTION_STATUSES = {401, 402, 429}

_fish_audio_exhausted: bool = False
_config: dict = {}
_fish_client: Any = None
_fish_lock = threading.Lock()

_kokoro_ctx: dict = {}  # {'kokoro': obj, 'voice': str, 'speed': float}


def configure(config: dict) -> None:
    """Load the tts block from config.json and prepare the Fish Audio
    client if that backend is selected. Safe to call more than once."""
    global _config, _fish_client
    _config = config
    tts_cfg = config.get("tts", {})
    backend = tts_cfg.get("backend", "kokoro")
    if backend == "fishaudio":
        api_key = tts_cfg.get("fishaudio_api_key")
        if not api_key:
            logger.warning("Fish Audio backend selected but no API key in config — using Kokoro")
            _fish_client = None
            return
        try:
            from fish_audio_sdk import Session
            _fish_client = Session(api_key)
        except Exception as e:
            logger.warning(f"Fish Audio SDK init failed: {e} — using Kokoro")
            _fish_client = None


def register_kokoro(kokoro: Any, voice: str, speed: float) -> None:
    """Hand the already-loaded Kokoro model to the engine so we don't
    load it twice."""
    _kokoro_ctx["kokoro"] = kokoro
    _kokoro_ctx["voice"] = voice
    _kokoro_ctx["speed"] = float(speed)


def get_active_backend() -> str:
    """Return 'fishaudio' if Fish Audio is configured, initialized, and
    not exhausted; otherwise 'kokoro'."""
    tts_cfg = _config.get("tts", {})
    backend = tts_cfg.get("backend", "kokoro")
    if (
        backend == "fishaudio"
        and not _fish_audio_exhausted
        and _fish_client is not None
    ):
        return "fishaudio"
    return "kokoro"


def is_fishaudio_exhausted() -> bool:
    return _fish_audio_exhausted


def _mark_exhausted() -> None:
    global _fish_audio_exhausted
    _fish_audio_exhausted = True


def _reset_exhaustion_for_testing() -> None:
    """Used only by _test_tts_engine.py."""
    global _fish_audio_exhausted
    _fish_audio_exhausted = False


def _is_exhaustion_error(exc: BaseException) -> bool:
    status = getattr(exc, "status", None)
    if isinstance(status, int) and status in _EXHAUSTION_STATUSES:
        return True
    cls = exc.__class__.__name__.lower()
    if "auth" in cls or "ratelimit" in cls or "rate_limit" in cls:
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in _EXHAUSTION_KEYWORDS)


def _synthesize_kokoro(text: str) -> np.ndarray:
    kokoro = _kokoro_ctx.get("kokoro")
    if kokoro is None:
        raise RuntimeError(
            "Kokoro engine not registered — call tts_engine.register_kokoro() first"
        )
    samples, _sr = kokoro.create(
        text,
        voice=_kokoro_ctx["voice"],
        speed=_kokoro_ctx["speed"],
        lang="en-us",
    )
    return np.asarray(samples, dtype=np.float32)


def _synthesize_fishaudio(text: str) -> np.ndarray:
    """Stream PCM directly from Fish Audio at our playback sample rate
    so no resample step is needed. Returns float32 mono in [-1, 1]."""
    from fish_audio_sdk import TTSRequest

    tts_cfg = _config.get("tts", {})
    voice_id = tts_cfg.get("fishaudio_voice_id")
    backend = tts_cfg.get("fishaudio_model", "s2-pro")

    req = TTSRequest(
        text=text,
        reference_id=voice_id,
        format="pcm",
        sample_rate=TTS_SAMPLE_RATE,
        latency="balanced",
    )

    buf = bytearray()
    # Serialize Fish Audio calls — the SDK's streaming iterator is not
    # thread-safe on a shared Session and speak_direct can be invoked
    # from overlapping threads.
    with _fish_lock:
        for chunk in _fish_client.tts(req, backend=backend):
            if chunk:
                buf.extend(chunk)

    if not buf:
        raise RuntimeError("Fish Audio returned empty audio stream")

    pcm_i16 = np.frombuffer(bytes(buf), dtype=np.int16)
    return (pcm_i16.astype(np.float32) / 32768.0).copy()


def synthesize(text: str) -> np.ndarray:
    """Single entry point for all TTS synthesis.

    Returns float32 mono @ TTS_SAMPLE_RATE — the exact format Kokoro
    produces, so SeamlessPlayer.feed() accepts it unchanged."""
    global _fish_audio_exhausted
    tts_cfg = _config.get("tts", {})
    backend = tts_cfg.get("backend", "kokoro")
    fallback = bool(tts_cfg.get("fallback_on_exhaustion", True))

    if (
        backend != "fishaudio"
        or _fish_audio_exhausted
        or _fish_client is None
    ):
        return _synthesize_kokoro(text)

    try:
        return _synthesize_fishaudio(text)
    except Exception as e:
        if _is_exhaustion_error(e):
            _mark_exhausted()
            logger.warning(
                "Fish Audio credits exhausted or unavailable — switching to Kokoro for this session"
            )
            if not fallback:
                raise
            return _synthesize_kokoro(text)
        logger.warning(f"Fish Audio transient error — falling back to Kokoro for this call: {e}")
        if not fallback:
            raise
        return _synthesize_kokoro(text)
