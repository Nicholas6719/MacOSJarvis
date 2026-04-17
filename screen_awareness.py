"""
Screen awareness for Jarvis — one-shot screenshot → Moondream 2 description.

Privacy contract:
  • Screenshots are captured to /tmp only (/tmp/jarvis_screen_capture.png).
  • Each screenshot is used once for a single spoken description, then
    immediately deleted by cleanup_screenshot().
  • Screenshots are NEVER stored long-term, logged, persisted to the
    Jarvis memory database, or transmitted off-device. Moondream 2 runs
    fully locally.

macOS screencapture requires Screen Recording permission. The user grants
this once in System Settings → Privacy & Security → Screen Recording.
"""

from __future__ import annotations

import os
import subprocess
import threading
import urllib.request
from pathlib import Path
from typing import Optional


# ── Feature flag ──────────────────────────────────────────────────────────────
# "orb"       → show the captured screenshot in the orb WebView while the
#               description is being generated and spoken; clear after.
# "quicklook" → silent capture → describe → speak (no orb preview).
# This is the ONLY toggle point. No other code needs to change.
SCREEN_PREVIEW_MODE = "orb"


# ── Paths ─────────────────────────────────────────────────────────────────────
SCREENSHOT_PATH = "/tmp/jarvis_screen_capture.png"

# Moondream 2 model cache. We default to the 0.5B int8 model (~500 MB) —
# it's 3-4x faster than the 2B on Apple Silicon and plenty accurate for
# "what's on my screen" style questions. Override via JARVIS_MOONDREAM_MODEL
# if you want the larger 2B.
_MOONDREAM_CACHE_DIR = Path.home() / ".cache" / "moondream"
_MOONDREAM_MODEL_FILENAME = os.environ.get(
    "JARVIS_MOONDREAM_MODEL", "moondream-2b-int8.mf.gz"
)
_MOONDREAM_MODEL_URL = (
    "https://huggingface.co/vikhyatk/moondream2/resolve/onnx/"
    + _MOONDREAM_MODEL_FILENAME
)

# Max pixels on the long edge before we downscale. Moondream crops to
# 378px tiles internally, so anything above ~1024 is pure overhead —
# on a Retina display that's a 3-4x speedup in the vision encoder.
_MAX_IMAGE_LONG_EDGE = 1024

# Cap generated tokens so the tail of generation doesn't dominate latency.
# 80 tokens is ~2-4 sentences of speech — matches the prompt guidance.
_MAX_OUTPUT_TOKENS = 80

# Cached model instance — loaded once per process, lazily.
_vision_model = None
_vision_model_lock = threading.Lock()


# ── Model loading ─────────────────────────────────────────────────────────────

def _ensure_model_file() -> str:
    """Ensure the Moondream .mf.gz file exists in the cache, downloading
    it on first use. Returns the absolute path to the model file."""
    _MOONDREAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MOONDREAM_CACHE_DIR / _MOONDREAM_MODEL_FILENAME
    if model_path.is_file() and model_path.stat().st_size > 100_000_000:
        return str(model_path)

    print(f"[screen_awareness] downloading Moondream weights to {model_path} "
          f"(first run only)")
    tmp = model_path.with_suffix(model_path.suffix + ".part")
    try:
        urllib.request.urlretrieve(_MOONDREAM_MODEL_URL, str(tmp))
        tmp.rename(model_path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    return str(model_path)


def load_vision_model():
    """Lazy-load Moondream. Cached for the life of the process so repeat
    invocations are instant. Thread-safe — calling from both the warm-up
    thread and the voice handler is fine."""
    global _vision_model
    if _vision_model is not None:
        return _vision_model
    with _vision_model_lock:
        if _vision_model is not None:
            return _vision_model
        import moondream as md  # imported lazily so startup isn't affected
        model_path = _ensure_model_file()
        _vision_model = md.vl(model=model_path)
    return _vision_model


def warm_up_in_background() -> None:
    """Fire-and-forget background load so the first real screen-describe
    call doesn't pay the model-load cost. Safe to call at startup even
    before the model file has been downloaded."""
    def _run() -> None:
        try:
            load_vision_model()
            print("[screen_awareness] vision model warm.")
        except Exception as e:
            print(f"[screen_awareness] warm-up failed: {e}")
    threading.Thread(target=_run, daemon=True, name="moondream-warmup").start()


# ── Capture ───────────────────────────────────────────────────────────────────

def capture_screen() -> str:
    """Capture the full screen to SCREENSHOT_PATH using macOS `screencapture
    -x` (silent, no shutter sound). Returns the path."""
    # Remove any stale file first so a failure can't be mistaken for success.
    try:
        if os.path.exists(SCREENSHOT_PATH):
            os.remove(SCREENSHOT_PATH)
    except OSError:
        pass
    subprocess.run(
        ["screencapture", "-x", SCREENSHOT_PATH],
        check=False,
        timeout=10,
    )
    return SCREENSHOT_PATH


# ── Describe ──────────────────────────────────────────────────────────────────

# Short, imperative prompt — fewer input tokens = faster prefill, and the
# 0.5B model follows terse instructions better than verbose ones.
_DESCRIBE_PROMPT = (
    "Describe what is on this screen in 1 to 2 short sentences. "
    "Focus on what is most prominent or actively in use."
)


def _prepare_image(image_path: str):
    """Open, convert to RGB, and downscale so the long edge is at most
    _MAX_IMAGE_LONG_EDGE pixels. Moondream tiles to 378px internally —
    feeding it a 4K Retina screenshot is pure overhead."""
    from PIL import Image
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > _MAX_IMAGE_LONG_EDGE:
        scale = _MAX_IMAGE_LONG_EDGE / float(long_edge)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def describe_screen(image_path: str) -> str:
    """Load the image at image_path, query Moondream, return the
    description string (stripped). Image is downscaled first and output
    is capped at _MAX_OUTPUT_TOKENS for latency."""
    model = load_vision_model()
    img = _prepare_image(image_path)

    settings = {"max_tokens": _MAX_OUTPUT_TOKENS}
    try:
        encoded = model.encode_image(img)
        result = model.query(encoded, _DESCRIBE_PROMPT, settings=settings)
    except TypeError:
        # Older moondream builds may not accept settings kwarg.
        encoded = model.encode_image(img)
        result = model.query(encoded, _DESCRIBE_PROMPT)
    except AttributeError:
        result = model.query(img, _DESCRIBE_PROMPT)

    if isinstance(result, dict):
        answer = result.get("answer") or result.get("response") or ""
    else:
        answer = str(result)
    return (answer or "").strip()


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_screenshot() -> None:
    """Delete the captured screenshot immediately. Safe to call always."""
    try:
        if os.path.exists(SCREENSHOT_PATH):
            os.remove(SCREENSHOT_PATH)
    except Exception as e:
        print(f"[screen_awareness] cleanup error: {e}")


# ── Public helper for the voice pipeline ──────────────────────────────────────

def is_orb_mode() -> bool:
    return SCREEN_PREVIEW_MODE == "orb"


def get_active_screenshot_path() -> Optional[str]:
    """Read by ws_server to serve /preview_screen."""
    if os.path.isfile(SCREENSHOT_PATH):
        return SCREENSHOT_PATH
    return None
