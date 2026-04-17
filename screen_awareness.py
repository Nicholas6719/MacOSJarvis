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

# Moondream 2 model cache. Downloaded once on first use (~1.5 GB).
_MOONDREAM_CACHE_DIR = Path.home() / ".cache" / "moondream"
_MOONDREAM_MODEL_FILENAME = "moondream-2b-int8.mf.gz"
_MOONDREAM_MODEL_URL = (
    "https://huggingface.co/vikhyatk/moondream2/resolve/onnx/"
    + _MOONDREAM_MODEL_FILENAME
)

# Cached model instance — loaded once per process, lazily.
_vision_model = None


# ── Model loading ─────────────────────────────────────────────────────────────

def _ensure_model_file() -> str:
    """Ensure the Moondream 2 .mf.gz file exists in the cache, downloading
    it on first use. Returns the absolute path to the model file."""
    _MOONDREAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MOONDREAM_CACHE_DIR / _MOONDREAM_MODEL_FILENAME
    if model_path.is_file() and model_path.stat().st_size > 100_000_000:
        return str(model_path)

    print(f"[screen_awareness] downloading Moondream 2 weights to {model_path} "
          f"(~1.5 GB, first run only)")
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
    """Lazy-load Moondream 2. Cached for the life of the process so repeat
    invocations are instant. Returns the model instance (moondream.VLM)."""
    global _vision_model
    if _vision_model is not None:
        return _vision_model
    import moondream as md  # imported lazily so startup isn't affected
    model_path = _ensure_model_file()
    _vision_model = md.vl(model=model_path)
    return _vision_model


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

_DESCRIBE_PROMPT = (
    "Describe what is on this screen naturally and conversationally, as if "
    "you are speaking it aloud to the user. Be specific but concise — 2 to "
    "4 sentences. Focus on what is most prominent or actively in use."
)


def describe_screen(image_path: str) -> str:
    """Load the image at image_path, query Moondream 2, return the
    description string (stripped)."""
    from PIL import Image
    model = load_vision_model()
    img = Image.open(image_path)
    try:
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
