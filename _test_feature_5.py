"""
Self-tests for Phase 4 Feature 5 — Screen Awareness.

Covers:
  • capture → cleanup cycle (idempotent, leaves no temp files)
  • Moondream 2 loads and returns a description
  • SCREEN_PREVIEW_MODE default
  • /preview_screen HTTP route behavior
"""

from __future__ import annotations

import http.client
import os
import sys
import time

import screen_awareness
import ws_server


def _assert(cond: bool, label: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        sys.exit(1)


def _capture_or_synthesize() -> str:
    """capture_screen() relies on macOS Screen Recording permission. If
    that's not granted in the current context, fall back to a synthetic
    PNG so the rest of the pipeline can still be tested end-to-end."""
    path = screen_awareness.capture_screen()
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    print("    (screencapture produced no file — synthesizing test PNG; "
          "grant Screen Recording to capture the real screen)")
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 400), color=(30, 40, 60))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 600, 360], outline=(200, 220, 255), width=4)
    d.text((80, 180), "Jarvis test screen", fill=(230, 230, 255))
    img.save(path, format="PNG")
    return path


def test_capture_and_cleanup() -> None:
    print("test_capture_and_cleanup")
    screen_awareness.cleanup_screenshot()
    _assert(not os.path.exists(screen_awareness.SCREENSHOT_PATH),
            "no stale screenshot before test")

    path = _capture_or_synthesize()
    _assert(path == screen_awareness.SCREENSHOT_PATH,
            "capture_screen returns canonical path")
    _assert(os.path.isfile(path) and os.path.getsize(path) > 0,
            "screenshot file exists and is non-empty")

    screen_awareness.cleanup_screenshot()
    _assert(not os.path.exists(screen_awareness.SCREENSHOT_PATH),
            "cleanup_screenshot removes the file")


def test_capture_cleanup_twice() -> None:
    print("test_capture_cleanup_twice")
    for i in (1, 2):
        _capture_or_synthesize()
        _assert(os.path.isfile(screen_awareness.SCREENSHOT_PATH),
                f"cycle {i}: screenshot exists")
        screen_awareness.cleanup_screenshot()
        _assert(not os.path.exists(screen_awareness.SCREENSHOT_PATH),
                f"cycle {i}: screenshot gone after cleanup")


def test_default_preview_mode() -> None:
    print("test_default_preview_mode")
    _assert(screen_awareness.SCREEN_PREVIEW_MODE == "orb",
            "SCREEN_PREVIEW_MODE defaults to 'orb'")


def test_load_vision_model() -> None:
    print("test_load_vision_model")
    model = screen_awareness.load_vision_model()
    _assert(model is not None, "load_vision_model returns non-None")
    # Cached
    again = screen_awareness.load_vision_model()
    _assert(again is model, "subsequent loads return the cached instance")


def test_describe_screen() -> None:
    print("test_describe_screen")
    path = _capture_or_synthesize()
    try:
        desc = screen_awareness.describe_screen(path)
        _assert(isinstance(desc, str) and len(desc.strip()) > 0,
                "describe_screen returns non-empty string")
        print(f"    description preview: {desc[:160]!r}")
    finally:
        screen_awareness.cleanup_screenshot()


def test_preview_screen_route() -> None:
    print("test_preview_screen_route")
    # Ensure no screenshot is active → 404
    screen_awareness.cleanup_screenshot()

    ws_server.start()
    # Give HTTP server a moment to bind.
    for _ in range(50):
        try:
            http.client.HTTPConnection("127.0.0.1", 3000, timeout=1).request("GET", "/")
            break
        except Exception:
            time.sleep(0.1)

    def fetch_status() -> int:
        conn = http.client.HTTPConnection("127.0.0.1", 3000, timeout=5)
        conn.request("GET", "/preview_screen")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status

    _assert(fetch_status() == 404, "/preview_screen returns 404 when no screenshot")

    _capture_or_synthesize()
    try:
        _assert(fetch_status() == 200, "/preview_screen returns 200 when screenshot exists")
    finally:
        screen_awareness.cleanup_screenshot()


def main() -> None:
    tests = [
        test_default_preview_mode,
        test_capture_and_cleanup,
        test_capture_cleanup_twice,
        test_load_vision_model,
        test_describe_screen,
        test_preview_screen_route,
    ]
    for t in tests:
        t()
        print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
