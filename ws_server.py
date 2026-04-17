"""
Minimal WebSocket state-broadcast server + static HTTP server.

Runs two background threads alongside the voice assistant:
  • WebSocket on port 8765 – broadcasts state changes to the browser
  • HTTP     on port 3000  – serves the pre-built frontend (frontend/dist/)

No Node.js / npm required at runtime; build once with `npm run build`.

Usage inside Python:
    import ws_server
    ws_server.start()          # call once at startup
    ws_server.set_state("listening")
"""

import asyncio
import http.server
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol

PORT      = 8765
HTTP_PORT = 3000

# When running inside the macOS app, JARVIS_DIST_DIR points to the pre-built
# frontend bundled in Contents/Resources/dist/.
# When running locally (start.command), fall back to frontend/dist/ next to this file.
_dist_env = os.environ.get("JARVIS_DIST_DIR")
_DIST_DIR = (
    Path(_dist_env)
    if _dist_env and Path(_dist_env).is_dir()
    else Path(__file__).parent / "frontend" / "dist"
)

_clients: Set[WebSocketServerProtocol] = set()
_loop: asyncio.AbstractEventLoop | None = None
_current_state: str = "idle"
_muted: bool = False
_state_lock = threading.Lock()
_start_lock = threading.Lock()
_servers_started = False

logger = logging.getLogger(__name__)


# ── Internal async helpers ────────────────────────────────────────────────────

async def _handler(ws: WebSocketServerProtocol) -> None:
    _clients.add(ws)
    try:
        # Send current state immediately so the UI is in sync on connect
        await ws.send(json.dumps({"state": _current_state, "muted": _muted}))
        # Keep connection alive; we don't expect messages from the browser
        await ws.wait_closed()
    finally:
        _clients.discard(ws)


async def _broadcast(state: str, muted: bool) -> None:
    if not _clients:
        return
    message = json.dumps({"state": state, "muted": muted})
    await asyncio.gather(
        *[ws.send(message) for ws in list(_clients)],
        return_exceptions=True,
    )


async def _serve() -> None:
    async with websockets.serve(_handler, "127.0.0.1", PORT):
        await asyncio.Future()  # run forever


# ── Public API ────────────────────────────────────────────────────────────────

def set_state(state: str) -> None:
    """Broadcast a new state to all connected browser clients (thread-safe)."""
    global _current_state
    with _state_lock:
        _current_state = state
        muted = _muted
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(state, muted), _loop)


def send_event(event: dict) -> None:
    """Broadcast an arbitrary JSON event to all connected clients (thread-safe)."""
    if _loop is None:
        return

    async def _do() -> None:
        if not _clients:
            return
        msg = json.dumps(event)
        await asyncio.gather(
            *[ws.send(msg) for ws in list(_clients)],
            return_exceptions=True,
        )

    asyncio.run_coroutine_threadsafe(_do(), _loop)


def set_muted(muted: bool) -> None:
    """Update mute state and broadcast it to connected browser clients."""
    global _muted
    with _state_lock:
        _muted = muted
        state = _current_state
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(state, muted), _loop)


def is_muted() -> bool:
    with _state_lock:
        return _muted


def get_status() -> dict[str, str | bool]:
    with _state_lock:
        return {"status": "ok", "state": _current_state, "muted": _muted}


def _cors_end_headers(handler: http.server.BaseHTTPRequestHandler) -> None:
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()


def _handle_api_status(handler: http.server.BaseHTTPRequestHandler) -> None:
    body = json.dumps(get_status()).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    _cors_end_headers(handler)
    handler.wfile.write(body)


def _handle_preview_file(handler: http.server.BaseHTTPRequestHandler) -> None:
    """Serve the file currently staged by file_manager.prepare_preview().
    Only one preview file is active at a time — the path is owned by
    file_manager and advances as the user confirms or rejects files."""
    try:
        import file_manager  # local import to avoid cycle at module load
    except Exception as e:
        handler.send_error(500, f"preview unavailable: {e}")
        return
    path = file_manager.get_active_preview_path()
    if not path or not os.path.isfile(path):
        handler.send_error(404, "no active preview")
        return
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as e:
        handler.send_error(500, f"preview read error: {e}")
        return

    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".pdf":  "application/pdf",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
        ".bmp":  "image/bmp",
        ".heic": "image/heic",
        ".tiff": "image/tiff",
        ".svg":  "image/svg+xml",
        ".txt":  "text/plain; charset=utf-8",
    }.get(ext, "application/octet-stream")

    handler.send_response(200)
    handler.send_header("Content-Type", mime)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Content-Disposition", "inline")
    _cors_end_headers(handler)
    handler.wfile.write(data)


def _handle_api_mute(handler: http.server.BaseHTTPRequestHandler) -> None:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        data = json.loads(raw.decode("utf-8"))
        muted = bool(data["muted"])
    except (json.JSONDecodeError, KeyError, TypeError):
        handler.send_error(400, "Invalid mute payload")
        return

    set_muted(muted)
    body = json.dumps(get_status()).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    _cors_end_headers(handler)
    handler.wfile.write(body)


class _APIOnlyHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler if frontend/dist is missing — still binds :3000 for the macOS app."""

    def log_message(self, *_):
        pass

    def do_GET(self):
        path0 = self.path.split("?", 1)[0]
        if path0 == "/api/status":
            _handle_api_status(self)
            return
        if path0 == "/preview_file":
            _handle_preview_file(self)
            return
        if path0 in ("/", "/index.html"):
            html = (
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/><title>Jarvis</title></head>"
                "<body style=\"font-family:system-ui;padding:2rem\">"
                "<h1>Jarvis</h1>"
                "<p>The web UI is missing. Run this in the project folder:</p>"
                "<pre style=\"background:#eee;padding:1rem\">cd frontend && npm install && npm run build</pre>"
                "<p>Then restart Jarvis.</p></body></html>"
            )
            b = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            _cors_end_headers(self)
            self.wfile.write(b)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path.split("?", 1)[0] == "/api/mute":
            _handle_api_mute(self)
            return
        self.send_error(404)


def _serve_http() -> None:
    """Serve frontend/dist/ on HTTP_PORT; if dist is missing, still listen (API + stub page)."""
    has_dist = _DIST_DIR.is_dir()

    if not has_dist:
        msg = (
            f"[http-server] {_DIST_DIR} missing — starting API-only on port {HTTP_PORT} "
            "(cd frontend && npm run build for the full UI)\n"
        )
        logger.warning(msg.strip())
        print(msg, flush=True)

        with http.server.ThreadingHTTPServer(("", HTTP_PORT), _APIOnlyHandler) as httpd:
            logger.info("[http-server] API-only http://localhost:%d", HTTP_PORT)
            print(f"[http-server] listening on http://127.0.0.1:{HTTP_PORT} (API-only)\n", flush=True)
            httpd.serve_forever()
        return

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(_DIST_DIR), **kwargs)

        def log_message(self, *_):
            pass

        def do_GET(self):
            path0 = self.path.split("?", 1)[0]
            if path0 == "/api/status":
                _handle_api_status(self)
                return
            if path0 == "/preview_file":
                _handle_preview_file(self)
                return
            super().do_GET()

        def do_POST(self):
            if self.path.split("?", 1)[0] == "/api/mute":
                _handle_api_mute(self)
                return
            self.send_error(404)

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    with http.server.ThreadingHTTPServer(("", HTTP_PORT), _QuietHandler) as httpd:
        logger.info("[http-server] serving %s on http://localhost:%d", _DIST_DIR, HTTP_PORT)
        print(f"[http-server] serving { _DIST_DIR } on http://127.0.0.1:{HTTP_PORT}\n", flush=True)
        httpd.serve_forever()


# ── Stale-process cleanup ────────────────────────────────────────────────────
#
# When Jarvis is stopped forcefully from Xcode (Stop button, window closed
# mid-launch, crash, etc.), the Python subprocess can leak — leaving the
# previous Jarvis holding ports 3000 and 8765. On the next launch,
# `socket.bind(...)` fails with `OSError: [Errno 48] Address already in use`
# and both servers die in background threads before the main event loop
# ever sees them, leaving the UI permanently dark.
#
# Setting SO_REUSEADDR doesn't help on macOS for this case — it only covers
# TIME_WAIT state, not a live process holding the port. The correct fix is
# to detect the conflict up front, identify whether it's a stray Jarvis,
# and kill it cleanly before we try to bind.

def _pids_on_port(port: int) -> list[int]:
    """Return PIDs currently bound to `port`, via `lsof -ti`. Empty list on
    any error (lsof missing, permission denied, nothing bound, etc.)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _is_stale_jarvis(pid: int) -> bool:
    """True if `pid` is a Python process running chatbot_speech_to_speech.py
    (or some other Jarvis entry-point script). We only auto-kill processes
    that look like previous Jarvis instances — never random other apps
    that happen to be using the port."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return False
    cmd = (result.stdout or "").strip().lower()
    if not cmd:
        return False
    return (
        "python" in cmd
        and ("chatbot_speech_to_speech" in cmd or "jarvis" in cmd)
    )


def _ensure_port_free(port: int, label: str) -> None:
    """If `port` is held by a stale Jarvis Python process, kill it and wait
    for the OS to release the socket. If held by something else, print a
    loud warning so the user knows exactly which process is the culprit.

    This is safe to run on every startup — it's a no-op when the port is
    already free."""
    pids = _pids_on_port(port)
    if not pids:
        return

    killed_any = False
    for pid in pids:
        if _is_stale_jarvis(pid):
            msg = (
                f"[ws_server] port {port} ({label}) is held by a stale Jarvis "
                f"process PID {pid} — killing it so the new instance can bind"
            )
            print(msg, flush=True)
            logger.warning(msg)
            try:
                os.kill(pid, 9)
                killed_any = True
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"[ws_server] failed to kill PID {pid}: {e}", flush=True)
        else:
            # Something that isn't Jarvis is on the port. Do NOT auto-kill
            # arbitrary processes — just tell the user loudly.
            try:
                ps = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "command="],
                    capture_output=True, text=True, timeout=3,
                )
                cmd = (ps.stdout or "").strip()
            except Exception:
                cmd = "(unknown)"
            msg = (
                f"[ws_server] port {port} ({label}) is held by NON-Jarvis "
                f"process PID {pid}: {cmd!r}. Close it manually and "
                f"restart Jarvis."
            )
            print(msg, flush=True)
            logger.error(msg)

    if killed_any:
        # Give the kernel a moment to release the socket before we bind.
        time.sleep(0.5)
        remaining = _pids_on_port(port)
        if remaining:
            print(
                f"[ws_server] port {port} still held by {remaining} after kill attempt — "
                f"the new server may fail to bind",
                flush=True,
            )


def start() -> None:
    """Start both the WebSocket server and the static HTTP server."""
    global _loop, _servers_started

    with _start_lock:
        if _servers_started:
            return
        _servers_started = True

    # Proactively clear ports before binding — fixes the "UI never comes up"
    # class of failures caused by a leaked Python subprocess from a previous
    # Xcode/JarvisApp run.
    _ensure_port_free(PORT, "WebSocket")
    _ensure_port_free(HTTP_PORT, "HTTP")

    def _run_ws() -> None:
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        try:
            _loop.run_until_complete(_serve())
        except Exception as exc:
            # Print in addition to logging — logging isn't configured so
            # log calls alone would vanish, leaving the user with no idea
            # why the WebSocket is down.
            msg = f"[ws_server] WebSocket server stopped: {exc}"
            print(msg, file=sys.stderr, flush=True)
            logger.warning(msg)

    def _run_http() -> None:
        """Wrap _serve_http with explicit error reporting so a bind failure
        surfaces as a clear message instead of an unhandled thread exception."""
        try:
            _serve_http()
        except Exception as exc:
            msg = (
                f"[http-server] HTTP server stopped: {exc}. "
                f"The UI at http://localhost:{HTTP_PORT} will not be available."
            )
            print(msg, file=sys.stderr, flush=True)
            logger.error(msg)

    threading.Thread(target=_run_ws,   daemon=True, name="ws-server").start()
    threading.Thread(target=_run_http, daemon=True, name="http-server").start()

    logger.info("[ws_server] started on ws://localhost:%d", PORT)
