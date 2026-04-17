"""
Brave Browser control — AppleScript only, no extra dependencies.
─────────────────────────────────────────────────────────────────────────────
Every public function here runs a short AppleScript snippet via `osascript`.
Functions raise RuntimeError on failure so callers can speak a graceful
error instead of crashing the main voice loop.
"""

import subprocess
import urllib.parse
from typing import Optional

_APP_NAME = "Brave Browser"
_DEFAULT_TIMEOUT_S = 8


# ── Known sites — spoken name → canonical URL ────────────────────────────
_KNOWN_SITES = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://www.x.com",
    "x": "https://www.x.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "amazon": "https://www.amazon.com",
    "wikipedia": "https://www.wikipedia.org",
    "github": "https://www.github.com",
    "netflix": "https://www.netflix.com",
    "gmail": "https://mail.google.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "chatgpt": "https://chat.openai.com",
    "claude": "https://claude.ai",
    "linkedin": "https://www.linkedin.com",
    "stack overflow": "https://stackoverflow.com",
    "stackoverflow": "https://stackoverflow.com",
    "hacker news": "https://news.ycombinator.com",
    "twitch": "https://www.twitch.tv",
    "tiktok": "https://www.tiktok.com",
}


# ── Internals ────────────────────────────────────────────────────────────

def _run_osa(script: str, timeout: int = _DEFAULT_TIMEOUT_S) -> str:
    try:
        result = subprocess.run(
            ["osascript"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"osascript timed out after {timeout}s") from e
    except Exception as e:
        raise RuntimeError(f"osascript launch failed: {e}") from e
    if result.returncode != 0:
        err = (result.stderr or "").strip() or "unknown AppleScript error"
        raise RuntimeError(err)
    return (result.stdout or "").strip()


def _is_brave_frontmost() -> bool:
    try:
        out = _run_osa(
            'tell application "System Events" to get name of '
            'first application process whose frontmost is true'
        )
        return out.strip() == _APP_NAME
    except Exception:
        return False


def _ensure_brave_frontmost() -> None:
    """Bring Brave to the foreground so keystroke-based actions land there."""
    if _is_brave_frontmost():
        return
    # `activate` launches if not running.
    _run_osa(f'tell application "{_APP_NAME}" to activate')
    # Short settle so System Events keystrokes target Brave.
    subprocess.run(["sleep", "0.3"], check=False)


def _ensure_brave_running() -> None:
    """Launch Brave if not already running; no focus change."""
    try:
        check = subprocess.run(
            ["pgrep", "-x", _APP_NAME],
            capture_output=True, text=True, timeout=3,
        )
        if check.returncode == 0:
            return
    except Exception:
        pass
    try:
        subprocess.run(
            ["open", "-a", _APP_NAME],
            capture_output=True, timeout=10,
        )
    except Exception as e:
        raise RuntimeError(f"couldn't launch Brave: {e}")


def _escape(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


# ── Public API ───────────────────────────────────────────────────────────

def open_url(url: str) -> None:
    """Open `url` in Brave. Launches Brave if needed. Reuses the active
    window — does NOT force a new tab. The caller decides whether they
    wanted a new tab (use `new_tab(url)` for that)."""
    if not url:
        raise RuntimeError("empty URL")
    _ensure_brave_running()
    script = (
        f'tell application "{_APP_NAME}"\n'
        f'  activate\n'
        f'  if (count of windows) = 0 then\n'
        f'    make new window\n'
        f'  end if\n'
        f'  set URL of active tab of front window to "{_escape(url)}"\n'
        f'end tell\n'
    )
    _run_osa(script)


def new_tab(url: Optional[str] = None) -> None:
    """Open a fresh tab in Brave. If `url` is given, navigate to it."""
    _ensure_brave_running()
    if url:
        script = (
            f'tell application "{_APP_NAME}"\n'
            f'  activate\n'
            f'  if (count of windows) = 0 then\n'
            f'    make new window\n'
            f'    set URL of active tab of front window to "{_escape(url)}"\n'
            f'  else\n'
            f'    tell front window to make new tab with properties {{URL:"{_escape(url)}"}}\n'
            f'  end if\n'
            f'end tell\n'
        )
    else:
        script = (
            f'tell application "{_APP_NAME}"\n'
            f'  activate\n'
            f'  if (count of windows) = 0 then\n'
            f'    make new window\n'
            f'  else\n'
            f'    tell front window to make new tab\n'
            f'  end if\n'
            f'end tell\n'
        )
    _run_osa(script)


def close_tab() -> None:
    """Close the active tab in Brave's front window."""
    _ensure_brave_running()
    script = (
        f'tell application "{_APP_NAME}"\n'
        f'  if (count of windows) > 0 then\n'
        f'    tell front window to close active tab\n'
        f'  end if\n'
        f'end tell\n'
    )
    _run_osa(script)


def go_back() -> None:
    """Trigger the browser Back action in Brave via Cmd+Left."""
    _ensure_brave_frontmost()
    _run_osa(
        'tell application "System Events" to key code 123 using {command down}'
    )


def go_forward() -> None:
    """Trigger the browser Forward action in Brave via Cmd+Right."""
    _ensure_brave_frontmost()
    _run_osa(
        'tell application "System Events" to key code 124 using {command down}'
    )


def scroll_down() -> None:
    """Scroll the active page down by one screen (Page Down keystroke)."""
    _ensure_brave_frontmost()
    # Key code 121 = Page Down on US keyboards.
    _run_osa('tell application "System Events" to key code 121')


def scroll_up() -> None:
    """Scroll the active page up by one screen (Page Up keystroke)."""
    _ensure_brave_frontmost()
    # Key code 116 = Page Up.
    _run_osa('tell application "System Events" to key code 116')


def get_current_url() -> str:
    """Return the URL of Brave's active tab, or an empty string if Brave
    isn't running or has no windows open."""
    try:
        check = subprocess.run(
            ["pgrep", "-x", _APP_NAME],
            capture_output=True, text=True, timeout=3,
        )
        if check.returncode != 0:
            return ""
    except Exception:
        return ""
    script = (
        f'tell application "{_APP_NAME}"\n'
        f'  if (count of windows) = 0 then return ""\n'
        f'  return URL of active tab of front window\n'
        f'end tell\n'
    )
    try:
        return _run_osa(script)
    except RuntimeError:
        return ""


def get_current_title() -> str:
    """Return the title of Brave's active tab, or an empty string if
    Brave isn't running or has no windows open."""
    try:
        check = subprocess.run(
            ["pgrep", "-x", _APP_NAME],
            capture_output=True, text=True, timeout=3,
        )
        if check.returncode != 0:
            return ""
    except Exception:
        return ""
    script = (
        f'tell application "{_APP_NAME}"\n'
        f'  if (count of windows) = 0 then return ""\n'
        f'  return title of active tab of front window\n'
        f'end tell\n'
    )
    try:
        return _run_osa(script)
    except RuntimeError:
        return ""


def resolve_spoken_url(query: str) -> str:
    """Translate a spoken site name / search phrase into a full URL.
    Known sites in _KNOWN_SITES map to their canonical domain. Anything
    already shaped like a URL is returned as-is (with https:// prepended
    if the scheme is missing). Everything else becomes a Google search."""
    if not query:
        return "https://www.google.com"
    q = query.strip().rstrip(".?!,")
    q_lower = q.lower()

    if q_lower in _KNOWN_SITES:
        return _KNOWN_SITES[q_lower]

    # If the user said a bare domain like "example.com" or a full URL,
    # route straight to it instead of a Google search.
    if q_lower.startswith("http://") or q_lower.startswith("https://"):
        return q
    if "." in q_lower and " " not in q_lower and "/" not in q_lower:
        # Looks like example.com or subdomain.example.org
        return f"https://{q_lower}"

    return "https://www.google.com/search?q=" + urllib.parse.quote(q)
