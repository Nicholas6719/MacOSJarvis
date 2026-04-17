"""
Local Voice Assistant — Jarvis Edition
─────────────────────────────────────────────────────────────────────────────
LLM  : Llama-3.2-3B-Instruct Q4_K_M via llama-cpp-python (Apple Metal GPU)
TTS  : Kokoro-82M ONNX  ·  voice: am_fenrir (male)  ·  ~200 ms/sentence
STT  : faster-whisper 'base' + int8 quantisation + VAD filter
─────────────────────────────────────────────────────────────────────────────
Pipeline   : LLM-stream → TTS-stream → SeamlessPlayer (zero-gap audio)
System cmds: volume, apps, screenshot, timer — executed locally, no LLM
"""

import collections
import datetime
import json
import logging
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time

logger = logging.getLogger("jarvis")
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import webrtcvad
from openwakeword.model import Model as WakeWordModel

import ws_server
import calendar_reminders
import file_manager
import browser_control
import screen_awareness
import tts_engine

# Set of known spoken site names from browser_control, used to decide
# whether a bare "open X" utterance should route to Brave vs the Mac app
# launcher. Lowercased for case-insensitive matching.
_BROWSER_KNOWN_SITE_KEYS = set(browser_control._KNOWN_SITES.keys())


# ── Feature 7: proactive notification state (module-level) ────────────────
# These sets dedupe notifications across the lifetime of the process so
# Jarvis never announces the same event or reminder twice in one day.
# Event keys look like "<title>|<YYYY-MM-DD>"; reminder keys look like
# "<title>|<ISO due timestamp>" so the same title on different days still
# fires, but the same due time never fires twice.
announced_event_notifications: set = set()
announced_reminder_notifications: set = set()

# Poll cadence for the notification monitor thread. 60s matches the spec:
# calendar and reminders both resolve at minute precision.
_NOTIFICATION_POLL_INTERVAL_S = 60
# Calendar events announce when they're between these many minutes away.
# 9-11 is deliberately inclusive of 10 minutes from either side of a poll
# so a once-a-minute poll doesn't miss the 10-minute mark.
_EVENT_NOTIFY_MIN_MINUTES = 9
_EVENT_NOTIFY_MAX_MINUTES = 11


def _parse_applescript_date(raw: str) -> Optional["datetime.datetime"]:
    """Best-effort parse of AppleScript date strings like
    'Friday, April 17, 2026 at 10:00:00 AM' into a Python datetime.
    Returns None on any parse failure — callers should treat None as
    'not announceable yet' rather than crashing the monitor thread."""
    if not raw:
        return None
    raw = raw.strip()
    # Try the common AppleScript format first.
    patterns = [
        "%A, %B %d, %Y at %I:%M:%S %p",
        "%A, %B %d, %Y at %I:%M %p",
        "%a, %b %d, %Y at %I:%M:%S %p",
        "%A %B %d %Y %I:%M:%S %p",
    ]
    for fmt in patterns:
        try:
            return datetime.datetime.strptime(raw, fmt)
        except Exception:
            continue
    # The speech-formatter variant used by calendar_reminders for
    # reminder 'due' strings: "Friday, April 17 at 10 AM" — no year.
    today = datetime.date.today()
    fallback_patterns = [
        ("%A, %B %d at %I:%M %p", False),
        ("%A, %B %d at %I %p", False),
    ]
    for fmt, _ in fallback_patterns:
        try:
            dt = datetime.datetime.strptime(raw, fmt)
            return dt.replace(year=today.year)
        except Exception:
            continue
    return None


def _event_is_due_for_notification(
    event: dict,
    now: Optional["datetime.datetime"] = None,
    announced: Optional[set] = None,
) -> Optional[str]:
    """If the event should be announced right now, return its dedup key.
    Otherwise return None. Exposed at module level so tests can verify
    the window logic without having to run the full monitor thread."""
    if now is None:
        now = datetime.datetime.now()
    if announced is None:
        announced = announced_event_notifications
    title = (event.get("title") or "").strip()
    start_raw = event.get("start") or ""
    if not title or not start_raw:
        return None
    start_dt = _parse_applescript_date(start_raw)
    if start_dt is None:
        return None
    delta = (start_dt - now).total_seconds() / 60.0
    if not (_EVENT_NOTIFY_MIN_MINUTES <= delta <= _EVENT_NOTIFY_MAX_MINUTES):
        return None
    key = f"{title}|{start_dt.date().isoformat()}"
    if key in announced:
        return None
    return key


def _reminder_is_due_for_notification(
    reminder: dict,
    now: Optional["datetime.datetime"] = None,
    announced: Optional[set] = None,
) -> Optional[str]:
    """If the reminder's due time falls inside the current minute and we
    haven't already announced it, return its dedup key. Otherwise None."""
    if now is None:
        now = datetime.datetime.now()
    if announced is None:
        announced = announced_reminder_notifications
    title = (reminder.get("title") or "").strip()
    due_raw = reminder.get("due") or ""
    if not title or not due_raw:
        return None
    due_dt = _parse_applescript_date(due_raw)
    if due_dt is None:
        return None
    # Match at minute precision — same (year, month, day, hour, minute).
    if (due_dt.year, due_dt.month, due_dt.day, due_dt.hour, due_dt.minute) != \
       (now.year, now.month, now.day, now.hour, now.minute):
        return None
    key = f"{title}|{due_dt.isoformat(timespec='minutes')}"
    if key in announced:
        return None
    return key

from memory import MemoryManager
memory = MemoryManager()
memory.seed_initial_facts()


# ── Calendar: feature flags ─────────────────────────────────────────────────
# Set False to suppress the brief "One moment, Sir" ack that Jarvis speaks
# before running a calendar command. The ack gives the user instant feedback
# while the background worker fetches data from Calendar.app / Reminders.app.
# Flip this to False if the ack feels tedious — no other change needed.
SPEAK_CALENDAR_ACK = True

# When True, use the LLM to generate natural-language confirmations after
# successful calendar writes ("Done, Sir. I've scheduled your shift on
# Saturday — let me know if anything needs adjusting."). When False, use a
# fast hard-coded template that saves 4-6 seconds per calendar command at
# the cost of feeling more scripted. The template path is kept as an
# emergency fallback and as a latency escape hatch — flip to False if the
# LLM path ever feels too slow again.
NATURAL_CALENDAR_CONFIRMATIONS = True


class JarvisPauseRequest(Exception):
    """Raised when the user asks Jarvis to go to sleep via voice command."""
    pass


# ── Data directory ────────────────────────────────────────────────────────────
# When running inside the macOS app, the Swift wrapper sets JARVIS_DATA_DIR to
# ~/Library/Application Support/Jarvis/ so models and config survive app updates.
# When running locally (start.command / CLI), falls back to the script directory.
_DATA_DIR = Path(os.environ["JARVIS_DATA_DIR"]) if "JARVIS_DATA_DIR" in os.environ else Path(__file__).parent

# ── Constants ─────────────────────────────────────────────────────────────────
# config.json: user's copy in data dir first, bundled default as fallback
CONFIG_PATH = _DATA_DIR / "config.json"
if not CONFIG_PATH.is_file():
    CONFIG_PATH = Path(__file__).parent / "config.json"
SAMPLE_RATE  = 16_000
TTS_RATE     = 24_000
FRAME_MS     = 30
FRAME_SIZE   = int(SAMPLE_RATE * FRAME_MS / 1_000)

# Adaptive silence: short for quick commands, longer once you've been speaking a while.
# Raised from 500/600 to 700/750 — the tighter values were cutting users off
# mid-sentence on natural pauses ("do you know my … favorite color"). 700ms
# leaves room for mid-phrase breathing without feeling sluggish.
SILENCE_CUTOFF_SHORT_MS  = 700
SILENCE_CUTOFF_LONG_MS   = 750
LONG_SPEECH_THRESHOLD_MS = 2_500   # use long cutoff after 2.5 s of speech

# Hard cap on how long a single utterance recording can run. Prevents a
# stuck VAD state or a runaway mic from filling RAM indefinitely. 15s is
# comfortably longer than any natural voice command.
MAX_RECORDING_MS = 15_000

# Dropped from 4096 (~170 ms at 24 kHz) to 1024 (~42 ms). sounddevice's
# callback fires once per block, so the first-audio latency after we feed()
# the first synth chunk is bounded by this. 1024 is small enough to eliminate
# the perceptible head-of-speech gap and still large enough that macOS Core
# Audio never underruns on M-series silicon.
PLAYER_BLOCKSIZE = 1_024

# ── Wake word ────────────────────────────────────────────────────────────────
WAKE_WORD_MODEL = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.75
CONVERSATION_TIMEOUT = 15.0
WAKE_CHUNK_SIZE = 1280
STARTUP_MODE = os.environ.get("JARVIS_STARTUP_MODE", "wake")
WAKE_MODE_SENTINEL = "__WAKE_MODE__"
NOISE_FLOOR_RMS = 150  # Minimum RMS energy to consider audio as speech

# Memory operations are always silent. Any "updating my memory" / "loaded
# memory" / auto-save acknowledgment has been removed — the user never hears
# about background memory work. Explicit commands ("remember X", "forget X",
# "what do you know about me") still speak naturally.
SPEAK_MEMORY_UPDATE = False

# Keywords that signal an utterance needs the full intent pipeline (calendar,
# file, browser, screen, system command, memory command). Checked as whole
# words. Keep broad — a false positive just costs us the fast path for one
# turn; a false negative routes a command through the LLM instead of the
# matching handler.
_FAST_PATH_BANNED = frozenset({
    "remind", "calendar", "schedule", "remember", "forget",
    "file", "move", "rename", "find", "browser", "open",
    "screen", "timer", "battery", "volume", "brightness",
    "music", "lock", "shutdown", "restart",
})

_FAST_PATH_MAX_WORDS = 8


def is_simple_conversational_turn(text: str) -> bool:
    """Cheap heuristic: would it be wasteful to run the full intent pipeline
    and inject the full memory blob into the LLM for this utterance?

    Returns True when the utterance is short (<= 8 words) AND contains none
    of the command keywords that route to the calendar/file/browser/system
    handlers. These turns skip intent detection, skip memory context
    injection, and go straight to a minimal-context LLM call with just the
    last 3 pairs of conversation history."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    words = re.findall(r"[A-Za-z']+", stripped.lower())
    if not words or len(words) > _FAST_PATH_MAX_WORDS:
        return False
    return not any(w in _FAST_PATH_BANNED for w in words)


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_RE   = re.compile(r"(?<=[,;:])\s+")
MIN_CLAUSE_WORDS = 8


# ── Helpers ───────────────────────────────────────────────────────────────────
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} …", flush=True)

    def _hook(count: int, block: int, total: int) -> None:
        pct = min(100, count * block * 100 // max(total, 1))
        sys.stdout.write(f"\r  {pct:3d}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    print()


def _clean(text: str) -> str:
    """Strip LLM artefacts: <think> tags, markdown symbols, excess newlines.
    Also force real sentence breaks where the LLM used a dash so TTS pauses."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    # Turn " - ", " -- ", " — " into ". " so TTS inserts a real pause.
    text = re.sub(r"\s+[-—–]{1,2}\s+", ". ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── Seamless audio player ──────────────────────────────────────────────────────
class SeamlessPlayer:
    """
    Plays a continuous stream of float32 mono audio fed from a queue.
    Uses sounddevice.OutputStream with a callback so chunks are joined
    at sample level — no gap, click, or silence between sentences.
    """

    def __init__(self, sample_rate: int = TTS_RATE) -> None:
        self._sr      = sample_rate
        self._buf     = np.empty(0, dtype=np.float32)
        self._lock    = threading.Lock()
        self._done    = threading.Event()
        self._feeding = True
        self._stream: Optional[sd.OutputStream] = None

    def start(self) -> None:
        self._done.clear()
        self._feeding = True
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=PLAYER_BLOCKSIZE,
            callback=self._callback,
        )
        self._stream.start()

    def feed(self, audio: np.ndarray) -> None:
        with self._lock:
            self._buf = np.concatenate((self._buf, audio.ravel()))

    def mark_done(self) -> None:
        self._feeding = False

    def wait(self) -> None:
        self._done.wait()
        self._close()

    def stop(self) -> None:
        self._feeding = False
        self._done.set()
        self._close()

    def _close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, outdata: np.ndarray, frames: int, _time, _status) -> None:
        with self._lock:
            have = len(self._buf)
            if have >= frames:
                outdata[:, 0] = self._buf[:frames]
                self._buf = self._buf[frames:]
            elif have > 0:
                outdata[:have, 0] = self._buf
                outdata[have:, 0] = 0.0
                self._buf = np.empty(0, dtype=np.float32)
                if not self._feeding:
                    threading.Timer(0.05, self._done.set).start()
            else:
                outdata[:, 0] = 0.0
                if not self._feeding:
                    self._done.set()


# ── Voice Assistant ────────────────────────────────────────────────────────────
class VoiceAssistant:
    def __init__(self) -> None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            self.cfg: dict = json.load(f)

        # LLM loads first on the main thread — biggest model, wants Metal's
        # GPU allocation before the smaller models compete for it. Moondream
        # warm-up is kicked off from inside _load_llm as a daemon thread,
        # so it overlaps the parallel block below at no extra cost.
        self._load_llm()

        # Parallel load of the three remaining models. On M3 Pro each loads
        # into its own address space with no cross-dependencies, so wall-time
        # is bounded by the slowest of the three rather than their sum.
        timings: dict[str, float] = {}

        def _timed(label: str, fn):
            def _run():
                t0 = time.monotonic()
                try:
                    fn()
                except Exception as e:
                    logger.exception(f"[Startup] {label} load error: {e}")
                finally:
                    timings[label] = time.monotonic() - t0
            return _run

        threads = [
            threading.Thread(target=_timed("STT", self._load_stt),
                             name="load-stt", daemon=True),
            threading.Thread(target=_timed("TTS", self._load_tts),
                             name="load-tts", daemon=True),
            threading.Thread(target=_timed("WakeWord", self._load_wake_word_model),
                             name="load-wake", daemon=True),
        ]
        parallel_t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        parallel_elapsed = time.monotonic() - parallel_t0
        for label, elapsed in timings.items():
            logger.info(f"{label} loaded in {elapsed:.2f}s")
        logger.info(f"Parallel model-load block finished in {parallel_elapsed:.2f}s")

        self.vad = webrtcvad.Vad(3)
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self.history: list[dict] = []
        prior_exchanges = memory.get_recent_exchanges(n=20)
        if prior_exchanges:
            self.history.extend(prior_exchanges)
            print(f"[Memory] Loaded {len(prior_exchanges)} messages from previous session.")
        self._base_prompt: str = self.cfg["llm"].get(
            "prompt_behavior",
            "You are Jarvis, a helpful and concise voice assistant. "
            "Your name is Jarvis. The user's name is Nicholas. "
            "Address the user naturally as 'Sir' or 'Nicholas' when it fits. "
            "If asked for your name, say your name is Jarvis. "
            "Keep answers brief and conversational. No bullet points or markdown.",
        )
        self.system_prompt: str = ""
        self._rebuild_system_prompt()
        self._pending_memory_ack = False
        self._needs_wake_summarization = True  # Fires on first wake-mode iteration
        self._summarizing = False              # Prevent concurrent summarization runs
        self._first_summarization = True       # Only speak the memory-update line at startup
        # llama-cpp-python is NOT thread-safe — two concurrent calls to
        # self._llm.create_chat_completion on the same Llama instance will
        # crash the whole process with SIGSEGV. This lock serializes every
        # call site (chat streaming, calendar extraction/summaries, memory
        # summarization). Any method that touches self._llm must hold it.
        self._llm_lock = threading.Lock()
        self._stop_speak = threading.Event()
        self._tts_speaking = False
        self._cancel_timer = threading.Event()
        self.pending_confirmation: Optional[dict] = None

        # ── Calendar worker state ────────────────────────────────────────
        # _calendar_working is an Event set while a background calendar
        # worker thread is running. While it's set, the main loop skips
        # audio recording — no concurrent TTS, no swallowed commands, no
        # state contention. The worker clears it from its finally block.
        self._calendar_working = threading.Event()
        # _pending_calendar_action holds a pending clarifying question
        # (e.g. "Where are you working, Sir?"). The next user utterance
        # is interpreted as the answer, not a new command. Instance-level
        # (not module-level) so it dies with the VoiceAssistant and
        # cannot leak across restarts.
        self._pending_calendar_action: Optional[dict] = None

        # ── File manager state ──────────────────────────────────────────
        # _pending_file_action holds a pending confirmation (or
        # disambiguation between multiple matches). The next user
        # utterance answers it — not a new command. Structure:
        #   {"action": "move"|"rename"|"describe",
        #    "original_path": "...",
        #    "destination": "...",  # move
        #    "new_name": "...",     # rename
        #    "temp_preview_path": "..." or None,
        #    "candidates": [...remaining paths...],
        #    "waiting_for": "confirmation"|"selection"}
        self._pending_file_action: Optional[dict] = None

        # _last_file_action records the most recent successful file op
        # so follow-up utterances like "actually, move that to Documents"
        # can resolve the pronoun back to the file. Shape:
        #   {"path": "<current filesystem path>",
        #    "original_path": "<pre-action path>",
        #    "action": "move"|"rename",
        #    "timestamp": float}
        # Times out after a few minutes — see _has_recent_file_action.
        self._last_file_action: Optional[dict] = None

        # Wake word state
        self._in_conversation = False
        # Set True immediately after the wake word fires, cleared as soon
        # as the run loop processes the first post-wake utterance. Used
        # by the run loop to discard an accidental transcription of the
        # wake phrase itself (e.g. "Hey Jarvis") as if it were a command.
        self._just_woke = False
        self._conversation_timer: Optional[threading.Timer] = None
        self._return_to_wake = threading.Event()
        self._wake_model: Optional[WakeWordModel] = None
        # Persistent audio stream shared between wake-word detection and
        # command recording. Eliminates the ~300-500ms gap that used to
        # exist when closing the pyaudio wake stream and opening a fresh
        # sounddevice stream for record_audio — that gap was swallowing
        # single-breath commands after the wake word fired.
        self._persistent_stream: Optional[sd.RawInputStream] = None
        # Buffer for accumulating audio samples until we have enough for
        # the wake-word model (WAKE_CHUNK_SIZE = 1280 samples = 80ms).
        # The persistent stream delivers 480-sample (30ms) frames.
        self._wake_buf = np.array([], dtype=np.int16)
        # Rolling buffer of the most recent ~2 seconds of raw audio frames
        # (at 30ms each → 67 frames). Maintained while the wake model is
        # still listening. When wake fires, this buffer contains the user's
        # audio from BEFORE wake detection — crucial for single-breath
        # commands like "Hey Jarvis, tell me a joke" where the wake model
        # needs about a second to confidently fire, and by then the user
        # has already said most of the command. record_audio replays these
        # frames through its VAD/buffer logic so the full utterance is
        # captured, not just what arrived after wake fire.
        self._pre_wake_buf: collections.deque[bytes] = collections.deque(maxlen=67)
        # Stashed at wake-fire time, consumed by the next record_audio call.
        self._pending_pre_wake_audio: bytes = b""

        # ── Notification monitor (Feature 7) ─────────────────────────────
        # A background thread polls Calendar and Reminders every 60s and
        # queues proactive announcements when something is due. The speaker
        # loop drains the queue, waits for any in-flight TTS to finish,
        # respects the paused/muted state, then speaks via the existing
        # TTS pipeline. Queue is unbounded — the announced_* sets prevent
        # duplicate pile-ups.
        self._notification_queue: queue.Queue = queue.Queue()
        self._notification_monitor_thread: Optional[threading.Thread] = None
        self._notification_speaker_thread: Optional[threading.Thread] = None
        self._notification_stop = threading.Event()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_llm(self) -> None:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from llama_cpp import Llama

        c = self.cfg["llm"]
        repo_id  = c["repo_id"]
        filename = c["filename"]
        print(f"[LLM] Loading {repo_id}  ({filename}) …")

        cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if cached and Path(cached).is_file():
            model_path = cached
            print("[LLM] Found in local cache — skipping network.")
        else:
            # try_to_load_from_cache missed it — try local_files_only before going to the network
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, local_files_only=True
                )
                print("[LLM] Found in HF cache (local_files_only) — skipping network.")
            except Exception:
                print("[LLM] Not cached — downloading from HuggingFace …")
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=c.get("n_gpu_layers", -1),
            n_ctx=c.get("n_ctx", 2048),
            n_threads=c.get("n_threads", 6),
            n_batch=c.get("n_batch", 512),
            f16_kv=c.get("f16_kv", True),
            use_mmap=c.get("use_mmap", True),
            use_mlock=c.get("use_mlock", False),
            verbose=False,
        )
        self._llm_cfg = c
        print("[LLM] Ready  (Metal GPU layers active)")

        # Warm Moondream 2 in the background so the first "what's on my
        # screen" call doesn't pay the model-load cost. Fire-and-forget —
        # if it fails, the lazy load in _screen_worker_body still runs.
        screen_awareness.warm_up_in_background()

    def _load_tts(self) -> None:
        from kokoro_onnx import Kokoro

        c = self.cfg["tts"]
        base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

        # Build ordered list of directories to search for model files.
        _search_dirs = [
            _DATA_DIR,                          # ~/Library/Application Support/Jarvis/
            Path(__file__).parent,              # bundle Resources or project root (local dev)
        ]
        _bundle_dir = os.environ.get("JARVIS_BUNDLE_DIR")
        if _bundle_dir:
            _search_dirs.append(Path(_bundle_dir))  # folder containing the .app

        def _resolve(key: str, default: str) -> Path:
            """Search known dirs for the model file; fall back to _DATA_DIR as download target."""
            name = Path(c.get(key, default)).name
            for d in _search_dirs:
                p = d / name
                if p.is_file():
                    return p
            return _DATA_DIR / name   # download destination

        model_p  = _resolve("model_file",  "kokoro-v1.0.onnx")
        voices_p = _resolve("voices_file", "voices-v1.0.bin")
        if not model_p.is_file():
            _download(f"{base}/{model_p.name}", model_p)
        if not voices_p.is_file():
            _download(f"{base}/{voices_p.name}", voices_p)
        print(f"[TTS] Loading Kokoro ONNX  (voice: {c['voice']}) …")
        self._kokoro = Kokoro(str(model_p), str(voices_p))
        self._voice: str  = c["voice"]
        self._speed: float = float(c.get("speed", 1.0))
        print("[TTS] Ready")

        tts_engine.configure(self.cfg)
        tts_engine.register_kokoro(self._kokoro, self._voice, self._speed)
        active = tts_engine.get_active_backend()
        if active == "fishaudio":
            logger.info(
                f"TTS backend: Fish Audio (voice {c.get('fishaudio_voice_id')})"
            )
        else:
            logger.info("TTS backend: Kokoro (local)")

    def _load_stt(self) -> None:
        from faster_whisper import WhisperModel

        c    = self.cfg["stt"]
        size = c.get("model_size", "base")
        print(f"[STT] Loading faster-whisper '{size}' …")
        try:
            # Always try local cache first — avoids HuggingFace network call when offline.
            self._stt = WhisperModel(
                size, device="cpu", compute_type="int8", local_files_only=True
            )
        except Exception:
            # Model not in local cache yet — download it (requires internet).
            print(f"[STT] Model not cached — downloading faster-whisper '{size}' …")
            self._stt = WhisperModel(
                size, device="cpu", compute_type="int8", local_files_only=False
            )
        self._lang: str = c.get("language", "en")
        print("[STT] Ready")

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _drain_q(self) -> None:
        """Discard stale frames left in the audio queue."""
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    def record_audio(self) -> bytes:
        """
        Record a full user utterance.
        Uses adaptive silence: short commands cut off at 600 ms,
        longer speech (> 2.5 s) gets 950 ms — so you can finish long sentences.

        Reads from the shared self._audio_q which is fed by the persistent
        sd.RawInputStream started in _init_wake_word. There's no stream
        open/close here — the stream has been running continuously since
        startup, which means when the wake word fires, any audio the user
        is speaking right now is ALREADY in the queue and available
        immediately. This is what makes single-breath commands work.
        """
        while ws_server.is_muted():
            ws_server.set_state("idle")
            time.sleep(0.1)

        # Drain + brief pause to let residual TTS audio clear from the
        # mic buffer. CRITICAL: skip this on the first capture after
        # wake-word fires — at that moment there's no TTS to echo, and
        # any frames currently in the queue are the user's command
        # (they just finished saying the wake phrase and are continuing
        # with the actual request). Throwing them away is why every
        # previous attempt missed single-breath commands.
        if not self._just_woke:
            time.sleep(0.2)
            self._drain_q()

        ws_server.set_state("listening")
        print("🎤  Listening …", flush=True)
        buf        = b""
        silence_ms = 0
        speech_ms  = 0
        total_ms   = 0   # safety cap — bounded by MAX_RECORDING_MS
        speaking   = False
        # Rolling buffer of recent frames so the first syllable isn't clipped.
        pre_roll: collections.deque[bytes] = collections.deque(maxlen=5)

        # Priming frames from the pre-wake rolling buffer. When the wake
        # word fires, _listen_for_wake_word stashes the last ~2 seconds of
        # audio in self._pending_pre_wake_audio. We replay them through
        # the same VAD/buffer logic so the full utterance gets captured —
        # this is the ONLY way single-breath commands like "Hey Jarvis,
        # tell me a joke" work, because by the time the wake model fires
        # the user has already spoken most of the command.
        priming_frames: list = []
        if self._pending_pre_wake_audio:
            raw = self._pending_pre_wake_audio
            self._pending_pre_wake_audio = b""
            bytes_per_frame = FRAME_SIZE * 2  # int16 = 2 bytes per sample
            for i in range(0, len(raw) - bytes_per_frame + 1, bytes_per_frame):
                priming_frames.append(raw[i:i + bytes_per_frame])
        processing_priming = bool(priming_frames)

        # No `with sd.RawInputStream(...)` — the persistent stream is
        # already running. Read directly from the shared queue.
        while True:
            # Pull the next frame: priming first, then the live queue.
            if priming_frames:
                frame = priming_frames.pop(0)
            else:
                if processing_priming:
                    # Transition from priming to live audio. Don't reset
                    # silence_ms — if the priming buffer ended with silence
                    # after a short command, we want that silence to count
                    # toward the cutoff.
                    processing_priming = False
                if ws_server.is_muted():
                    return b""
                if self._tts_speaking:
                    # Discard all incoming audio while Jarvis is speaking
                    try:
                        self._audio_q.get_nowait()
                    except queue.Empty:
                        time.sleep(0.01)
                    speaking = False
                    buf = b""
                    continue
                try:
                    frame = self._audio_q.get(timeout=1.0)
                except queue.Empty:
                    continue

            if self.vad.is_speech(frame, SAMPLE_RATE):
                if not speaking:
                    # Prepend buffered frames so the start of speech is preserved
                    buf = b"".join(pre_roll)
                buf       += frame
                silence_ms = 0
                speaking   = True
                speech_ms += FRAME_MS
                total_ms  += FRAME_MS
            elif speaking:
                buf        += frame
                silence_ms += FRAME_MS
                total_ms   += FRAME_MS
                # Don't break out while still processing priming frames —
                # the "silence" might just be a pause between the wake
                # phrase and the command in the original utterance. Only
                # check the cutoff once we're reading live audio.
                if not processing_priming:
                    cutoff = (
                        SILENCE_CUTOFF_LONG_MS
                        if speech_ms >= LONG_SPEECH_THRESHOLD_MS
                        else SILENCE_CUTOFF_SHORT_MS
                    )
                    if silence_ms > cutoff:
                        break
            else:
                # Not speaking yet — keep recent frames for pre-roll
                pre_roll.append(frame)

            # Hard cap: never record longer than MAX_RECORDING_MS once we've
            # started capturing speech. Prevents pathological silence-free
            # audio (background hum, music) from growing the buffer forever.
            if speaking and total_ms >= MAX_RECORDING_MS:
                break
        # Check RMS energy — ignore if it's just background noise
        audio_np = np.frombuffer(buf, dtype=np.int16)
        if len(audio_np) == 0:
            return b""
        rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
        if rms < NOISE_FLOOR_RMS:
            return b""
        return buf

    # ── STT ───────────────────────────────────────────────────────────────────

    def transcribe(self, audio_bytes: bytes) -> str:
        audio = np.frombuffer(audio_bytes, dtype="int16").astype("float32") / 32_768.0
        segments, _ = self._stt.transcribe(
            audio,
            language=self._lang,
            beam_size=5,
            temperature=0,                      # deterministic, no random sampling
            condition_on_previous_text=False,   # no hallucination from prior context
            vad_filter=True,
            vad_parameters={"threshold": 0.45, "min_silence_duration_ms": 500},
            initial_prompt=(
                "Jarvis, hey Jarvis, yes, no, remind me, remember, calendar, "
                "schedule, tomorrow, today, desktop, downloads, documents, "
                "move, rename, find, open, close, browser, timer, screenshot, "
                "what's on my screen, battery, volume, brightness, music"
            ),
        )
        return " ".join(s.text.strip() for s in segments).strip()

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _synthesise(self, text: str) -> np.ndarray:
        return tts_engine.synthesize(text)

    # Cutoff (in words) below which speak_direct synthesizes the whole
    # phrase in one shot instead of firing up the streaming pipeline.
    # The pipeline has a small thread-setup cost (~10-20 ms) that isn't
    # worth paying for "Opening Brave, Sir." but absolutely is worth
    # paying for a 3-sentence calendar summary.
    _SPEAK_DIRECT_PIPELINE_WORD_FLOOR = 12

    @staticmethod
    def _split_for_tts(text: str) -> list[str]:
        """Break a spoken phrase into sentence-sized chunks for the
        streaming TTS pipeline. Uses the same sentence regex as the LLM
        streamer so the two paths produce the same cadence on the same
        text. Newlines are treated as sentence boundaries too — long
        bulleted calendar summaries occasionally arrive with embedded
        \\n separators and we want each line spoken separately."""
        if not text:
            return []
        # Normalise newlines to a sentence boundary so SENTENCE_RE picks
        # them up alongside . / ? / !
        t = re.sub(r"\n+", ". ", text).strip()
        if not t:
            return []
        parts = [p.strip() for p in SENTENCE_RE.split(t)]
        return [p for p in parts if p]

    def speak_direct(self, text: str) -> None:
        """Speak text immediately via TTS — no LLM involved.

        Streaming-pipelined: the input is split into sentences and each
        one is synthesized + queued for playback while the next is still
        generating. The first syllable comes out as soon as the first
        sentence's Kokoro pass returns, which on a 3-sentence calendar
        summary shaves 300-500 ms off the perceived latency compared to
        the old "synthesize everything, then play" path.

        Short one-liners (< 12 words and a single sentence) bypass the
        pipeline — the thread overhead costs more than they save.

        Blocks until all audio finishes playing so existing callers see
        no behavioural difference."""
        if not text or not text.strip():
            return

        sentences = self._split_for_tts(text)
        is_one_liner = (
            len(sentences) <= 1
            and len(text.split()) < self._SPEAK_DIRECT_PIPELINE_WORD_FLOOR
        )

        self._tts_speaking = True
        self._cancel_conversation_timer()  # Pause timer while speaking
        ws_server.set_state("speaking")
        try:
            if is_one_liner:
                # Fast path — single synth call, no thread setup.
                wav = self._synthesise(text)
                player = SeamlessPlayer(sample_rate=TTS_RATE)
                player.start()
                player.feed(wav)
                player.mark_done()
                player.wait()
            else:
                # Streaming path — synthesize sentence N while sentence
                # N-1 plays. Uses the same SeamlessPlayer the LLM-driven
                # handle_turn path uses, so there's zero gap between
                # sentences and the first audio comes out as soon as
                # Kokoro finishes the first sentence.
                player = SeamlessPlayer(sample_rate=TTS_RATE)
                player.start()
                sentence_q: queue.Queue[Optional[str]] = queue.Queue()
                for s in sentences:
                    sentence_q.put(s)
                sentence_q.put(None)  # sentinel — end of input

                def _tts_worker() -> None:
                    while True:
                        s = sentence_q.get()
                        if s is None:
                            break
                        try:
                            wav = self._synthesise(s)
                        except Exception as e:
                            logger.error(f"speak_direct synth error on {s!r}: {e}")
                            continue
                        player.feed(wav)
                    player.mark_done()

                tts_t = threading.Thread(
                    target=_tts_worker, daemon=True, name="speak-direct-tts",
                )
                tts_t.start()
                # wait() blocks until the player drains. tts_t will have
                # fed everything + called mark_done by then.
                player.wait()
                tts_t.join(timeout=1.0)
        finally:
            time.sleep(0.3)
            self._drain_q()
            self._tts_speaking = False
            ws_server.set_state("idle")
            # Resume timer after speaking if still in conversation mode
            if self._in_conversation:
                self._start_conversation_timer()

    def stop_speaking(self) -> None:
        self._stop_speak.set()

    # ── Notification monitor (Feature 7) ─────────────────────────────────

    def start_notification_monitor(self) -> None:
        """Launch the background poll + speaker threads. Idempotent — if
        the monitor is already running, this is a no-op."""
        if (
            self._notification_monitor_thread is not None
            and self._notification_monitor_thread.is_alive()
        ):
            return
        self._notification_stop.clear()
        self._notification_monitor_thread = threading.Thread(
            target=self._notification_monitor_loop,
            daemon=True,
            name="notification-monitor",
        )
        self._notification_speaker_thread = threading.Thread(
            target=self._notification_speaker_loop,
            daemon=True,
            name="notification-speaker",
        )
        self._notification_monitor_thread.start()
        self._notification_speaker_thread.start()
        print("[Notify] monitor started", flush=True)

    def stop_notification_monitor(self) -> None:
        self._notification_stop.set()

    def _notification_monitor_loop(self) -> None:
        """Poll Calendar and Reminders on a fixed cadence. Any hits are
        pushed onto _notification_queue for the speaker thread to drain."""
        # 90-second startup delay. The monitor thread is registered and
        # running, but we do NOT touch Calendar/Reminders AppleScript
        # during startup — that was waking Calendar.app and Reminders.app
        # unnecessarily every launch. After this delay, polling resumes
        # at the normal 60-second cadence.
        for _ in range(900):
            if self._notification_stop.is_set():
                return
            time.sleep(0.1)

        while not self._notification_stop.is_set():
            try:
                self._check_calendar_notifications()
            except Exception as e:
                logger.error(f"[Notify] _check_calendar_notifications error: {e}")
            try:
                self._check_reminder_notifications()
            except Exception as e:
                logger.error(f"[Notify] _check_reminder_notifications error: {e}")
            # Sleep in small chunks so stop requests are responsive.
            for _ in range(_NOTIFICATION_POLL_INTERVAL_S * 10):
                if self._notification_stop.is_set():
                    return
                time.sleep(0.1)

    def _check_calendar_notifications(self) -> None:
        try:
            events = calendar_reminders.get_today_events()
        except Exception as e:
            logger.error(f"[Notify] get_today_events failed: {e}")
            return
        now = datetime.datetime.now()
        for ev in events:
            key = _event_is_due_for_notification(ev, now=now)
            if key is None:
                continue
            announced_event_notifications.add(key)
            title = ev.get("title", "your next event")
            msg = f"Heads up — your {title} starts in 10 minutes, Sir."
            self._notification_queue.put(msg)
            print(f"[Notify] queued event notification: {title}", flush=True)

    def _check_reminder_notifications(self) -> None:
        try:
            reminders = calendar_reminders.get_all_reminders()
        except Exception as e:
            logger.error(f"[Notify] get_all_reminders failed: {e}")
            return
        now = datetime.datetime.now()
        for r in reminders:
            key = _reminder_is_due_for_notification(r, now=now)
            if key is None:
                continue
            announced_reminder_notifications.add(key)
            title = r.get("title", "a reminder")
            msg = f"Just a heads up — you wanted to remember to {title}, Sir."
            self._notification_queue.put(msg)
            print(f"[Notify] queued reminder notification: {title}", flush=True)

    def _notification_speaker_loop(self) -> None:
        """Drain the queue and speak each message. Waits for any current
        TTS to finish and respects the paused/muted state — if Jarvis is
        paused, the message stays queued until resume."""
        while not self._notification_stop.is_set():
            try:
                msg = self._notification_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # Respect mute/paused — re-queue at the head by looping until
            # the user unmutes. Keep the message; we don't drop it.
            while ws_server.is_muted() and not self._notification_stop.is_set():
                time.sleep(0.5)
            # Never interrupt an in-flight TTS response. Wait until the
            # speaking flag clears before taking the floor.
            while self._tts_speaking and not self._notification_stop.is_set():
                time.sleep(0.1)
            if self._notification_stop.is_set():
                return
            # Also wait out any active calendar worker so we don't step on
            # a "One moment, Sir" or a calendar read that's mid-flight.
            while self._calendar_working.is_set() and not self._notification_stop.is_set():
                time.sleep(0.2)
            try:
                # speak_direct uses the existing TTS pipeline and is
                # already safe to call from a background thread. Serialize
                # on _llm_lock briefly so we don't collide with an in-
                # flight LLM-driven TTS chunk generation.
                with self._llm_lock:
                    pass  # lock is for LLM callers; acquiring + releasing
                # is our way of waiting out any LLM generation that's in
                # progress. speak_direct itself does not need the lock.
                self.speak_direct(msg)
                print(f"[Notify] spoke: {msg}", flush=True)
            except Exception as e:
                logger.error(f"[Notify] _notification_speaker_loop speak error: {e}")

    # ── System commands ───────────────────────────────────────────────────────

    # Spoken folder names → filesystem paths
    _FINDER_FOLDERS: dict[str, str] = {
        "downloads":    "~/Downloads",
        "download":     "~/Downloads",
        "desktop":      "~/Desktop",
        "documents":    "~/Documents",
        "document":     "~/Documents",
        "home":         "~",
        "pictures":     "~/Pictures",
        "picture":      "~/Pictures",
        "movies":       "~/Movies",
        "music":        "~/Music",
        "applications": "/Applications",
    }

    # Common spoken names → exact macOS .app names
    _APP_ALIASES: dict[str, str] = {
        "brave":                "Brave Browser",
        "brave browser":        "Brave Browser",
        "safari":               "Safari",
        "chrome":               "Google Chrome",
        "google chrome":        "Google Chrome",
        "firefox":              "Firefox",
        "facetime":             "FaceTime",
        "face time":            "FaceTime",
        "app store":            "App Store",
        "appstore":             "App Store",
        "reminders":            "Reminders",
        "reminder":             "Reminders",
        "system settings":      "System Settings",
        "settings":             "System Settings",
        "system preferences":   "System Preferences",
        "activity monitor":     "Activity Monitor",
        "finder":               "Finder",
        "mail":                 "Mail",
        "messages":             "Messages",
        "imessage":             "Messages",
        "photos":               "Photos",
        "photo":                "Photos",
        "calendar":             "Calendar",
        "notes":                "Notes",
        "note":                 "Notes",
        "terminal":             "Terminal",
        "spotify":              "Spotify",
        "claude":               "Claude",
        "clawed":               "Claude",
        "chatgpt":              "ChatGPT",
        "chat gpt":             "ChatGPT",
        "music":                "Music",
        "apple music":          "Music",
        "podcasts":             "Podcasts",
        "maps":                 "Maps",
        "preview":              "Preview",
        "calculator":           "Calculator",
        "zoom":                 "Zoom",
        "slack":                "Slack",
        "discord":              "Discord",
        "notion":               "Notion",
        "figma":                "Figma",
        "xcode":                "Xcode",
        "vs code":              "Visual Studio Code",
        "vscode":               "Visual Studio Code",
        "visual studio code":   "Visual Studio Code",
        "cursor":               "Cursor",
        "arc":                  "Arc",
        "whatsapp":             "WhatsApp",
        "telegram":             "Telegram",
    }

    def _resolve_app_name(self, raw: str) -> str:
        """Clean up transcription noise and map spoken names to exact app names.

        Priority: alias dict → title-case guess → fuzzy search in /Applications.
        """
        clean = re.sub(r"[^\w\s]", "", raw).strip().lower()
        clean = re.sub(r"^(?:the|a|an)\s+", "", clean)
        if clean in self._APP_ALIASES:
            return self._APP_ALIASES[clean]
        # Try title-case first (works for simple names like "Preview")
        title = clean.title()
        check = subprocess.run(
            ["mdfind", f"kMDItemKind == 'Application' && kMDItemDisplayName == '{title}'"],
            capture_output=True, text=True,
        )
        if check.stdout.strip():
            return title
        # Fuzzy search /Applications for any .app whose name contains the words
        try:
            apps = os.listdir("/Applications")
            for app in apps:
                if app.endswith(".app"):
                    app_lower = app[:-4].lower()
                    if clean in app_lower or all(w in app_lower for w in clean.split()):
                        return app[:-4]
        except OSError:
            pass
        return title

    # Words that signal the captured text is NOT an app name
    _NON_APP_FIRST_WORDS = {
        "up", "down", "in", "out", "on", "off", "to", "with", "about",
        "for", "new", "my", "your", "this", "that", "some", "all", "more",
        "less", "much", "another", "any", "every", "it", "him", "her",
        "them", "us", "me", "both", "few", "many",
    }

    def _is_app_command(self, raw: str) -> bool:
        """
        Return True only if the captured text genuinely looks like an app name.
        Guards against false positives like 'open up about...' or 'close enough'.
        """
        clean = re.sub(r"[^\w\s]", "", raw).strip().lower()
        clean = re.sub(r"^(?:the|a|an)\s+", "", clean)
        if clean in self._APP_ALIASES:
            return True
        words = clean.split()
        return (
            1 <= len(words) <= 3
            and bool(words)
            and words[0] not in self._NON_APP_FIRST_WORDS
        )

    def _applescript(self, script: str) -> str:
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"_applescript timeout after 10s: {e}")
            return ""
        except Exception as e:
            logger.error(f"_applescript subprocess error: {e}")
            return ""
        return result.stdout.strip()

    # ── Browser control ──────────────────────────────────────────────────────

    _BROWSER_WHERE_PATTERNS = (
        r"\bwhat\s+page\s+am\s+i\s+on\b",
        r"\bwhat\s+site\s+is\s+this\b",
        r"\bwhere\s+am\s+i\b",
        r"\bwhat(?:'?s|\s+is)\s+open\s+in\s+(?:my\s+)?browser\b",
        r"\bwhat(?:'?s|\s+is)\s+(?:my\s+)?browser\s+(?:on|showing)\b",
    )

    def _handle_browser_command(self, t: str) -> Optional[str]:
        """If `t` (already lowercased) is a browser command, execute it and
        return the spoken response. Otherwise return None.

        Ambiguous commands like 'go back' and 'scroll' only fire when Brave
        is the frontmost app, so non-browser uses of the same phrase
        (music 'go back', generic scrolling elsewhere) still work.
        """
        # ── Where am I / what page is this ────────────────────────────────
        for pat in self._BROWSER_WHERE_PATTERNS:
            if re.search(pat, t):
                try:
                    title = browser_control.get_current_title()
                    url = browser_control.get_current_url()
                except Exception as e:
                    return f"I couldn't read Brave's current page, Sir. ({e})"
                if not title and not url:
                    return "Brave doesn't seem to have a page open right now, Sir."
                if title and url:
                    return f"You're on {title}. The URL is {url}, Sir."
                return f"You're on {title or url}, Sir."

        # ── New tab ───────────────────────────────────────────────────────
        if re.search(r"\b(?:open\s+(?:a\s+)?new\s+tab|new\s+tab)\b", t):
            try:
                browser_control.new_tab()
                return "Opened a new tab, Sir."
            except Exception as e:
                return f"I couldn't open a new tab, Sir. ({e})"

        # ── Close tab ─────────────────────────────────────────────────────
        if re.search(r"\bclose\s+(?:this|the|current)\s+tab\b", t):
            try:
                browser_control.close_tab()
                return "Closed the tab, Sir."
            except Exception as e:
                return f"I couldn't close the tab, Sir. ({e})"

        # ── Navigate / open / search ──────────────────────────────────────
        # "search for <query>" always means Google search in the browser.
        m_search = re.search(r"\bsearch\s+(?:for\s+|the\s+web\s+for\s+)(.+)$", t)
        if m_search:
            query = m_search.group(1).strip().rstrip(".?!,")
            if query:
                url = browser_control.resolve_spoken_url(query)
                try:
                    browser_control.open_url(url)
                    return f"Searching the web for {query}, Sir."
                except Exception as e:
                    return f"I couldn't open Brave, Sir. ({e})"

        # "go to X" / "navigate to X" / "take me to X" / "pull up X"
        m_goto = re.search(
            r"\b(?:go\s+to|navigate\s+to|take\s+me\s+to|pull\s+up|bring\s+up)\s+(.+)$",
            t,
        )
        if m_goto:
            site = m_goto.group(1).strip().rstrip(".?!,")
            if site:
                url = browser_control.resolve_spoken_url(site)
                try:
                    browser_control.open_url(url)
                    return self._browser_open_confirmation(site, url)
                except Exception as e:
                    return f"I couldn't open Brave, Sir. ({e})"

        # "open <site>" — but only if <site> is a known site name or looks
        # like a URL/domain. Otherwise let the app-launch path handle it.
        m_open = re.search(r"^open\s+(.+?)\s*$", t)
        if m_open:
            target = m_open.group(1).strip().rstrip(".?!,")
            t_lower = target.lower()
            looks_like_url = (
                t_lower in _BROWSER_KNOWN_SITE_KEYS
                or t_lower.startswith("http://")
                or t_lower.startswith("https://")
                or (
                    "." in t_lower
                    and " " not in t_lower
                    and "/" not in t_lower
                )
            )
            if looks_like_url:
                url = browser_control.resolve_spoken_url(target)
                try:
                    browser_control.open_url(url)
                    return self._browser_open_confirmation(target, url)
                except Exception as e:
                    return f"I couldn't open Brave, Sir. ({e})"

        return None

    def _browser_open_confirmation(self, spoken_site: str, url: str) -> str:
        """Pick a natural confirmation for an open/navigate command."""
        if "google.com/search" in url:
            return f"Searching Google for {spoken_site}, Sir."
        nice = spoken_site.strip().title()
        return f"Opening {nice} in Brave, Sir."

    # ── Clipboard helpers ─────────────────────────────────────────────────────

    # Phrases that signal "process my clipboard with the LLM"
    _CLIPBOARD_TRIGGERS = (
        "improve this", "fix this", "rewrite this", "correct this",
        "proofread this", "summarize this", "summarize the text",
        "translate this", "make this shorter", "make this longer",
        "make this more formal", "make this casual", "simplify this",
        "explain this",
    )

    def _try_augment_clipboard(self, text: str) -> tuple[str, bool]:
        """
        If the utterance is a clipboard command, read the clipboard and
        append its content to the prompt so the LLM can act on it.
        Returns (augmented_text, is_clipboard_command).
        """
        t = text.lower()
        if not any(trigger in t for trigger in self._CLIPBOARD_TRIGGERS):
            return text, False
        clipboard = subprocess.run(
            ["pbpaste"], capture_output=True, text=True
        ).stdout.strip()
        if not clipboard:
            return text + "\n(Note: clipboard is empty)", False
        return f"{text}\n\nClipboard content:\n{clipboard}", True

    def _copy_to_clipboard(self, text: str) -> None:
        subprocess.run(["pbcopy"], input=text.encode(), check=False)

    # Spoken word numbers → digits (used by timer & reminder parsing)
    _WORD_NUMBERS: dict[str, int] = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "fifteen": 15, "twenty": 20, "thirty": 30,
        "forty five": 45, "forty-five": 45, "sixty": 60,
    }

    def _parse_number(self, s: str) -> Optional[int]:
        """Convert a string that is either digits or a spoken word number."""
        s = s.strip().lower()
        if s.isdigit():
            return int(s)
        return self._WORD_NUMBERS.get(s)

    def _timer_callback(self, seconds: int, label: str) -> None:
        for _ in range(seconds * 10):
            if self._cancel_timer.is_set():
                self._cancel_timer.clear()
                return
            time.sleep(0.1)
        msg = f"Sir, your {label} timer is up."
        print(f"\n⏰  {msg}", flush=True)
        subprocess.run(
            ["osascript", "-e",
             f'display notification "Timer complete!" with title "Jarvis" subtitle "{label}"'],
            check=False, timeout=10,
        )
        # Drain the audio queue so the mic doesn't pick up the notification sound
        time.sleep(0.3)
        self._drain_q()
        self.speak_direct(msg)
        # Drain again after speaking so Jarvis doesn't hear itself
        time.sleep(0.5)
        self._drain_q()

    # ── Wake word ─────────────────────────────────────────────────────────────

    def _load_wake_word_model(self) -> None:
        """Load just the OpenWakeWord model. Safe to call from a background
        thread — no audio device is touched here."""
        print("[WAKE] Loading hey_jarvis model...", flush=True)
        self._wake_model = WakeWordModel(
            wakeword_models=[WAKE_WORD_MODEL],
            inference_framework='onnx'
        )
        print("[WAKE] Model loaded.", flush=True)

    def _init_wake_word(self) -> None:
        """Start the persistent audio stream and mark wake-word listening live.
        The model itself is loaded earlier in __init__'s parallel block — this
        only has to bring the mic stream up once everything else is ready."""
        if self._wake_model is None:
            # Fallback in case the parallel load block was skipped — keeps
            # the method self-contained and safe to call on its own.
            self._load_wake_word_model()
        self._start_persistent_stream()
        print("[WAKE] Ready — listening for 'Hey Jarvis'", flush=True)

    def _start_persistent_stream(self) -> None:
        """Create and start the always-on sd.RawInputStream. Idempotent:
        calling it when the stream is already running is a no-op."""
        if self._persistent_stream is not None:
            return

        def _cb(indata, frames, time_info, status):
            # Push raw bytes to the shared queue. Both the wake-word loop
            # and record_audio consume from this queue.
            self._audio_q.put(bytes(indata))

        self._persistent_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            dtype="int16",
            channels=1,
            callback=_cb,
        )
        self._persistent_stream.start()

    def _stop_persistent_stream(self) -> None:
        if self._persistent_stream is None:
            return
        try:
            self._persistent_stream.stop()
            self._persistent_stream.close()
        except Exception:
            pass
        self._persistent_stream = None

    def _listen_for_wake_word(self) -> bool:
        """Consume from the shared queue, accumulate samples, feed openwakeword.
        Returns True the instant the model fires, and stashes the last ~2
        seconds of audio in self._pending_pre_wake_audio so record_audio can
        replay it — that's how we capture single-breath commands where the
        wake model fired halfway through the user's sentence."""
        # Pull one frame (30ms of audio). Short timeout so we don't block
        # the main loop if the stream stalls for any reason.
        try:
            frame = self._audio_q.get(timeout=0.1)
        except queue.Empty:
            return False

        # Keep a rolling copy in the pre-wake buffer (2-second ring). deque
        # with maxlen automatically drops the oldest frame when full.
        self._pre_wake_buf.append(frame)

        samples = np.frombuffer(frame, dtype=np.int16)
        self._wake_buf = np.concatenate((self._wake_buf, samples))

        # Not enough for a prediction yet — wait for more frames.
        if len(self._wake_buf) < WAKE_CHUNK_SIZE:
            return False

        # Run prediction on the oldest 1280 samples, keep the remainder
        # for next time (streaming-style).
        chunk = self._wake_buf[:WAKE_CHUNK_SIZE]
        self._wake_buf = self._wake_buf[WAKE_CHUNK_SIZE:]

        try:
            prediction = self._wake_model.predict(chunk)
        except Exception as e:
            print(f"[WAKE] predict error: {e}", flush=True)
            return False

        score = prediction.get(WAKE_WORD_MODEL, 0)
        if score >= WAKE_WORD_THRESHOLD:
            try:
                self._wake_model.reset()
            except Exception:
                pass
            # Clear the wake sample accumulator.
            self._wake_buf = np.array([], dtype=np.int16)
            # Stash the pre-wake rolling buffer for record_audio to replay.
            # This is the crucial step for single-breath command capture.
            self._pending_pre_wake_audio = b"".join(self._pre_wake_buf)
            self._pre_wake_buf.clear()
            return True
        return False

    def _cleanup_wake_word(self) -> None:
        """Shut down the persistent stream on session exit."""
        self._stop_persistent_stream()

    # ── Conversation timer ───────────────────────────────────────────────────

    def _start_conversation_timer(self) -> None:
        self._cancel_conversation_timer()
        self._conversation_timer = threading.Timer(
            CONVERSATION_TIMEOUT,
            self._end_conversation
        )
        self._conversation_timer.daemon = True
        self._conversation_timer.start()

    def _cancel_conversation_timer(self) -> None:
        if self._conversation_timer is not None:
            self._conversation_timer.cancel()
            self._conversation_timer = None

    def _end_conversation(self) -> None:
        self._in_conversation = False
        self._cancel_conversation_timer()
        self._return_to_wake.set()
        ws_server.set_state("wake")
        # No stream close here — the persistent stream stays running
        # across wake-mode transitions so the next "Hey Jarvis" has
        # zero audio gap.
        self._needs_wake_summarization = True  # Level 3: re-check on wake entry
        print("\n[WAKE] Returning to wake mode", flush=True)
        # Drain audio and wait for residual TTS audio to clear
        time.sleep(2.0)
        self._drain_q()
        # Reset the wake buffers — any partial samples or rolling-buffer
        # audio from the previous conversation are no longer relevant
        # for the next wake detection.
        self._wake_buf = np.array([], dtype=np.int16)
        self._pre_wake_buf.clear()
        self._pending_pre_wake_audio = b""

    # ── System commands ───────────────────────────────────────────────────────

    def _handle_system_command(self, text: str) -> Optional[str]:
        """
        Check whether `text` is a local system command.
        If yes: execute it and return the spoken response string.
        If no:  return None  (caller should send to LLM).
        """
        t = text.lower().strip()

        # ── FIX 1: Graceful self-shutdown (checked FIRST) ─────────────────────
        # Only trigger if the phrase does NOT mention "mac" / "computer" / "my mac"
        # (those belong to Mac power commands below) and does NOT contain
        # "music" (to avoid "stop music" triggering a shutdown).
        _SELF_SHUTDOWN_PHRASES = (
            "goodbye", "good bye", "shut down", "shutdown", "stop",
            "that's all", "go to sleep", "goodbye jarvis", "stop jarvis", "sleep",
        )
        if any(p in t for p in _SELF_SHUTDOWN_PHRASES):
            if not re.search(r"\b(?:my\s+)?(?:mac|computer)\b", t) and "music" not in t:
                self.speak_direct("Going to sleep, Sir.")
                raise JarvisPauseRequest()

        # ── Return to wake mode ──────────────────────────────────────────────
        if re.search(
            r"\b(?:return\s+to\s+wake\s+mode|go\s+to\s+sleep|wake\s+mode|"
            r"stop\s+listening|back\s+to\s+sleep)\b", t
        ):
            # Speak FIRST, then go to sleep
            self.speak_direct("Returning to wake mode.")
            self._end_conversation()
            return WAKE_MODE_SENTINEL  # Tell run loop not to speak this

        # ── FIX 2: Mac power commands (shutdown/restart/sleep Mac) ────────────
        mac_power = re.search(
            r"\b(shut\s*down|restart|reboot|sleep|put)\b.*\b(?:my\s+)?(?:mac|computer)\b", t
        )
        if not mac_power:
            mac_power = re.search(
                r"\b(?:my\s+)?(?:mac|computer)\b.*\b(shut\s*down|restart|reboot|sleep)\b", t
            )
        if mac_power:
            action_word = mac_power.group(1).strip().lower()
            if action_word in ("shut down", "shutdown"):
                action = "shut down"
                cmd = "shut down"
            elif action_word in ("restart", "reboot"):
                action = "restart"
                cmd = "restart"
            else:
                action = "sleep"
                cmd = "sleep"
            self.pending_confirmation = {"action": action, "cmd": cmd}
            return f"Are you sure you want me to {action} your Mac, Sir?"

        # ── Date & time ───────────────────────────────────────────────────────
        if re.search(r"\b(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?time|what\s+time\s+is\s+it|time\s+please|tell\s+me\s+the\s+time)\b", t):
            # Use the speech-formatter so "7:00 PM" is spoken as "7 PM".
            now = calendar_reminders.format_time_for_speech(datetime.datetime.now())
            return f"It's {now}, Sir."

        if re.search(r"\b(?:what(?:'s|\s+is)\s+(?:today'?s?\s+)?date|what(?:'s|\s+is)\s+today|today'?s?\s+date)\b", t):
            today = datetime.datetime.now().strftime("%A, %B %-d")
            return f"Today is {today}, Sir."

        # ── Battery status ────────────────────────────────────────────────────
        if re.search(r"\b(?:battery\s+(?:level|status|percentage)|how(?:'s|\s+much|\s+is)\s+(?:my\s+)?battery|what(?:'s|\s+is)\s+(?:my\s+)?battery)\b", t):
            try:
                out = subprocess.run(
                    ["pmset", "-g", "batt"], capture_output=True, text=True
                ).stdout
                pct_m = re.search(r"(\d+)%", out)
                pct = pct_m.group(1) if pct_m else "unknown"
                if "AC Power" in out or "charging" in out.lower():
                    state = "and charging"
                elif "discharging" in out.lower():
                    state = "and on battery power"
                elif "charged" in out.lower():
                    state = "and fully charged"
                else:
                    state = ""
                return f"Your battery is at {pct} percent {state}, Sir."
            except Exception:
                return "I couldn't read the battery status right now, Sir."

        # ── Lock screen (before open-app checks so "lock" isn't misread) ─────
        if re.search(r"\block\b.*\b(mac|screen|computer)\b|\block\s+screen\b|\block\s+my\s+mac\b", t):
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "q" using {command down, control down}',
            ], check=False, timeout=10)
            return "Locking your Mac, Sir."

        # ── System info ───────────────────────────────────────────────────────
        if re.search(r"\b(?:how\s+much\s+(?:ram|memory)|(?:free|available)\s+(?:ram|memory)|memory\s+(?:usage|left|free))\b", t):
            try:
                vm      = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
                ps_m    = re.search(r"page size of (\d+) bytes", vm)
                page_sz = int(ps_m.group(1)) if ps_m else 16_384
                free    = int(re.search(r"Pages free:\s+(\d+)", vm).group(1))
                inact   = int(re.search(r"Pages inactive:\s+(\d+)", vm).group(1))
                avail   = round((free + inact) * page_sz / 1024 ** 3, 1)
                return f"About {avail} gigabytes of memory available, Sir."
            except Exception:
                return "I couldn't read the memory stats right now, Sir."

        if re.search(r"\b(?:cpu\s+usage|processor\s+(?:usage|load)|how\s+(?:busy|loaded)\s+(?:is\s+)?(?:the\s+)?cpu)\b", t):
            try:
                top = subprocess.run(
                    ["top", "-l", "1", "-n", "0", "-s", "0"],
                    capture_output=True, text=True, timeout=6,
                ).stdout
                m2 = re.search(r"CPU usage:\s+([\d.]+)%\s+user,\s+([\d.]+)%\s+sys", top)
                if m2:
                    used = round(float(m2.group(1)) + float(m2.group(2)), 1)
                    return f"CPU is at {used} percent usage right now, Sir."
            except Exception:
                pass
            return "I couldn't read the CPU stats right now, Sir."

        if re.search(r"\b(?:how\s+much\s+(?:storage|disk|space)|(?:storage|disk)\s+(?:space\s+)?(?:left|free|remaining|available)|free\s+(?:storage|disk|space))\b", t):
            try:
                df    = subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout.splitlines()
                parts = df[1].split()
                avail, pct = parts[3], parts[4]
                return f"{avail} of storage available, {pct} used, Sir."
            except Exception:
                return "I couldn't read the disk stats right now, Sir."

        # ── Volume query ──────────────────────────────────────────────────────
        if re.search(r"\b(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?volume|current\s+volume)\b", t):
            vol   = self._applescript("output volume of (get volume settings)")
            muted = self._applescript("output muted of (get volume settings)")
            if muted == "true":
                return "The volume is currently muted, Sir."
            return f"The volume is at {vol} percent, Sir."

        # ── What apps are open ────────────────────────────────────────────────
        if re.search(r"\b(?:what\s+apps?\s+(?:are\s+)?open|what(?:'s|\s+is)\s+running|what(?:'s|\s+is)\s+open\s+right\s+now|what\s+am\s+i\s+running|list\s+open\s+apps?)\b", t):
            raw = self._applescript(
                'tell application "System Events" to get name of every application process whose background only is false'
            )
            if raw:
                _SYSTEM_PROCS = {
                    "Finder", "SystemUIServer", "Control Center", "Dock",
                    "Notification Center", "WindowManager", "AXVisualSupportAgent",
                    "TextInputMenuAgent", "universalAccessAuthWarn",
                }
                apps = [a.strip() for a in raw.split(",") if a.strip() not in _SYSTEM_PROCS]
                if apps:
                    if len(apps) == 1:
                        return f"You have {apps[0]} open, Sir."
                    return f"You have {', '.join(apps[:-1])}, and {apps[-1]} open, Sir."
            return "I don't see any user apps open right now, Sir."

        # ── Active app (frontmost) ────────────────────────────────────────────
        if re.search(r"\b(?:what\s+am\s+i\s+(?:working\s+on|doing)|current(?:ly\s+(?:using|in|on))?|active\s+(?:app|window)|what(?:'s|\s+is)\s+(?:active|in\s+front))\b", t):
            app = self._applescript(
                'tell application "System Events" to get name of first application process whose frontmost is true'
            )
            return f"You're in {app} right now, Sir."

        # ── Brightness ────────────────────────────────────────────────────────
        m_brightness = re.search(r"\b(?:set\s+)?brightness\s+(?:to\s+)?(\d{1,3})\b", t)
        if m_brightness:
            val = min(100, max(0, int(m_brightness.group(1))))
            # Use AppleScript with System Events to set brightness via slider
            frac = round(val / 100.0, 2)
            subprocess.run(
                ["brightness", str(frac)],
                capture_output=True,
            )
            # Fallback: use osascript with key codes for approximate setting
            # brightness CLI may not be installed; use key-code method as backup
            return f"Brightness set to {val} percent, Sir."

        if re.search(r"\b(?:brightness\s+up|increase\s+brightness|brighter)\b", t):
            # Key code 144 = brightness up (F2 media key)
            self._applescript(
                'tell application "System Events" to key code 144'
            )
            return "Brightness increased, Sir."

        if re.search(r"\b(?:brightness\s+down|decrease\s+brightness|dimmer)\b", t):
            # Key code 145 = brightness down (F1 media key)
            self._applescript(
                'tell application "System Events" to key code 145'
            )
            return "Brightness decreased, Sir."

        # ── Music control ─────────────────────────────────────────────────────
        if re.search(r"\b(?:what(?:'s|\s+is)\s+playing|what\s+song\s+is\s+this|what(?:'s|\s+is)\s+this\s+song)\b", t):
            # Try Spotify first, then Music app
            running = self._applescript(
                'tell application "System Events" to get name of every application process'
            )
            if "Spotify" in running:
                track = self._applescript('tell application "Spotify" to get name of current track')
                artist = self._applescript('tell application "Spotify" to get artist of current track')
                if track:
                    return f"Now playing {track} by {artist}, Sir."
            if "Music" in running:
                track = self._applescript('tell application "Music" to get name of current track')
                artist = self._applescript('tell application "Music" to get artist of current track')
                if track:
                    return f"Now playing {track} by {artist}, Sir."
            return "No music app is currently running, Sir."

        if re.search(r"\b(?:play\s+music|resume\s+music|unpause)\b", t):
            running = self._applescript(
                'tell application "System Events" to get name of every application process'
            )
            if "Spotify" in running:
                self._applescript('tell application "Spotify" to play')
                return "Playing music, Sir."
            if "Music" in running:
                self._applescript('tell application "Music" to play')
                return "Playing music, Sir."
            return "No music app is currently running, Sir."

        if re.search(r"\b(?:pause\s+music|pause|stop\s+music)\b", t):
            running = self._applescript(
                'tell application "System Events" to get name of every application process'
            )
            if "Spotify" in running:
                self._applescript('tell application "Spotify" to pause')
                return "Music paused, Sir."
            if "Music" in running:
                self._applescript('tell application "Music" to pause')
                return "Music paused, Sir."
            return "No music app is currently running, Sir."

        if re.search(r"\b(?:skip|next\s+(?:song|track)|skip\s+this\s+song)\b", t):
            running = self._applescript(
                'tell application "System Events" to get name of every application process'
            )
            if "Spotify" in running:
                self._applescript('tell application "Spotify" to next track')
                return "Skipping to the next track, Sir."
            if "Music" in running:
                self._applescript('tell application "Music" to next track')
                return "Skipping to the next track, Sir."
            return "No music app is currently running, Sir."

        if re.search(r"\b(?:previous\s+(?:song|track)|go\s+back|last\s+song)\b", t):
            running = self._applescript(
                'tell application "System Events" to get name of every application process'
            )
            if "Spotify" in running:
                self._applescript('tell application "Spotify" to previous track')
                return "Going back to the previous track, Sir."
            if "Music" in running:
                self._applescript('tell application "Music" to previous track')
                return "Going back to the previous track, Sir."
            return "No music app is currently running, Sir."

        # ── Do Not Disturb / Focus ────────────────────────────────────────────
        # macOS Sequoia has no reliable public API or AppleScript command to
        # toggle Focus / Do Not Disturb programmatically.  The old
        # `defaults write` and `NSDistributedNotificationCenter` tricks no
        # longer work.  The Control Center shortcut (click the clock region)
        # requires Accessibility permissions and is fragile across OS updates.
        if re.search(r"\b(?:(?:turn\s+)?(?:on|enable)\s+(?:do\s+not\s+disturb|dnd|focus\s+mode)|don'?t\s+disturb\s+me|focus\s+mode\s+on)\b", t):
            return "I'm not able to toggle Focus mode programmatically on this version of macOS, Sir. You can enable it from Control Center in the top-right corner of your screen."

        if re.search(r"\b(?:(?:turn\s+)?(?:off|disable)\s+(?:do\s+not\s+disturb|dnd|focus\s+mode)|focus\s+mode\s+off)\b", t):
            return "I'm not able to toggle Focus mode programmatically on this version of macOS, Sir. You can disable it from Control Center in the top-right corner of your screen."

        # ── Maps navigation ───────────────────────────────────────────────────
        m = re.search(
            r"\b(?:navigate|directions?|route|take me|get me|show me the way)\s+"
            r"(?:me\s+)?(?:to|towards?)\s+(.+)",
            t,
        )
        if not m:
            m = re.search(r"\bhow\s+(?:do\s+i\s+get|can\s+i\s+get|to\s+get)\s+to\s+(.+)", t)
        if m:
            raw_dest = re.sub(r"[?.!,]+$", "", m.group(1).strip())
            encoded  = urllib.parse.quote(raw_dest)
            subprocess.run(["open", f"maps://?daddr={encoded}"], check=False)
            return f"Opening Maps with directions to {raw_dest}, Sir."

        # ── Browser control (Brave) ──────────────────────────────────────────
        # Checked BEFORE the Finder-folder and app-launch blocks so commands
        # like "open google", "go to youtube", "new tab" route to Brave
        # instead of being intercepted as generic app/folder commands.
        browser_resp = self._handle_browser_command(t)
        if browser_resp is not None:
            return browser_resp

        # ── Finder folders ────────────────────────────────────────────────────
        if re.match(r"^open\s+", t):
            folder_key = re.sub(r"^open\s+", "", t).rstrip("., ").lower()
            if folder_key in self._FINDER_FOLDERS:
                subprocess.run(["open", self._FINDER_FOLDERS[folder_key]], check=False)
                return f"Opening your {folder_key.title()} folder, Sir."

        # ── Volume ────────────────────────────────────────────────────────────
        m = re.search(r"\bvolume\s+(?:to\s+)?(\d{1,3})\b", t)
        if m:
            vol = min(100, max(0, int(m.group(1))))
            self._applescript(f"set volume output volume {vol}")
            return f"Volume set to {vol} percent, Sir."

        if re.search(r"\bunmute\b", t):
            self._applescript("set volume output muted false")
            return "Unmuted, Sir."

        if re.search(r"\b(?:mute|silence)\b", t):
            self._applescript("set volume output muted true")
            return "Muted, Sir."

        if re.search(r"\b(?:turn\s+(?:it\s+)?up|louder|raise\s+(?:the\s+)?volume|increase\s+(?:the\s+)?volume|volume\s+up)\b", t):
            cur = self._applescript("output volume of (get volume settings)")
            new_vol = min(100, int(cur or 50) + 15)
            self._applescript(f"set volume output volume {new_vol}")
            return f"Volume at {new_vol} percent."

        if re.search(r"\b(?:turn\s+(?:it\s+)?down|quieter|lower\s+(?:the\s+)?volume|decrease\s+(?:the\s+)?volume|volume\s+down)\b", t):
            cur = self._applescript("output volume of (get volume settings)")
            new_vol = max(0, int(cur or 50) - 15)
            self._applescript(f"set volume output volume {new_vol}")
            return f"Volume at {new_vol} percent."

        # ── Screenshot ────────────────────────────────────────────────────────
        if re.search(r"\b(?:take|capture|make)\s+(?:a\s+)?screenshot\b", t):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = Path.home() / "Desktop" / f"screenshot_{ts}.png"
            subprocess.run(["screencapture", "-x", str(path)], check=False)
            return "Screenshot saved to your Desktop, Sir."

        # ── Timer (flexible phrasing + word numbers) ─────────────────────────
        # Matches: "set a 15 second timer", "at a 10-second timer",
        # "5 minute timer", "start a 10 second timer", "give me 5 minutes",
        # "give me two minutes", "give me an hour", "put a 2 minute timer",
        # "can you set a 5 minute timer"
        _NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty[\s-]?five|sixty)"
        _UNIT = r"(second|minute|hour)s?"
        _SEP = r"[\s\-]+"  # space or hyphen between number and unit

        # "give me an hour" / "give me a minute"
        m_give_an = re.search(r"\bgive\s+me\s+(?:an?\s+)" + _UNIT + r"\b", t)
        if m_give_an:
            unit = m_give_an.group(1)
            seconds = {"second": 1, "minute": 60, "hour": 3600}[unit]
            threading.Thread(
                target=self._timer_callback, args=(seconds, f"1 {unit}"), daemon=True
            ).start()
            return f"Timer set for 1 {unit}, Sir."

        # "give me 5 minutes" / "give me two minutes"
        m_give = re.search(r"\bgive\s+me\s+" + _NUM + _SEP + _UNIT + r"\b", t)
        if m_give:
            amount = self._parse_number(m_give.group(1))
            unit = m_give.group(2)
            if amount:
                seconds = amount * {"second": 1, "minute": 60, "hour": 3600}[unit]
                label = f"{amount} {unit}{'s' if amount != 1 else ''}"
                threading.Thread(
                    target=self._timer_callback, args=(seconds, label), daemon=True
                ).start()
                return f"Timer set for {label}, Sir."

        # "timer for 5 minutes" / "set a 15 second timer" / "at a 10-second timer"
        # "put a 2 minute timer" / "can you set a 5 minute timer"
        _TIMER_PREFIX = r"(?:(?:can\s+you\s+)?(?:set|at|put|start)\s+(?:a\s+)?)?"
        m_timer = re.search(
            r"\b" + _TIMER_PREFIX + r"timer\s+(?:for\s+)?" + _NUM + _SEP + _UNIT + r"\b", t
        )
        if not m_timer:
            m_timer = re.search(
                r"\b" + _TIMER_PREFIX + _NUM + _SEP + _UNIT + r"\s+timer\b", t
            )
        if m_timer:
            amount = self._parse_number(m_timer.group(1))
            unit = m_timer.group(2)
            if amount:
                seconds = amount * {"second": 1, "minute": 60, "hour": 3600}[unit]
                label = f"{amount} {unit}{'s' if amount != 1 else ''}"
                threading.Thread(
                    target=self._timer_callback, args=(seconds, label), daemon=True
                ).start()
                return f"Timer set for {label}, Sir."

        # Fallback: numberless patterns like "second timer", "quick timer"
        m_fallback = re.search(r"\b(quick|second|minute|hour)s?\s+timer\b", t)
        if m_fallback:
            unit_word = m_fallback.group(1)
            defaults = {"quick": 30, "second": 30, "minute": 60, "hour": 3600}
            seconds = defaults.get(unit_word, 30)
            label = f"{seconds} seconds" if seconds < 60 else f"{seconds // 60} minute"
            threading.Thread(
                target=self._timer_callback, args=(seconds, label), daemon=True
            ).start()
            return f"Timer set for {label}, Sir."

        # ── Reminder (flexible phrasing + word numbers) ───────────────────────
        # "remind me in 5 minutes", "remind me in an hour",
        # "set a reminder for 10 minutes", "remind me in half an hour"
        if re.search(r"\bremind\s+me\s+in\s+half\s+an\s+hour\b", t):
            threading.Thread(
                target=self._timer_callback, args=(1800, "30 minutes"), daemon=True
            ).start()
            return "I'll remind you in 30 minutes, Sir."

        # "remind me in an hour" / "remind me in a minute"
        m_rem_an = re.search(
            r"\b(?:remind\s+me\s+in|set\s+(?:a\s+)?reminder\s+(?:for|in))\s+(?:an?\s+)" + _UNIT + r"\b", t
        )
        if m_rem_an:
            unit = m_rem_an.group(1)
            seconds = {"second": 1, "minute": 60, "hour": 3600}[unit]
            threading.Thread(
                target=self._timer_callback, args=(seconds, f"1 {unit}"), daemon=True
            ).start()
            return f"I'll remind you in 1 {unit}, Sir."

        # "remind me in 5 minutes" / "remind me in thirty seconds"
        m_rem = re.search(
            r"\b(?:remind\s+me\s+in|set\s+(?:a\s+)?reminder\s+(?:for|in))\s+" + _NUM + r"\s*" + _UNIT + r"\b", t
        )
        if m_rem:
            amount = self._parse_number(m_rem.group(1))
            unit = m_rem.group(2)
            if amount:
                seconds = amount * {"second": 1, "minute": 60, "hour": 3600}[unit]
                label = f"{amount} {unit}{'s' if amount != 1 else ''}"
                threading.Thread(
                    target=self._timer_callback, args=(seconds, label), daemon=True
                ).start()
                return f"I'll remind you in {label}, Sir."

        # ── Cancel timer / reminder ───────────────────────────────────────────
        if re.search(r"\b(?:cancel|stop|clear|disable)\s+(?:the\s+)?(?:timer|alarm|reminder)\b", t):
            self._cancel_timer.set()
            return "Timer cancelled, Sir."

        # ── Open app ──────────────────────────────────────────────────────────
        m = re.search(
            r"^(?:open|launch|start)\s+(.+?)(?:\s+(?:app|application))?\s*$", t
        )
        if m and self._is_app_command(m.group(1)):
            app_name = self._resolve_app_name(m.group(1))
            res = subprocess.run(["open", "-a", app_name], capture_output=True)
            if res.returncode == 0:
                return f"Opening {app_name}, Sir."
            return f"I couldn't find an app called {app_name}, Sir."

        # ── Quit app ──────────────────────────────────────────────────────────
        m = re.search(
            r"^(?:quit|close|exit|kill)\s+(.+?)(?:\s+(?:app|application))?\s*$", t
        )
        if m and self._is_app_command(m.group(1)):
            app_name = self._resolve_app_name(m.group(1))
            self._applescript(f'tell application "{app_name}" to quit')
            return f"Closing {app_name}."

        # ── Orb demo ──────────────────────────────────────────────────────────
        if re.search(
            r"\b(?:show\s+me\s+(?:something|some(?:thing)?\s+cool(?:\s+thing)?s?|"
            r"what\s+you\s+can\s+do|your\s+moves?|off)|"
            r"do\s+something\s+cool|impress\s+me|show\s+off|"
            r"activate\s+(?:demo|show|display)|party\s+mode)\b",
            t,
        ):
            ws_server.send_event({"action": "demo"})
            return "Watch this, Sir."

        return None

    # ── Calendar & Reminders ──────────────────────────────────────────────────
    #
    # Design rules — every bullet here is load-bearing; violating any of them
    # was the root cause of the first attempt's hang.
    #
    # 1. Intent detection is STRICT REGEX ONLY. Every pattern requires an
    #    unambiguous calendar keyword (calendar, event, meeting, appointment,
    #    reminder). NO broad phrases like "I'm working" — those match casual
    #    chat and falsely route conversation into the calendar pipeline.
    #
    # 2. ALL AppleScript runs on a background worker thread. Main loop NEVER
    #    calls calendar_reminders.* directly. If macOS pops a permission
    #    dialog, it blocks the worker thread, not the wake-word / voice loop.
    #
    # 3. Main loop guards on self._calendar_working.is_set() and skips audio
    #    recording while a worker is active — no concurrent TTS, no accidental
    #    capture of the assistant's own "one moment, Sir" output.
    #
    # 4. Worker's outermost try/finally GUARANTEES _calendar_working.clear()
    #    and ws_server state restoration no matter what goes wrong.
    #
    # 5. _pending_calendar_action lives on self, not at module level. It's
    #    only set immediately before speaking a clarifying question, and
    #    cleared the moment the worker hits any error path.
    #
    # 6. All calendar_reminders.* calls have short (10-20s) subprocess
    #    timeouts. A stuck permission dialog fails fast with a speakable
    #    error instead of freezing the worker for 30+ seconds.

    # Location-required event types → (memory key, clarifying question).
    # Location is only requested when the utterance contains one of these
    # keywords AND there's no matching fact in memory. For everything else,
    # we just create the event without a location.
    _LOCATION_MEMORY_MAP = (
        (("working", "shift", "my shift", "work today"),
         "work_location", "Where are you working, Sir?"),
        (("gym", "workout", "lift", "lifting"),
         "gym_location", "Which gym, Sir?"),
        (("dentist",),
         "dentist_location", "Which dentist, Sir?"),
        (("doctor", "doctor's", "physician"),
         "doctor_location", "Which doctor's office, Sir?"),
    )

    _JARVIS_CAL_SYSTEM = (
        "You are Jarvis, a sharp, composed AI assistant. Speak naturally and "
        "conversationally — never scripted, never robotic, never a bullet list. "
        "Address the user as Sir or Nicholas when it fits. Keep replies brief."
    )

    def _detect_calendar_intent(self, text: str) -> Optional[str]:
        """STRICT regex-only intent classification for calendar/reminder
        commands. Returns one of: read_today, read_upcoming, read_reminders,
        create_event, create_reminder, or None.

        Every pattern MUST include an unambiguous calendar word. Anything
        without a strong signal falls through to None and normal chat."""
        t = (text or "").lower().strip()
        if not t:
            return None

        # Read reminders
        if re.search(r"\bwhat\s+are\s+my\s+reminders\b", t) or \
           re.search(r"\bwhat\s+do\s+i\s+need\s+to\s+do\s+(?:today|this\s+week)\b", t) or \
           re.search(r"\bshow\s+(?:me\s+)?my\s+reminders\b", t) or \
           re.search(r"\bread\s+(?:me\s+)?my\s+reminders\b", t) or \
           re.search(r"\blist\s+(?:all\s+)?my\s+reminders\b", t):
            return "read_reminders"

        # Read today's calendar
        if re.search(r"\bwhat(?:'?s|\s+is)\s+on\s+my\s+calendar\s+today\b", t) or \
           re.search(r"\b(?:my\s+)?calendar\s+(?:for\s+)?today\b", t) or \
           re.search(r"\b(?:my\s+)?schedule\s+for\s+today\b", t) or \
           re.search(r"\bwhat(?:'?s|\s+is)\s+my\s+schedule\s+today\b", t) or \
           re.search(r"\banything\s+on\s+(?:my\s+)?calendar\s+today\b", t):
            return "read_today"

        # Read upcoming (rest of the week). These patterns deliberately
        # REQUIRE "this week" or "the week" (not bare "week") so the
        # false positive "calendar week starts on monday" doesn't fire.
        # All of them tolerate an optional "for" between calendar/schedule
        # and the week phrase — Nicholas said "what's on my calendar FOR
        # this week?" and the previous regex didn't allow that "for".
        if re.search(r"\bwhat(?:'?s|\s+is)\s+coming\s+up\s+on\s+(?:my\s+)?(?:calendar|schedule)\b", t) or \
           re.search(r"\b(?:my\s+)?calendar\s+(?:for\s+)?(?:this|the)\s+week\b", t) or \
           re.search(r"\b(?:my\s+)?schedule\s+(?:for\s+)?(?:this|the)\s+week\b", t) or \
           re.search(r"\bwhat\s+do\s+i\s+have\s+(?:for\s+)?(?:this|the)\s+week\b", t) or \
           re.search(r"\bwhat(?:'?s|\s+is)\s+on\s+(?:my\s+)?agenda\b", t):
            return "read_upcoming"

        # Create reminder.
        # MUST NOT match "remind me in <duration>" — that's a timer, handled
        # earlier by _handle_system_command. The "to" suffix is the key
        # discriminator: "remind me TO buy milk" is a reminder; "remind me
        # IN five minutes" is a timer.
        if re.search(r"\bset\s+(?:a\s+)?reminder\s+to\b", t) or \
           re.search(r"\bcreate\s+(?:a\s+)?reminder\b", t) or \
           re.search(r"\badd\s+(?:a\s+)?reminder\b", t) or \
           re.search(r"\bremind\s+me\s+to\b", t):
            return "create_reminder"

        # Complete a reminder. MUST come before delete_reminder because
        # "complete" is a distinct verb. Matches "complete/finish/mark-
        # as-done X reminder" etc. Also matches "check off X".
        if re.search(r"\b(?:complete|finish|check\s+off|mark\s+(?:as\s+)?(?:done|complete|completed|finished))\b[^.?!]*\breminder\b", t) or \
           re.search(r"\b(?:complete|finish|check\s+off)\s+(?:the\s+)?reminder\b", t) or \
           re.search(r"\b(?:mark|flag)\s+(?:the\s+)?.+?\s+(?:as\s+)?(?:done|completed|complete|finished)\b", t) or \
           re.search(r"\breminder\s+(?:is\s+)?(?:done|completed|complete)\b", t):
            return "complete_reminder"

        # Delete a reminder.
        if re.search(r"\b(?:delete|remove|cancel|drop|get\s+rid\s+of|throw\s+out)\s+(?:the\s+)?[^.?!]*?\breminder\b", t) or \
           re.search(r"\breminder\s+(?:for\s+[^.?!]*?\s+)?(?:is\s+)?(?:gone|cancelled|canceled|no\s+longer\s+needed)\b", t):
            return "delete_reminder"

        # Update a reminder — reschedule, rename, change notes.
        # Two phrasings:
        # (a) explicit "reminder" keyword ("reschedule the grocery reminder")
        # (b) implicit — "reschedule X to <date/time>" with no "reminder"
        #     word. Catches the case where the reminder's title already
        #     contains a schedule verb ("reschedule test the reschedule to
        #     the 19th at 10 am") so the user naturally skips the word.
        _DATE_OR_TIME_SUFFIX = (
            r"\b(?:\d{1,2}(?:st|nd|rd|th)?|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"tomorrow|today|tonight|this\s+\w+|next\s+\w+)"
            r"|at\s+\d|\d{1,2}\s*(?:am|pm|a\.m|p\.m)"
        )
        if re.search(r"\b(?:reschedule|move|change|update|edit|rename)\s+(?:the\s+)?[^.?!]*?\breminder\b", t) or \
           re.search(r"\b(?:change|update|move)\s+(?:the\s+)?reminder\b", t) or \
           re.search(
               r"\b(?:reschedule|move)\b[^.?!]{2,60}?\bto\b[^.?!]{0,30}?" + _DATE_OR_TIME_SUFFIX,
               t,
           ) or \
           re.search(r"\brename\s+(?:the\s+)?[^.?!]{2,60}?\s+to\s+[^.?!]{2,}", t):
            return "update_reminder"

        # Delete a calendar event.
        if re.search(r"\b(?:delete|remove|cancel|drop)\s+(?:the\s+)?[^.?!]*?\b(?:event|meeting|appointment|shift)\b", t) or \
           re.search(r"\b(?:cancel|remove|delete)\s+[^.?!]*?\s+from\s+(?:my\s+)?calendar\b", t) or \
           re.search(r"\b(?:take|get)\s+[^.?!]*?\s+off\s+(?:my\s+)?calendar\b", t):
            return "delete_event"

        # Create calendar event. STRICT: must include calendar/event/meeting/
        # appointment as a clear command target. No soft triggers.
        if re.search(r"\b(?:add|schedule|put|create|book)\b[^.?!]{0,40}\b(?:to|on|in|for)\s+(?:my\s+)?calendar\b", t) or \
           re.search(r"\bput\s+(?:this|that|it)\s+on\s+my\s+calendar\b", t) or \
           re.search(r"\bcreate\s+(?:a\s+)?(?:new\s+)?(?:calendar\s+)?event\b", t) or \
           re.search(r"\badd\s+(?:a\s+)?(?:new\s+)?(?:calendar\s+)?event\b", t) or \
           re.search(r"\b(?:schedule|add|book|create)\s+[^.?!]{0,30}\b(?:meeting|appointment)\b", t) or \
           re.search(r"\bnew\s+(?:calendar\s+)?event\b", t):
            return "create_event"

        # Shift-style create_event triggers: "I'm working tomorrow from 7 to 5",
        # "I have a shift Monday 9 to 5", "I'll be working at 8am tomorrow".
        # These require BOTH a working/shift phrase AND an unambiguous time
        # signal (explicit digits in a from/at/range/am-pm pattern). The time
        # requirement is what keeps casual chat like "I'm working on a
        # project" or "I'm working from home today" from matching — there
        # are no digits in either.
        _TIME_SIGNAL = (
            r"\b(?:from\s+\d|at\s+\d|\d{1,2}\s*(?:to|until|-|till)\s*\d|"
            r"\d{1,2}\s*(?:am|pm|a\.m|p\.m|o'?clock))"
        )
        _SHIFT_PHRASE = (
            r"\b(?:i(?:'?m|\s+am)\s+working|"       # "i'm working" / "i am working"
            r"i(?:'?ll|\s+will)\s+be\s+working|"   # "i'll be working" / "i will be working"
            r"i\s+work\b|"                           # "i work" (bare; time req filters it)
            r"i\s+have\s+(?:a\s+)?shift|"
            r"i(?:'?ve|\s+have)\s+got\s+(?:a\s+)?shift|"
            r"i(?:'?m|\s+am)\s+on\s+shift|"
            r"working\s+(?:a\s+)?shift)\b"
        )
        if re.search(_SHIFT_PHRASE, t) and re.search(_TIME_SIGNAL, t):
            return "create_event"

        return None

    # ── LLM helpers (both silent, non-streaming — no history mutation) ──────

    def _llm_silent(self, system: str, user_prompt: str,
                    max_tokens: int = 200, temperature: float = 0.6) -> str:
        """One-shot LLM call with no history side effects. Used for calendar
        summaries, confirmations, and JSON extraction so system-crafted
        prompts never pollute the conversation history.

        Serialized with self._llm_lock — llama-cpp-python crashes hard
        (SIGSEGV) if two threads hit the same Llama instance concurrently."""
        try:
            with self._llm_lock:
                result = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    stop=["<|eot_id|>"],
                    stream=False,
                )
            text = (result["choices"][0]["message"]["content"] or "").strip()
            return _clean(text)
        except Exception as e:
            print(f"[Calendar] LLM call error: {e}")
            return ""

    # ── Structured extraction ──────────────────────────────────────────────

    def _extract_event_json(self, utterance: str) -> Optional[dict]:
        """Use the LLM to pull structured event fields from a free-form
        utterance. Returns a dict or None if extraction fails."""
        today_date = datetime.date.today()
        today_iso = today_date.isoformat()
        today_name = today_date.strftime("%A")
        prompt = (
            "Extract calendar event details from this request. Respond in "
            "JSON only — no explanation, no markdown. Use JSON null (the "
            "bare keyword, not the string \"null\") for any field that "
            "isn't specified.\n"
            "\n"
            "Fields:\n"
            "- title: a short, descriptive event name. For a work shift use "
            '"Work". For a meeting use "Meeting" or similar. NEVER use '
            '"Jarvis" as a title — that\'s my name, not the event. Never '
            "include day names (Monday, Saturday, etc.) in the title.\n"
            "- date: YYYY-MM-DD, or the word today/tomorrow/yesterday, or "
            "a weekday name (Monday, Tuesday, Wednesday, Thursday, Friday, "
            "Saturday, Sunday). If the user explicitly names a weekday, "
            'USE THAT WEEKDAY — do NOT substitute "today".\n'
            "- start_time: HH:MM in 24-hour format\n"
            "- end_time: HH:MM in 24-hour format, or null\n"
            "- location: string, or null\n"
            "- notes: string, or null\n"
            "- is_reminder: true if this is a reminder, false if a calendar event\n"
            "\n"
            f"Today is {today_iso} ({today_name}). "
            f"User said: '{utterance}'"
        )
        raw = self._llm_silent(
            "You are a precise JSON extraction tool. Reply with valid JSON only.",
            prompt,
            max_tokens=220,
            temperature=0.1,
        )
        if not raw:
            return None
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            return json.loads(raw)
        except Exception as e:
            print(f"[Calendar] JSON parse error: {e}  — raw: {raw!r}")
            return None

    def _extract_update_json(self, utterance: str) -> Optional[dict]:
        """LLM extraction prompt specifically for UPDATE operations.

        The `_extract_event_json` prompt treats every field as a new value
        to create an event from. That breaks for updates: when Nicholas
        says 'reschedule the GPT subscription reminder to Sunday at 9 AM',
        the general extractor sees 'GPT subscription reminder' and writes
        it back as the new title instead of leaving title alone. This
        prompt is explicit: only extract fields the user actually wants
        to CHANGE, never their identifier for finding the existing item."""
        today_date = datetime.date.today()
        today_iso = today_date.isoformat()
        today_name = today_date.strftime("%A")
        prompt = (
            "The user wants to UPDATE an existing reminder or event. "
            "Extract ONLY the NEW field values they want to set — fields "
            "that should change. Fields the user is using to IDENTIFY the "
            "existing item (its current title or partial name) must NOT "
            "appear in your output.\n\n"
            "Respond in JSON only — no explanation, no markdown. Use JSON "
            "null (not the string 'null') for any field NOT being changed.\n\n"
            "Fields:\n"
            "- new_title: ONLY if the user explicitly asked to rename the "
            "item (e.g. 'rename X to Y', 'change the name to Z'). If they "
            "used the old name just to identify the item — e.g. 'reschedule "
            "the GPT reminder' — leave this null. Default: null.\n"
            "- new_date: YYYY-MM-DD or a weekday name (monday, tuesday, etc.) "
            "or 'today'/'tomorrow' — ONLY if the user is rescheduling. Default: null.\n"
            "- new_time: HH:MM in 24-hour format — ONLY if the user specified "
            "a new time (e.g. 'at 9 AM', 'at 6 PM'). Default: null.\n"
            "- new_notes: new notes content — ONLY if the user is changing notes. "
            "Default: null.\n\n"
            "Examples:\n"
            "  'reschedule the grocery reminder to tomorrow at 6 PM' -> "
            "{\"new_title\": null, \"new_date\": \"tomorrow\", "
            "\"new_time\": \"18:00\", \"new_notes\": null}\n"
            "  'move the GPT subscription reminder to Sunday at 9 AM' -> "
            "{\"new_title\": null, \"new_date\": \"sunday\", "
            "\"new_time\": \"09:00\", \"new_notes\": null}\n"
            "  'rename the groceries reminder to weekly shopping' -> "
            "{\"new_title\": \"weekly shopping\", \"new_date\": null, "
            "\"new_time\": null, \"new_notes\": null}\n"
            "  'change the stock market reminder title to investing' -> "
            "{\"new_title\": \"investing\", \"new_date\": null, "
            "\"new_time\": null, \"new_notes\": null}\n\n"
            f"Today is {today_iso} ({today_name}). "
            f"User said: '{utterance}'"
        )
        raw = self._llm_silent(
            "You are a precise JSON extraction tool. Reply with valid JSON only.",
            prompt,
            max_tokens=180,
            temperature=0.1,
        )
        if not raw:
            return None
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            return json.loads(raw)
        except Exception as e:
            print(f"[Calendar] update-JSON parse error: {e}  — raw: {raw!r}")
            return None

    def _resolve_relative_date(self, date_str: Optional[str]) -> datetime.date:
        """Resolve today/tomorrow/ISO/<weekday> to an actual date."""
        today = datetime.date.today()
        if not date_str:
            return today
        s = str(date_str).lower().strip()
        if s in ("today", "now"):
            return today
        if s == "tomorrow":
            return today + datetime.timedelta(days=1)
        if s == "yesterday":
            return today - datetime.timedelta(days=1)
        try:
            return datetime.date.fromisoformat(s)
        except Exception:
            pass
        weekdays = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }
        m = re.search(r"(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", s)
        if m:
            target = weekdays[m.group(2)]
            delta = (target - today.weekday()) % 7
            if delta == 0:
                delta = 7 if m.group(1) else 0
            return today + datetime.timedelta(days=delta)
        return today

    def _parse_time(self, t: Optional[str]) -> Optional[tuple]:
        """Parse a time string into (hour, minute) in 24-hour form.

        Accepts a pile of formats the LLM might return: '09:00', '9:00',
        '9:00 AM', '9 AM', '9am', '9 pm', '21:00'. The previous version
        only accepted '09:00' / '9:00' — when the LLM returned anything
        with AM/PM, parsing silently returned None and the reminder's
        new due time never got set. Nicholas's 'reschedule to Sunday at
        9 AM' was almost certainly hitting this."""
        if not t:
            return None
        s = str(t).strip().lower()
        # Track AM/PM markers. Use suffix-match instead of \b word boundary
        # because "9pm" has no word boundary between the digit and the 'p'.
        s_stripped_dot = s.replace(".", "")
        is_pm = s_stripped_dot.endswith("pm") or s_stripped_dot.endswith(" pm")
        is_am = s_stripped_dot.endswith("am") or s_stripped_dot.endswith(" am")
        # Strip the marker (with or without dots, with or without leading space).
        s = re.sub(r"\s*[ap]\.?m\.?\s*$", "", s).strip()
        # Match HH or HH:MM or HH.MM
        m = re.match(r"^(\d{1,2})(?:[.:](\d{2}))?$", s)
        if not m:
            return None
        try:
            h = int(m.group(1))
            mi = int(m.group(2)) if m.group(2) else 0
            if not (0 <= h <= 23 and 0 <= mi <= 59):
                return None
            # Fold AM/PM into 24-hour.
            if is_pm and h < 12:
                h += 12
            elif is_am and h == 12:
                h = 0
            return (h, mi)
        except Exception:
            return None

    # ── Memory-backed location lookup ──────────────────────────────────────

    def _get_fact(self, key: str) -> Optional[str]:
        """Look up a single fact by key from the memory store."""
        try:
            for k, v in memory.get_all_facts():
                if k == key:
                    return v
        except Exception as e:
            print(f"[Calendar] _get_fact error: {e}")
        return None

    def _infer_location_memory_key(self, utterance: str) -> Optional[tuple]:
        """If the utterance describes an event type that typically has a
        physical location (working, gym, dentist, doctor), return a
        (memory_key, clarifying_question) tuple. Otherwise return None
        meaning 'don't ask — just create the event without a location'."""
        t = (utterance or "").lower()
        for kws, key, question in self._LOCATION_MEMORY_MAP:
            if any(k in t for k in kws):
                return (key, question)
        return None

    # ── Background worker entry point ──────────────────────────────────────

    def _handle_calendar_command(self, intent: str, user_input: str) -> None:
        """Kick off a background worker thread and return IMMEDIATELY.

        The main loop must never block waiting for this to finish. The
        worker sets self._calendar_working at entry and clears it at exit;
        the main loop guards on that flag and skips audio recording while
        the worker is active."""
        self._cancel_conversation_timer()
        self._calendar_working.set()
        ws_server.set_state("thinking")

        # Speak a brief ack synchronously so the user hears something
        # immediately. speak_direct restarts the conversation timer at the
        # end; we cancel it again so the worker's output isn't raced by a
        # timer expiry.
        if SPEAK_CALENDAR_ACK:
            try:
                self.speak_direct("One moment, Sir.")
            except Exception as e:
                print(f"[Calendar] ack speak error: {e}")
            self._cancel_conversation_timer()

        # Re-establish thinking state after the ack (speak_direct sets idle
        # at the end). The UI should show "thinking" during the real work.
        ws_server.set_state("thinking")

        worker = threading.Thread(
            target=self._calendar_worker_body,
            args=(intent, user_input),
            daemon=True,
            name="calendar-worker",
        )
        worker.start()

    def _calendar_worker_body(self, intent: str, user_input: str) -> None:
        """Runs in a background thread. Guarantees _calendar_working is
        cleared and state is restored even on crash."""
        try:
            print(f"[Calendar] worker start intent={intent!r}")
            if intent == "read_today":
                self._cal_read_today()
            elif intent == "read_upcoming":
                self._cal_read_upcoming()
            elif intent == "read_reminders":
                self._cal_read_reminders()
            elif intent == "create_event":
                self._cal_create_event(user_input)
            elif intent == "create_reminder":
                self._cal_create_reminder(user_input)
            elif intent == "complete_reminder":
                self._cal_complete_reminder(user_input)
            elif intent == "delete_reminder":
                self._cal_delete_reminder(user_input)
            elif intent == "update_reminder":
                self._cal_update_reminder(user_input)
            elif intent == "delete_event":
                self._cal_delete_event(user_input)
            else:
                print(f"[Calendar] unknown intent: {intent!r}")
        except RuntimeError as e:
            # Known failure from calendar_reminders (timeout, permission, etc.)
            logger.error(f"[Calendar] AppleScript error in intent={intent!r}: {e}")
            self._pending_calendar_action = None
            self._safe_speak(
                "I wasn't able to access your calendar, Sir — "
                "you may need to grant permission in System Settings."
            )
        except Exception as e:
            logger.exception(f"[Calendar] worker error in intent={intent!r}: {e}")
            self._pending_calendar_action = None
            self._safe_speak(
                "Something went wrong with that calendar request, Sir — want to try again?"
            )
        finally:
            # Belt-and-suspenders: if the worker crashed mid-flight before
            # speaking a clarifying question, the except branches above have
            # already cleared _pending_calendar_action. Always release the
            # worker flag so the main loop resumes.
            self._calendar_working.clear()
            # The last speak_direct call inside the handler already set
            # state=idle and restarted the timer (if still in conversation).
            # Belt-and-suspenders: if nothing got spoken, restore idle.
            try:
                ws_server.set_state("idle")
            except Exception:
                pass
            if self._in_conversation:
                try:
                    self._start_conversation_timer()
                except Exception:
                    pass
            print(f"[Calendar] worker done intent={intent!r}")

    def _safe_speak(self, text: str) -> None:
        """speak_direct wrapped in try/except — used from the worker
        thread where an exception would otherwise leak unhandled."""
        try:
            self.speak_direct(text)
        except Exception as e:
            print(f"[Calendar] safe_speak error: {e}")

    # ── Read handlers (worker thread) ───────────────────────────────────────

    def _cal_read_today(self) -> None:
        events = calendar_reminders.get_today_events()
        if not events:
            self._safe_speak("Your calendar is clear for today, Sir.")
            return
        lines = self._format_event_lines(events)
        prompt = (
            "These are the events on my calendar for today. Summarize them "
            "naturally and conversationally in Jarvis's voice — not a list, "
            "not a script, just how a sharp assistant would say it out loud. "
            "Keep it to two to four sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(self._JARVIS_CAL_SYSTEM, prompt, max_tokens=220)
        if text:
            self._safe_speak(text)
        else:
            self._safe_speak("I had trouble summarizing your events, Sir.")

    def _cal_read_upcoming(self) -> None:
        events = calendar_reminders.get_upcoming_events()
        if not events:
            self._safe_speak("Nothing on the books for the rest of the week, Sir.")
            return
        lines = self._format_event_lines(events)
        prompt = (
            "These are the events on my calendar between now and the end of "
            "this coming Saturday. Summarize them naturally and conversationally "
            "in Jarvis's voice — not a list, not a script, just how a sharp "
            "assistant would say it out loud.\n\n"
            "CRITICAL RULES:\n"
            "- Read the day of week for each event EXACTLY as given in the "
            "input. If the input says Saturday, say Saturday — do NOT say "
            "Wednesday or Monday or any other day.\n"
            "- Read each time EXACTLY as written. Do not change AM to PM.\n"
            "- Do not invent events that aren't in the input.\n"
            "- Keep it to three to five sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(self._JARVIS_CAL_SYSTEM, prompt, max_tokens=260)
        if text:
            self._safe_speak(text)
        else:
            # Deterministic template fallback — never hallucinates days/times.
            parts = [f"You have {len(events)} event" + ("s" if len(events) != 1 else "") + " this week, Sir."]
            for e in events:
                segment = e.get("title", "Untitled")
                if e.get("start"):
                    segment += f" on {e['start']}"
                if e.get("location"):
                    segment += f" at {e['location']}"
                parts.append(segment + ".")
            self._safe_speak(" ".join(parts))

    def _cal_read_reminders(self) -> None:
        reminders = calendar_reminders.get_all_reminders()
        if not reminders:
            self._safe_speak("You have no open reminders, Sir.")
            return
        # The reminders list is already sorted earliest-due-first by
        # calendar_reminders.get_all_reminders() — we just need to make
        # sure the LLM preserves that order when it phrases the summary.
        lines = []
        for idx, r in enumerate(reminders, start=1):
            parts = [f"{idx}. {r.get('title', '')}"]
            if r.get("due"):
                parts.append(f"(due {r['due']})")
            lines.append(" ".join(parts))
        prompt = (
            "These are the user's open reminders, listed in chronological "
            "order — earliest due date first. Summarize them naturally and "
            "conversationally in Jarvis's voice — not a list, not a script, "
            "just how a sharp assistant would say them out loud.\n\n"
            "CRITICAL RULES:\n"
            "- Mention the reminders in the exact order given below. Do not rearrange.\n"
            "- Read each due date and time EXACTLY as written in the input. Do not "
            "change AM to PM or vice versa. Do not change the hour. Do not round. "
            "'8 PM' means 8 PM, not 9 AM or 8 AM.\n"
            "- Keep it brief, two to four sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(self._JARVIS_CAL_SYSTEM, prompt, max_tokens=220)
        if text:
            self._safe_speak(text)
        else:
            # Template fallback — deterministic, never hallucinates times.
            if len(reminders) == 1:
                r = reminders[0]
                if r.get("due"):
                    self._safe_speak(f"You have one open reminder, Sir: {r['title']}, due {r['due']}.")
                else:
                    self._safe_speak(f"You have one open reminder, Sir: {r['title']}.")
            else:
                parts = [f"You have {len(reminders)} open reminders, Sir."]
                for r in reminders:
                    if r.get("due"):
                        parts.append(f"{r['title']} is due {r['due']}.")
                    else:
                        parts.append(f"{r['title']}.")
                self._safe_speak(" ".join(parts))

    def _format_event_lines(self, events: list) -> list:
        lines = []
        for e in events:
            parts = [f"- {e.get('title', '')}"]
            if e.get("start"):
                parts.append(f"starts {e['start']}")
            if e.get("end"):
                parts.append(f"ends {e['end']}")
            if e.get("location"):
                parts.append(f"at {e['location']}")
            if e.get("calendar"):
                parts.append(f"[{e['calendar']}]")
            lines.append(" ".join(parts))
        return lines

    # ── Create handlers (worker thread) ─────────────────────────────────────

    def _cal_create_event(self, user_input: str) -> None:
        data_json = self._extract_event_json(user_input)
        if not data_json:
            self._safe_speak(
                "I didn't quite catch that, Sir. Could you say it again a bit more clearly?"
            )
            return

        # DO NOT trust the LLM's `is_reminder` flag here. Our regex-based
        # intent detection already decided this is a calendar event (it
        # matched shift-style phrases with an explicit time range, or a
        # calendar/event/meeting keyword). Letting the LLM flip it to a
        # reminder after the fact was exactly the bug where "I'm working
        # tomorrow from 7 to 5" created a Reminder instead of a Calendar
        # event. The LLM doesn't know the intent routing already happened.

        # Sanitize LLM output: sometimes the model returns the STRING "null"
        # instead of real JSON null, which would then be used as a literal
        # event location / note / whatever. Normalize those to None.
        def _clean_optional(value):
            if value is None:
                return None
            s = str(value).strip()
            if s.lower() in ("null", "none", "n/a", "na", "nil", ""):
                return None
            return s

        title = (data_json.get("title") or "New event").strip()
        date_field = data_json.get("date") or "today"
        start_time = data_json.get("start_time")
        end_time = data_json.get("end_time")
        location = _clean_optional(data_json.get("location"))
        notes = _clean_optional(data_json.get("notes"))

        # Title sanitation. Strip a leading "Jarvis" or day name if the LLM
        # let it slip in despite the prompt's warnings. Also reject pure
        # day-name titles like "Saturday".
        _title_cleaned = re.sub(
            r"^\s*(?:jarvis|hey\s+jarvis|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*[,:\-]?\s*",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()
        if _title_cleaned:
            title = _title_cleaned
        if title.lower() in (
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday", "jarvis", "",
        ):
            # Fallback title based on the user's raw utterance
            if re.search(r"\b(?:working|shift|work)\b", user_input.lower()):
                title = "Work"
            elif re.search(r"\b(?:meeting|1:1|stand\s*up|standup)\b", user_input.lower()):
                title = "Meeting"
            elif re.search(r"\b(?:dentist|doctor|appointment)\b", user_input.lower()):
                title = "Appointment"
            else:
                title = "Event"

        # Date correction. The user's explicit weekday ALWAYS wins over
        # whatever the LLM extracted — STT transcribes weekday names
        # reliably (they're short and distinctive) and the LLM sometimes
        # hallucinates a different day entirely. This fixes BOTH the
        # "LLM said today instead of Saturday" bug AND the "LLM said
        # Friday instead of Saturday" bug.
        _weekday_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            user_input.lower(),
        )
        _has_today_phrase = bool(re.search(
            r"\b(?:today|right\s+now|this\s+morning|this\s+afternoon|"
            r"this\s+evening|tonight)\b",
            user_input.lower(),
        ))
        if _weekday_match and not _has_today_phrase:
            user_weekday = _weekday_match.group(1)
            llm_date_lower = str(date_field).lower().strip()
            # The LLM's extraction might already match in two ways: the
            # weekday name appears in the string, OR the extracted ISO
            # date falls on the user's stated weekday.
            llm_already_matches = user_weekday in llm_date_lower
            if not llm_already_matches:
                try:
                    _llm_dt = datetime.date.fromisoformat(llm_date_lower)
                    if _llm_dt.strftime("%A").lower() == user_weekday:
                        llm_already_matches = True
                except ValueError:
                    pass
            if not llm_already_matches:
                print(
                    f"[Calendar] User said {user_weekday!r} but LLM "
                    f"extracted {date_field!r} — overriding to "
                    f"{user_weekday!r}"
                )
                date_field = user_weekday

        resolved_date = self._resolve_relative_date(str(date_field))
        st = self._parse_time(start_time) or (9, 0)
        start_dt = datetime.datetime.combine(
            resolved_date, datetime.time(st[0], st[1])
        )
        end_dt = None
        et = self._parse_time(end_time)
        if et is not None:
            end_dt = datetime.datetime.combine(
                resolved_date, datetime.time(et[0], et[1])
            )

        cal_name = calendar_reminders.classify_calendar(user_input)

        data = {
            "title": title,
            "calendar": cal_name,
            "start_datetime": start_dt,
            "end_datetime": end_dt,
            "location": location,
            "notes": notes,
        }

        # Memory-backed location logic. Only ask when the utterance
        # signals a location-required event type AND no memory match.
        if not location:
            inferred = self._infer_location_memory_key(user_input)
            if inferred is not None:
                mem_key, question = inferred
                stored = self._get_fact(mem_key)
                if stored:
                    data["location"] = stored
                else:
                    # Set pending state and ask the question. The next
                    # user utterance will be routed to
                    # _resume_pending_calendar_action by the main loop.
                    self._pending_calendar_action = {
                        "type": "create_event",
                        "data": data,
                        "waiting_for": "location",
                        "memory_key": mem_key,
                    }
                    self._safe_speak(question)
                    return

        self._finalize_create_event(data)

    def _finalize_create_event(self, data: dict) -> None:
        """Actually create the event in Calendar.app + speak confirmation."""
        try:
            start_dt = data["start_datetime"]
            end_dt = data.get("end_datetime")
            assumed_end = False
            if end_dt is None:
                end_dt = start_dt + datetime.timedelta(hours=1)
                assumed_end = True

            calendar_reminders.create_calendar_event(
                title=data["title"],
                calendar_name=data["calendar"],
                start_datetime=start_dt,
                end_datetime=end_dt,
                location=data.get("location"),
                notes=data.get("notes"),
            )
        except Exception as e:
            print(f"[Calendar] create_calendar_event error: {e}")
            self._safe_speak(
                "I wasn't able to create that event, Sir — "
                "you may need to grant calendar permission in System Settings."
            )
            return

        # Build the confirmation. Template path is 4-6 seconds faster than
        # the LLM path because it skips the generation step entirely —
        # template text goes straight to TTS. See NATURAL_CALENDAR_CONFIRMATIONS
        # at the top of the file to switch back to the LLM path.
        # format_datetime_for_speech / format_time_for_speech drop ":00"
        # when minutes are zero so Jarvis says "7 PM" not "7:00 PM".
        when = calendar_reminders.format_datetime_for_speech(start_dt)
        end_str = calendar_reminders.format_time_for_speech(end_dt)
        loc_phrase = f" at {data['location']}" if data.get("location") else ""

        if NATURAL_CALENDAR_CONFIRMATIONS:
            end_clause = (
                f" The end time was assumed as a 1-hour default ({end_str})."
                if assumed_end else f" It ends at {end_str}."
            )
            action_desc = (
                f"Created an event titled '{data['title']}' on {when}{loc_phrase} "
                f"in the {data['calendar']} calendar.{end_clause}"
            )
            confirm_prompt = (
                "Confirm this calendar action naturally and conversationally — "
                "not scripted, not robotic. If an end time was assumed as a "
                "1-hour default, mention it briefly and offer to change it. "
                f"Action: {action_desc} "
                "Keep it to two to three sentences max."
            )
            text = self._llm_silent(
                self._JARVIS_CAL_SYSTEM, confirm_prompt, max_tokens=160
            )
            if text:
                self._safe_speak(text)
                return
            # LLM failed — fall through to the template.

        # Fast template path.
        if assumed_end:
            text = (
                f"Done, Sir. I've added {data['title']} on {when}{loc_phrase} "
                f"to your {data['calendar']} calendar. I assumed a one-hour "
                f"duration ending at {end_str} — let me know if you'd like to change it."
            )
        else:
            text = (
                f"Done, Sir. Your {data['title']} is scheduled for {when}"
                f"{loc_phrase}, ending at {end_str}, on your {data['calendar']} calendar."
            )
        self._safe_speak(text)

    def _cal_create_reminder(self, user_input: str) -> None:
        data_json = self._extract_event_json(user_input)
        if not data_json:
            self._safe_speak(
                "I didn't quite catch that, Sir. Could you say it again?"
            )
            return

        # The LLM sometimes returns the STRING "null" instead of JSON null
        # for fields it doesn't have a value for. Without filtering, that
        # string lands in the reminder's body and shows up as a literal
        # "null" under the reminder title in Reminders.app — exactly what
        # Nicholas saw on his "Test the reschedule" reminder.
        def _clean_optional(value):
            if value is None:
                return None
            s = str(value).strip()
            if s.lower() in ("null", "none", "n/a", "na", "nil", ""):
                return None
            return s

        title = (data_json.get("title") or "Reminder").strip()
        notes = _clean_optional(data_json.get("notes"))
        date_field = _clean_optional(data_json.get("date"))
        start_time = _clean_optional(data_json.get("start_time"))

        due_dt: Optional[datetime.datetime] = None
        if date_field or start_time:
            resolved_date = self._resolve_relative_date(
                str(date_field or "today")
            )
            st = self._parse_time(start_time) or (9, 0)
            due_dt = datetime.datetime.combine(
                resolved_date, datetime.time(st[0], st[1])
            )

        try:
            calendar_reminders.create_reminder(
                title=title, due_datetime=due_dt, notes=notes
            )
        except Exception as e:
            print(f"[Calendar] create_reminder error: {e}")
            self._safe_speak(
                "I wasn't able to create that reminder, Sir — "
                "you may need to grant reminders permission in System Settings."
            )
            return

        if NATURAL_CALENDAR_CONFIRMATIONS:
            if due_dt:
                when = calendar_reminders.format_datetime_for_speech(due_dt)
                action_desc = f"Created a reminder '{title}' due {when}."
            else:
                action_desc = f"Created a reminder '{title}' with no specific due date."
            confirm_prompt = (
                "Confirm this reminder action naturally and conversationally — "
                "not scripted, not robotic. "
                f"Action: {action_desc} "
                "Keep it to one to two sentences max."
            )
            text = self._llm_silent(
                self._JARVIS_CAL_SYSTEM, confirm_prompt, max_tokens=140
            )
            if text:
                self._safe_speak(text)
                return

        # Fast template path.
        if due_dt:
            when = calendar_reminders.format_datetime_for_speech(due_dt)
            text = f"Done, Sir. Reminder set: {title}, due {when}."
        else:
            text = f"Done, Sir. I've added a reminder to {title}."
        self._safe_speak(text)

    # ── Edit / delete / complete handlers ──────────────────────────────────

    # Words that appear in the intent verb or surrounding phrasing and should
    # be stripped before we use the remaining text as a fuzzy match target.
    _TITLE_STRIP_VERBS = (
        # STT mishears the wake word inside utterances — e.g. "Jervis please
        # delete..." — strip those variants so they don't leak into the
        # fuzzy match and throw off scoring.
        "jarvis", "jervis", "jarvus", "jarbis", "jarves", "javis", "jovis",
        "can you", "could you", "please", "would you",
        "complete", "completed", "finish", "finished", "mark",
        "check off", "check",
        "delete", "deleting", "remove", "removing", "cancel", "cancelling",
        "canceled", "cancelled", "drop", "get rid of", "throw out", "dismiss",
        "reschedule", "move", "change", "update", "edit", "rename",
        "the", "my", "a", "an", "that", "this",
        "for me", "as done", "as complete", "as completed", "as finished",
        "reminder", "event", "meeting", "appointment", "shift",
        "from my calendar", "off my calendar", "on my calendar",
    )

    def _extract_target_hint(self, user_input: str) -> str:
        """Return the 'payload' of an edit/delete/complete command with the
        verb words and item-type words stripped away.
        "complete the Amazon smartwatch reminder for today at 10 a.m." ->
        "Amazon smartwatch"."""
        t = " " + user_input.lower().strip() + " "
        # Strip command-verb / filler phrases
        for phrase in self._TITLE_STRIP_VERBS:
            t = re.sub(r"\s" + re.escape(phrase) + r"\s", " ", t)
        # Strip trailing time-phrase qualifiers AND everything after (they're
        # always at the end). Intentionally does NOT cover bare weekdays
        # because the weekday can appear MID-sentence before the title —
        # e.g. "delete the Monday work event" where "work event" is the
        # real title. Using `.*` on weekdays ate the title in an earlier
        # version of this code.
        t = re.sub(
            r"\s+(?:for\s+)?(?:today|tomorrow|tonight|"
            r"this\s+(?:morning|afternoon|evening|week)|next\s+\w+)\b.*",
            " ", t,
        )
        # Strip bare weekday names without eating surrounding text. Delete
        # handlers extract the weekday separately via their own regex and
        # pass it as a date_hint, so we don't need it in the fuzzy hint.
        t = re.sub(
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            " ", t,
        )
        t = re.sub(r"\s+at\s+\d[\w:.\s]*", " ", t)
        t = re.sub(r"\s+from\s+\d[\w:.\s]*", " ", t)
        # Collapse whitespace + trim stray punctuation
        t = re.sub(r"[^\w\s'-]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _cal_complete_reminder(self, user_input: str) -> None:
        hint = self._extract_target_hint(user_input)
        if not hint:
            self._safe_speak("Which reminder would you like me to complete, Sir?")
            return
        success, msg = calendar_reminders.complete_reminder(hint)
        if success:
            self._safe_speak(f"Done, Sir. I've completed the {msg} reminder.")
        else:
            self._safe_speak(
                f"I couldn't find a reminder matching {hint!r}, Sir."
            )

    def _cal_delete_reminder(self, user_input: str) -> None:
        hint = self._extract_target_hint(user_input)
        if not hint:
            self._safe_speak("Which reminder would you like me to delete, Sir?")
            return
        success, msg = calendar_reminders.delete_reminder(hint)
        if success:
            self._safe_speak(f"Deleted the {msg} reminder, Sir.")
        else:
            self._safe_speak(
                f"I couldn't find a reminder matching {hint!r}, Sir."
            )

    def _cal_update_reminder(self, user_input: str) -> None:
        """Update a reminder's title, due date/time, or notes via voice.
        Uses the dedicated `_extract_update_json` prompt (NOT the general
        event extractor) so the user's identifier for the reminder doesn't
        get mistakenly interpreted as a new title."""
        # ── Rename fast-path ──────────────────────────────────────────────
        # "rename X to Y" has an unambiguous structure — X is the target
        # hint, Y is the new title, split on " to ". The 3B LLM gets this
        # wrong when Y is long and syntactically continues X (e.g. Nicholas
        # said "rename GPT subscription reminder to consider canceling GPT
        # subscription" and the LLM packed BOTH halves into new_title).
        # Regex handles this deterministically and never hallucinates.
        _rename = re.match(
            r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
            r"rename\s+(?:the\s+)?(?P<old>.+?)\s+to\s+(?P<new>.+?)\s*[.!?]?\s*$",
            user_input.strip(),
            re.IGNORECASE,
        )
        if _rename:
            old_hint = _rename.group("old").strip()
            new_title = _rename.group("new").strip().rstrip(".,!?")
            # Strip "reminder" suffix from the old-title hint.
            old_hint = re.sub(
                r"\s+reminder\s*$", "", old_hint, flags=re.IGNORECASE
            ).strip()
            print(
                f"[Calendar] rename fast-path: old={old_hint!r} -> new={new_title!r}"
            )
            success, msg = calendar_reminders.update_reminder(
                old_hint, new_title=new_title
            )
            if success:
                self._safe_speak(
                    f"Renamed the {msg} reminder to {new_title}, Sir."
                )
            else:
                self._safe_speak(
                    f"I couldn't find a reminder matching {old_hint!r}, Sir."
                )
            return

        # ── General update path (reschedule, change notes, etc.) ──────────
        hint = self._extract_target_hint(user_input)
        print(f"[Calendar] update_reminder hint={hint!r}")
        if not hint:
            self._safe_speak(
                "Which reminder would you like me to update, Sir?"
            )
            return

        data_json = self._extract_update_json(user_input) or {}
        print(f"[Calendar] update_reminder LLM extraction: {data_json!r}")

        # Sanitize string fields — LLM sometimes returns "null" string
        def _clean_optional(value):
            if value is None:
                return None
            s = str(value).strip()
            if s.lower() in ("null", "none", "n/a", "na", "nil", ""):
                return None
            return s

        def _pick(*names):
            """Look up a value under any of several possible field names.
            The LLM doesn't always honor the `new_` prefix we asked for."""
            for n in names:
                v = data_json.get(n)
                if v is not None:
                    cleaned = _clean_optional(v)
                    if cleaned is not None:
                        return cleaned
            return None

        # Be tolerant of common alternate field names the LLM might use.
        new_title = _pick("new_title", "title", "rename_to", "name")
        new_date_field = _pick("new_date", "date", "due_date")
        new_time_field = _pick(
            "new_time", "time", "due_time", "start_time", "at_time"
        )
        new_notes = _pick("new_notes", "notes", "note", "body")

        # Safety net: only accept new_title if the user's utterance
        # contains an EXPLICIT rename verb. Without this check, the LLM
        # often invents a new title during reschedule operations — e.g.
        # Nicholas said "pick up groceries reminder to Sunday at 10 AM"
        # (no rename verb) and the LLM still returned new_title='Grocery'.
        # A real rename would have gone through the rename fast-path above,
        # so if we're here and there's no rename/call-it/retitle verb,
        # any title the LLM returned is hallucinated.
        _user_lower = user_input.lower()
        _has_rename_verb = bool(re.search(
            r"\b(?:rename|re[-\s]?title|call\s+it|change\s+(?:the\s+)?(?:name|title))\b",
            _user_lower,
        ))
        if new_title and not _has_rename_verb:
            print(
                f"[Calendar] nulling new_title (no rename verb in utterance): "
                f"{new_title!r}"
            )
            new_title = None
        elif new_title and new_title.lower().strip() == hint.lower().strip():
            # Original safety net kept for the rename-but-title-matches-hint edge
            print(f"[Calendar] nulling new_title (matches hint): {new_title!r}")
            new_title = None

        # Build the new due datetime. The user might change date only,
        # time only, or both. If only date or only time, use today or 9 AM
        # as a sensible default for the missing half.
        new_due: Optional[datetime.datetime] = None
        if new_date_field or new_time_field:
            if new_date_field:
                resolved_date = self._resolve_relative_date(new_date_field)
            else:
                resolved_date = datetime.date.today()
            parsed_time = self._parse_time(new_time_field)
            if parsed_time is None:
                # Either time wasn't provided, or _parse_time couldn't
                # handle the format. Default to 9 AM.
                parsed_time = (9, 0)
            new_due = datetime.datetime.combine(
                resolved_date, datetime.time(parsed_time[0], parsed_time[1])
            )

        print(
            f"[Calendar] update_reminder parsed: "
            f"new_title={new_title!r}, new_date={new_date_field!r}, "
            f"new_time={new_time_field!r}, new_due={new_due!r}, "
            f"new_notes={new_notes!r}"
        )

        if new_title is None and new_due is None and new_notes is None:
            self._safe_speak(
                "I heard you wanted to update a reminder, Sir, but I "
                "didn't catch what to change. Try saying for example, "
                "'reschedule the groceries reminder to tomorrow at 6 PM'."
            )
            return

        success, msg = calendar_reminders.update_reminder(
            hint, new_title=new_title, new_due=new_due, new_notes=new_notes,
        )
        print(f"[Calendar] update_reminder result: success={success}, msg={msg!r}")
        if not success:
            self._safe_speak(
                f"I couldn't find a reminder matching {hint!r}, Sir."
            )
            return

        parts = [f"Updated the {msg} reminder"]
        if new_title:
            parts.append(f"— renamed to {new_title}")
        if new_due:
            when = calendar_reminders.format_datetime_for_speech(new_due)
            parts.append(f"— due {when}")
        if new_notes:
            parts.append("— notes updated")
        self._safe_speak(" ".join(parts) + ", Sir.")

    def _cal_delete_event(self, user_input: str) -> None:
        hint = self._extract_target_hint(user_input)
        if not hint:
            self._safe_speak("Which event would you like me to delete, Sir?")
            return

        # If the user named a day, narrow the event search window.
        # Direct regex extraction (not LLM) — fast, deterministic, and
        # always respects the user's explicit day. Without this, a
        # "delete the Monday work event" command could match a Saturday
        # "Work" event if the LLM fails to extract the weekday via the
        # event-creation prompt (which was the Saturday-deleted bug).
        date_hint: Optional[datetime.date] = None
        _day_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"tomorrow|today|tonight)\b",
            user_input.lower(),
        )
        if _day_match:
            try:
                date_hint = self._resolve_relative_date(_day_match.group(1))
                print(
                    f"[Calendar] delete_event: regex-extracted day "
                    f"{_day_match.group(1)!r} -> {date_hint}"
                )
            except Exception:
                date_hint = None

        success, msg = calendar_reminders.delete_calendar_event(
            hint, date_hint=date_hint
        )
        if success:
            self._safe_speak(f"Deleted the {msg} event from your calendar, Sir.")
        else:
            self._safe_speak(
                f"I couldn't find an event matching {hint!r}, Sir."
            )

    # ── Pending clarification resume ────────────────────────────────────────

    def _resume_pending_calendar_action(self, answer: str) -> None:
        """Called by the main loop (not the worker) when a clarifying
        question is pending and the user has just answered it. Starts a
        new background worker to finish the action."""
        pending = self._pending_calendar_action
        self._pending_calendar_action = None
        if not pending:
            return

        waiting_for = pending.get("waiting_for")
        data = pending.get("data", {})
        atype = pending.get("type")

        if waiting_for != "location" or atype != "create_event":
            self._safe_speak("I'm not sure what to do with that, Sir.")
            return

        loc = (answer or "").strip().rstrip(".?!,;:")
        if not loc:
            self._safe_speak("I didn't catch a location, Sir. Try again.")
            return

        # Persist silently so we never have to ask again.
        mem_key = pending.get("memory_key")
        if mem_key:
            try:
                memory.save_fact(mem_key, loc)
                self._rebuild_system_prompt()
                print(f"[Calendar] Saved memory {mem_key} = {loc}")
            except Exception as e:
                print(f"[Calendar] save fact error: {e}")
        data["location"] = loc

        # Finish the event creation on a fresh background worker so the
        # main loop stays responsive — same safety contract as the
        # original dispatch path.
        self._cancel_conversation_timer()
        self._calendar_working.set()
        ws_server.set_state("thinking")

        def _finish():
            try:
                self._finalize_create_event(data)
            except Exception as e:
                logger.exception(f"[Calendar] resume finalize error: {e}")
                self._safe_speak(
                    "Something went wrong finishing that event, Sir — want to try again?"
                )
            finally:
                self._pending_calendar_action = None
                self._calendar_working.clear()
                try:
                    ws_server.set_state("idle")
                except Exception:
                    pass
                if self._in_conversation:
                    try:
                        self._start_conversation_timer()
                    except Exception:
                        pass

        threading.Thread(target=_finish, daemon=True,
                         name="calendar-resume").start()

    # ── File management ───────────────────────────────────────────────────
    #
    # Voice-driven file find/move/rename/describe. Jarvis always shows
    # the candidate file in the orb (or Quick Look, if the flag is
    # flipped) and confirms verbally before doing anything destructive.
    # The "describe" path is the exception — it's read-only and never
    # confirms.

    # Words Jarvis treats as confirmation when a file-action is pending.
    # Includes "please" by itself — a common polite single-word yes that
    # users give in response to "Is this the one you want to move?" and
    # that the previous list treated as a no.
    _FILE_YES_WORDS = (
        "yes", "yeah", "yep", "yup", "sure", "correct", "right", "that's it",
        "thats it", "do it", "go ahead", "confirm", "affirmative", "please do",
        "do it please", "please", "yes please", "please move", "please rename",
        "that one", "this one", "ok", "okay", "alright", "fine", "proceed",
        "absolutely", "definitely", "go for it", "by all means", "yup please",
    )
    _FILE_NO_WORDS = (
        "no", "nope", "nah", "wrong", "wrong file", "not that", "not that one",
        "not it", "cancel", "never mind", "nevermind", "stop", "abort",
        "don't", "dont", "don't do", "dont do",
    )

    # ── Follow-up context ──────────────────────────────────────────
    # A just-completed move/rename leaves _last_file_action set.
    # Within this window, pronoun references like "move that to
    # Documents" resolve back to the file we just touched, so the
    # user doesn't have to name it again.
    _FILE_FOLLOWUP_WINDOW_S = 180.0  # 3 minutes

    # Phrases that indicate the user is referring to a previously
    # handled file by pronoun rather than naming a new one.
    _FILE_PRONOUN_RE = re.compile(
        r"\b(?:it|that|this|them|those|the\s+(?:same\s+)?(?:file|one|document|doc|image|photo|picture)|"
        r"the\s+rmv|same\s+(?:file|one))\b"
    )

    def _has_recent_file_action(self) -> bool:
        lfa = self._last_file_action
        if not lfa:
            return False
        ts = lfa.get("timestamp", 0) or 0
        if (time.time() - ts) >= self._FILE_FOLLOWUP_WINDOW_S:
            self._last_file_action = None
            return False
        # Also verify the file is still where we last left it.
        path = lfa.get("path") or ""
        if not path or not os.path.exists(path):
            self._last_file_action = None
            return False
        return True

    def _utterance_is_pronoun_ref(self, text: str) -> bool:
        return bool(self._FILE_PRONOUN_RE.search((text or "").lower()))

    def _record_file_action(self, action: str, new_path: str,
                            original_path: str) -> None:
        self._last_file_action = {
            "action": action,
            "path": new_path,
            "original_path": original_path,
            "timestamp": time.time(),
        }

    # ── Screen awareness ────────────────────────────────────────────────────
    _SCREEN_DESCRIBE_PATTERNS = (
        r"\bwhat(?:'?s|\s+is)\s+on\s+(?:my\s+)?screen\b",
        r"\bwhat(?:'?s|\s+is)\s+on\s+(?:my\s+)?display\b",
        r"\bwhat\s+do\s+you\s+see\b",
        r"\bdescribe\s+(?:my\s+)?screen\b",
        r"\bwhat(?:'?s|\s+is)\s+open\b",
        r"\blook\s+at\s+(?:my\s+)?screen\b",
        r"\bwhat\s+am\s+i\s+looking\s+at\b",
        r"\bcan\s+you\s+see\s+(?:my\s+)?screen\b",
    )

    def _detect_screen_intent(self, text: str) -> Optional[str]:
        """Return 'screen_describe' if the user is asking about what's on
        their screen, else None. Strict regex only — casual mentions of
        'screen' shouldn't trigger a capture."""
        t = (text or "").lower().strip()
        if not t:
            return None
        for pat in self._SCREEN_DESCRIBE_PATTERNS:
            if re.search(pat, t):
                return "screen_describe"
        return None

    def _push_screen_preview(self) -> None:
        try:
            ws_server.send_event({
                "type": "screen_preview",
                "image_url": "http://localhost:3000/preview_screen",
                "cache_buster": int(time.time() * 1000),
            })
        except Exception as e:
            print(f"[Screen] push preview error: {e}")

    def _clear_screen_preview(self) -> None:
        try:
            ws_server.send_event({"type": "screen_preview_clear"})
        except Exception as e:
            print(f"[Screen] clear preview error: {e}")

    def _handle_screen_command(self, intent: str, user_input: str) -> None:
        """Kick off screen awareness on a background thread so the voice
        pipeline stays responsive. Announces briefly, captures, optionally
        previews in the orb, runs Moondream under the LLM lock, cleans up,
        then speaks the description."""
        worker = threading.Thread(
            target=self._screen_worker_body,
            args=(intent, user_input),
            daemon=True,
            name="screen-worker",
        )
        worker.start()

    def _screen_worker_body(self, intent: str, user_input: str) -> None:
        try:
            self._safe_speak("Let me take a look.")

            # Capture first so the orb preview has something to show.
            path = screen_awareness.capture_screen()
            if not os.path.isfile(path):
                self._safe_speak(
                    "I couldn't capture the screen, Sir. You may need to "
                    "grant Screen Recording permission in System Settings."
                )
                return

            orb_mode = screen_awareness.is_orb_mode()
            if orb_mode:
                self._push_screen_preview()

            # Warn the user if we're about to trigger a first-run download
            # or the warm-up hasn't finished loading the model yet.
            try:
                model_file = (
                    screen_awareness._MOONDREAM_CACHE_DIR
                    / screen_awareness._MOONDREAM_MODEL_FILENAME
                )
                first_run = not model_file.is_file()
            except Exception:
                first_run = False
            if first_run and screen_awareness._vision_model is None:
                self._safe_speak(
                    "Give me a moment — loading the vision model for the first time."
                )

            # Moondream is an LLM-class load — serialize with the main LLM
            # lock so we don't crash llama-cpp by running two heavy models
            # on the GPU at once.
            try:
                with self._llm_lock:
                    description = screen_awareness.describe_screen(path)
            except Exception as e:
                print(f"[Screen] describe error: {e}")
                import traceback
                traceback.print_exc()
                description = ""

            screen_awareness.cleanup_screenshot()
            if orb_mode:
                self._clear_screen_preview()

            if description:
                self._safe_speak(description)
            else:
                self._safe_speak(
                    "I wasn't able to make sense of what's on your screen, Sir."
                )
        except Exception as e:
            print(f"[Screen] worker error: {e}")
            import traceback
            traceback.print_exc()
            try:
                screen_awareness.cleanup_screenshot()
            except Exception:
                pass
            if screen_awareness.is_orb_mode():
                self._clear_screen_preview()
            self._safe_speak(
                "I ran into a problem looking at your screen, Sir."
            )

    def _detect_file_intent(self, text: str) -> Optional[str]:
        """Return one of file_move / file_rename / file_describe /
        file_find, or None if no file-management intent is present."""
        t = (text or "").lower().strip()
        if not t:
            return None

        has_file_noun = bool(re.search(
            r"\b(file|files|document|documents|doc|docs|pdf|pdfs|image|images|"
            r"picture|pictures|photo|photos|spreadsheet|spreadsheets|"
            r"resume|resumes|presentation|presentations|slides|"
            r"note|notes|word\s+doc|word\s+document)\b", t
        ))
        has_file_ext = bool(re.search(
            r"\.(?:pdf|docx?|txt|md|png|jpe?g|gif|heic|tiff|bmp|csv|xlsx?|pptx?)\b",
            t,
        ))
        # Pronoun reference counts as a file reference ONLY when we have
        # a recent file action to resolve it against. Otherwise "move
        # that to the left" would false-positive.
        has_followup_pronoun = (
            self._has_recent_file_action()
            and self._utterance_is_pronoun_ref(t)
        )
        file_ref = has_file_noun or has_file_ext or has_followup_pronoun

        # Rename — strong, unambiguous phrasing. Allow with or without
        # "file"/"document" because "rename X to Y" is itself a clear
        # file operation. Catch past/progressive tenses so a follow-up
        # like "actually rename it to X" still routes correctly.
        if re.search(r"\b(?:rename|renamed|renaming|renames)\b", t) or \
           re.search(r"\bchange\s+the\s+name\s+of\b", t):
            return "file_rename"

        # Move / put / transfer / send / take / stick a file to a location.
        # All tenses — STT sometimes transcribes "move" as "moved", and
        # the user may naturally say "actually moved that to Documents"
        # intending an imperative. file_ref guards against false
        # positives from unrelated sentences.
        _MOVE_VERBS = (
            r"\b(?:"
            r"move[sd]?|moving|"
            r"put(?:s|ting)?|"
            r"transfer(?:s|red|ring)?|"
            r"send(?:s|ing)?|sent|"
            r"drop(?:s|ped|ping)?|"
            r"relocate[sd]?|relocating|"
            r"take[sn]?|taking|took|"
            r"stick(?:s|ing)?|stuck|"
            r"copy|copied|copying|copies"
            r")\b"
        )
        if file_ref and re.search(
            _MOVE_VERBS + r"[^.?!]*?\b(?:to|into|onto|on|in)\b",
            t,
        ):
            return "file_move"

        # Describe / read / summarize a file.
        if file_ref and (
            re.search(r"\b(?:summarize|summarise|summarizing|summarized|"
                      r"describe[sd]?|describing|"
                      r"read\s+(?:me\s+)?(?:through\s+)?(?:out\s+)?(?:aloud\s+)?)\b", t)
            or re.search(r"\bwhat(?:'?s|\s+is|\s+does)\s+(?:in|inside)\b", t)
            or re.search(r"\btell\s+me\s+what(?:'?s|\s+is)\s+(?:in|inside)\b", t)
            or re.search(r"\bopen\s+and\s+read\b", t)
        ):
            return "file_describe"

        # Find / locate a file.
        if re.search(r"\b(?:find|finds|finding|located?|locating|"
                     r"where\s+is|where(?:'?s)?)\b", t) and file_ref:
            return "file_find"
        if re.search(r"\b(?:can\s+you\s+)?find\s+(?:me\s+)?(?:my\s+|the\s+)?", t) and file_ref:
            return "file_find"

        return None

    # Common spoken location names → the key used by resolve_destination.
    # Only words we're 100% sure refer to a standard user folder go here;
    # anything more exotic stays with the LLM.
    _DEST_REGEX = re.compile(
        r"\b(?:to|into|onto)\s+(?:my\s+|the\s+|a\s+)?"
        r"(desktop|downloads?|documents?|pictures?|photos?|"
        r"music|movies?|videos?|home)"
        r"(?:\s+folder|\s+directory)?\b"
    )

    def _parse_destination_from_utterance(self, utterance: str) -> Optional[str]:
        """Deterministic destination extraction. Strips out 'from X' spans
        first (so 'from my downloads folder to documents' doesn't match
        'downloads'), then pulls the word after the FINAL to/into/onto
        preposition. Returns a canonical short label (e.g. 'downloads')
        that resolve_destination knows how to map."""
        t = (utterance or "").lower()
        # Remove "from <folder>" and "from my <folder>" phrases up front,
        # so the destination regex can only match a post-"to" word.
        cleaned = re.sub(
            r"\bfrom\s+(?:my\s+|the\s+)?\w+(?:\s+folder|\s+directory)?\b",
            "",
            t,
        )
        matches = self._DEST_REGEX.findall(cleaned)
        if not matches:
            return None
        # Prefer the last occurrence — "take it from Downloads to Desktop"
        # has both matched by the regex, and the destination is the last.
        return matches[-1]

    def _extract_file_json(self, utterance: str) -> Optional[dict]:
        """LLM-extracted details for a file-management utterance. Returns
        a dict with query / action / destination / new_name, or None."""
        prompt = (
            "Extract file management details from this voice request. "
            "Respond in JSON only — no explanation, no markdown.\n\n"
            "Fields:\n"
            "- query: JUST the distinctive filename or keyword(s) the "
            "user mentioned. Short — no filler words like 'file', "
            "'document', 'the', 'my'. Do NOT include where it is or "
            "what to do with it. If the user said a full filename, "
            "use that. Examples: 'resume', 'taxes 2024', "
            "'rmv-realid-application-steps'.\n"
            "- action: one of move / rename / describe / find\n"
            "- destination: a string OR null. This is where the file is "
            "going TO — the TARGET. It is the place named after 'to', "
            "'into', 'onto', 'on', or 'in'. It is NEVER the place named "
            "after 'from' — that is the source, which you must ignore. "
            f"Resolve spoken locations to absolute paths: "
            f"Desktop -> {os.path.expanduser('~/Desktop')}, "
            f"Documents -> {os.path.expanduser('~/Documents')}, "
            f"Downloads -> {os.path.expanduser('~/Downloads')}, "
            f"Pictures -> {os.path.expanduser('~/Pictures')}, "
            f"Music -> {os.path.expanduser('~/Music')}. "
            "For anything that isn't a move, set this to null.\n"
            "- new_name: string OR null. Only set when the user is "
            "renaming the file; otherwise null.\n\n"
            "Examples:\n"
            "  'move the rmv file from my desktop to my documents folder' -> "
            "{\"query\": \"rmv\", \"action\": \"move\", "
            f"\"destination\": \"{os.path.expanduser('~/Documents')}\", "
            "\"new_name\": null}\n"
            "  'move the rmv file from my downloads folder into the documents folder' -> "
            "{\"query\": \"rmv\", \"action\": \"move\", "
            f"\"destination\": \"{os.path.expanduser('~/Documents')}\", "
            "\"new_name\": null}\n"
            "  'rename the groceries file to shopping list' -> "
            "{\"query\": \"groceries\", \"action\": \"rename\", "
            "\"destination\": null, \"new_name\": \"shopping list\"}\n"
            "  'find my resume' -> "
            "{\"query\": \"resume\", \"action\": \"find\", "
            "\"destination\": null, \"new_name\": null}\n"
            "  'what is in my notes file' -> "
            "{\"query\": \"notes\", \"action\": \"describe\", "
            "\"destination\": null, \"new_name\": null}\n\n"
            f"User said: '{utterance}'"
        )
        raw = self._llm_silent(
            "You are a precise JSON extraction tool. Reply with valid JSON only.",
            prompt,
            max_tokens=180,
            temperature=0.1,
        )
        if not raw:
            return None
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            return json.loads(raw)
        except Exception as e:
            print(f"[File] JSON parse error: {e}  — raw: {raw!r}")
            return None

    def _push_file_preview(self, filepath: str, preview: dict) -> None:
        """Send a preview event to the orb over the WebSocket."""
        try:
            event = {
                "type": "file_preview",
                "filename": Path(filepath).name,
                "file_type": preview.get("file_type", "other"),
                "mode": preview.get("mode", "orb"),
                "serve_url": preview.get("serve_url"),
                "text_content": preview.get("text_content"),
                # Bump a nonce so the WebView re-fetches /preview_file
                # instead of using a cached response from a prior file.
                "cache_buster": int(time.time() * 1000),
            }
            ws_server.send_event(event)
        except Exception as e:
            print(f"[File] push preview error: {e}")

    def _clear_file_preview(self) -> None:
        """Tell the orb to hide the preview overlay."""
        try:
            ws_server.send_event({"type": "file_preview_clear"})
        except Exception as e:
            print(f"[File] clear preview error: {e}")

    def _handle_file_command(self, intent: str, user_input: str) -> None:
        """Top-level dispatch for a file-management utterance. Runs on the
        main loop thread — operations are fast (mdfind + preview prep).
        Any exception is caught and spoken so the pipeline never crashes."""
        try:
            data = self._extract_file_json(user_input) or {}
            print(f"[File] extract -> {data}")
            query = (data.get("query") or "").strip()
            destination = data.get("destination")
            new_name = data.get("new_name")

            # Deterministic destination override. The 3B LLM sometimes
            # confuses 'from X to Y' — returning X as destination instead
            # of Y. A regex that strips 'from <folder>' spans first and
            # then pulls the post-'to' word is more reliable for the
            # common user-folder vocabulary.
            regex_dest = self._parse_destination_from_utterance(user_input)
            if regex_dest:
                if destination and regex_dest.lower() not in str(destination).lower():
                    print(f"[File] destination override: LLM={destination!r} "
                          f"-> regex={regex_dest!r}")
                destination = regex_dest

            # Follow-up branch: user said "move that to X" referring to
            # the just-touched file. Skip search — we already know the
            # path from _last_file_action. This is the only way an
            # utterance with no explicit filename can still succeed.
            if (
                self._has_recent_file_action()
                and self._utterance_is_pronoun_ref(user_input)
            ):
                lfa_path = self._last_file_action["path"]
                print(f"[File] follow-up pronoun ref — using last action path {lfa_path!r}")
                matches = [lfa_path]
            else:
                if not query:
                    # Fall back to the full utterance if the LLM gave us nothing.
                    query = re.sub(
                        r"\b(?:please|jarvis|could\s+you|can\s+you|would\s+you)\b",
                        "",
                        user_input,
                        flags=re.IGNORECASE,
                    ).strip()

                print(f"[File] search query={query!r}")
                matches = file_manager.search_file(query)
            print(f"[File] matches={len(matches)}")
            for m in matches:
                print(f"[File]   - {m}")
            if not matches:
                # If the walk hit permission errors, Spotlight is probably
                # also TCC-filtered — tell the user clearly so they can
                # grant access once and move on, instead of thinking the
                # file isn't there.
                denied = file_manager.get_last_permission_errors()
                if denied:
                    print(f"[File] permission denied on: {denied}")
                    self.speak_direct(
                        "I can't see your files right now, Sir. "
                        "You'll need to grant Jarvis access to your folders "
                        "in System Settings under Privacy and Security, "
                        "then try again."
                    )
                else:
                    self.speak_direct(
                        "I couldn't find anything matching that, Sir. "
                        "Can you be more specific?"
                    )
                return

            if intent == "file_find":
                self._handle_file_find(matches)
                return

            if intent == "file_describe":
                self._handle_file_describe(matches[0])
                return

            # move / rename — require confirmation with preview.
            action = "move" if intent == "file_move" else "rename"
            resolved_dest = (
                file_manager.resolve_destination(destination)
                if action == "move" else None
            )
            print(f"[File] action={action} destination={destination!r} "
                  f"resolved={resolved_dest!r} new_name={new_name!r}")

            if action == "move" and not resolved_dest:
                self.speak_direct(
                    "I didn't catch where you wanted me to move it, Sir. "
                    "Try again with a destination."
                )
                return

            # Guard: if the resolved destination is the folder the file
            # already lives in, the LLM likely confused source and
            # destination (e.g. "from my downloads folder into documents"
            # being extracted as destination=Downloads). Refuse rather
            # than queue a no-op move that will later fail as
            # "file already exists".
            if action == "move" and matches and resolved_dest:
                current_dir = str(Path(matches[0]).parent)
                dest_dir = str(Path(resolved_dest).expanduser())
                if os.path.normpath(current_dir) == os.path.normpath(dest_dir):
                    self.speak_direct(
                        f"That file is already in your {self._dest_label(dest_dir)} "
                        "folder, Sir. Where would you like me to move it to?"
                    )
                    return

            if action == "rename" and not (new_name and str(new_name).strip()):
                self.speak_direct(
                    "I didn't catch what you wanted to rename it to, Sir."
                )
                return

            self._present_file_for_confirmation(
                action=action,
                candidates=matches,
                destination=resolved_dest,
                new_name=new_name,
            )
        except Exception as e:
            logger.exception(f"[File] handle command error: {e}")
            # Clear pending state on error so Jarvis isn't stuck waiting
            # for a confirmation to a question that never got asked.
            self._pending_file_action = None
            self._safe_speak(
                "Something went wrong with that file request, Sir — want to try again?"
            )

    def _handle_file_find(self, matches: list[str]) -> None:
        """Informational: show where the file is, no confirmation needed."""
        if len(matches) == 1:
            path = matches[0]
            preview = file_manager.prepare_preview(path)
            self._push_file_preview(path, preview)
            folder = str(Path(path).parent.name) or "your home folder"
            name = Path(path).name
            # Give the orb a few seconds to render, then clear on the
            # next user command. We don't auto-hide here — the user sees
            # it until they move on.
            self.speak_direct(
                f"I found {name} in your {folder} folder, Sir."
            )
            # Track the temp path so we can clean it up when the preview
            # is later dismissed or replaced.
            self._pending_file_action = {
                "action": "find",
                "original_path": path,
                "temp_preview_path": preview.get("temp_path"),
                "candidates": [],
                "waiting_for": None,
            }
            return

        # Multiple: list the top matches verbally.
        lines = []
        for i, path in enumerate(matches[:5], start=1):
            folder = str(Path(path).parent.name) or "home"
            lines.append(f"{i}. {Path(path).name} in {folder}")
        summary = "I found a few matches, Sir. " + ". ".join(lines) + "."
        self.speak_direct(summary)

    def _handle_file_describe(self, path: str) -> None:
        """Read the file and speak a natural summary. No confirmation —
        this is read-only."""
        preview = file_manager.prepare_preview(path)
        self._push_file_preview(path, preview)
        # Build text content for the LLM from whatever prepare_preview
        # gave us. For images / other, we just describe the metadata.
        content_for_llm = preview.get("text_content") or ""
        if not content_for_llm:
            ft = preview.get("file_type", "other")
            if ft == "pdf":
                content_for_llm = f"[A PDF file named {Path(path).name}. Content not extracted.]"
            elif ft == "image":
                content_for_llm = f"[An image file named {Path(path).name}.]"
            else:
                content_for_llm = f"[A file named {Path(path).name}.]"
        # Truncate so we don't blow the context window.
        if len(content_for_llm) > 6000:
            content_for_llm = content_for_llm[:6000] + "\n… (truncated)"

        prompt = (
            "The user asked what's in this file. Give a brief, natural, "
            "spoken summary of the content — two to four sentences, "
            "conversational, no bullet points, no markdown. If the file "
            "is empty or unreadable, say so briefly.\n\n"
            f"FILE: {Path(path).name}\n\nCONTENT:\n{content_for_llm}"
        )
        text = self._llm_silent(self._JARVIS_CAL_SYSTEM, prompt, max_tokens=260)
        if text:
            self._safe_speak(text)
        else:
            self._safe_speak(f"I wasn't able to summarize {Path(path).name}, Sir.")

        # Record so a follow-up utterance clears the preview cleanly.
        self._pending_file_action = {
            "action": "describe",
            "original_path": path,
            "temp_preview_path": preview.get("temp_path"),
            "candidates": [],
            "waiting_for": None,
        }

    def _present_file_for_confirmation(
        self,
        action: str,
        candidates: list[str],
        destination: Optional[str] = None,
        new_name: Optional[str] = None,
    ) -> None:
        """Prep the top candidate's preview, push it to the orb, stash
        the pending action, and ask for confirmation verbally."""
        if not candidates:
            self._safe_speak(
                "I couldn't find the right file, Sir. Try describing it differently."
            )
            return

        path = candidates[0]
        remaining = candidates[1:]
        preview = file_manager.prepare_preview(path)
        self._push_file_preview(path, preview)

        self._pending_file_action = {
            "action": action,
            "original_path": path,
            "destination": destination,
            "new_name": new_name,
            "temp_preview_path": preview.get("temp_path"),
            "candidates": remaining,
            "waiting_for": "confirmation",
        }

        folder = str(Path(path).parent.name) or "your home folder"
        name = Path(path).name
        if action == "move":
            dest_label = self._dest_label(destination or "")
            question = (
                f"I found {name} in your {folder} folder. "
                f"Is this the one you want to move to the {dest_label}, Sir?"
            )
        else:  # rename
            question = (
                f"I found {name} in your {folder} folder. "
                f"Should I rename this one to {new_name}, Sir?"
            )
        self.speak_direct(question)

    @staticmethod
    def _dest_label(destination: str) -> str:
        """Turn an absolute path into a short spoken label."""
        if not destination:
            return "destination"
        name = Path(destination).name
        return name or destination

    def _resume_pending_file_action(self, answer: str) -> None:
        """Process the user's yes/no for the most recent file confirmation.
        On no, walk through any remaining candidates."""
        pending = self._pending_file_action
        if not pending:
            return

        # Read-only pending states (find/describe) — any new utterance
        # just clears the preview; re-dispatch the utterance as a normal
        # command.
        if pending.get("waiting_for") is None:
            cleanup = pending.get("temp_preview_path")
            file_manager.cleanup_temp_preview(cleanup)
            self._clear_file_preview()
            self._pending_file_action = None
            # NOTE: caller handles re-dispatch; we just cleared state.
            return

        t = (answer or "").lower().strip().rstrip(".?!,;:")
        is_yes = any(re.search(rf"\b{re.escape(w)}\b", t) for w in self._FILE_YES_WORDS)
        is_no = any(re.search(rf"\b{re.escape(w)}\b", t) for w in self._FILE_NO_WORDS)

        if is_yes and not is_no:
            self._execute_pending_file_action(pending)
            return

        if is_no or not is_yes:
            # Treat anything that isn't an explicit yes as a no so we
            # don't silently destroy the wrong file.
            cleanup = pending.get("temp_preview_path")
            file_manager.cleanup_temp_preview(cleanup)
            self._pending_file_action = None

            remaining = pending.get("candidates") or []
            if remaining:
                self._present_file_for_confirmation(
                    action=pending["action"],
                    candidates=remaining,
                    destination=pending.get("destination"),
                    new_name=pending.get("new_name"),
                )
            else:
                self._clear_file_preview()
                self._safe_speak(
                    "I couldn't find the right file, Sir. "
                    "Try describing it differently."
                )

    def _execute_pending_file_action(self, pending: dict) -> None:
        """Run the actual move or rename, clean up temp, clear state."""
        action = pending.get("action")
        original_path = pending.get("original_path") or ""
        temp_path = pending.get("temp_preview_path")

        # Always clean up the temp preview first — we never want to
        # touch the original file by accident.
        file_manager.cleanup_temp_preview(temp_path)
        self._pending_file_action = None
        self._clear_file_preview()

        if action == "move":
            destination = pending.get("destination") or ""
            ok, msg = file_manager.move_file(original_path, destination)
            if ok:
                dest_label = self._dest_label(destination)
                # `msg` is the new path — record it so follow-up "move
                # that to X" utterances can find the file.
                self._record_file_action("move", msg, original_path)
                self._safe_speak(f"Done, Sir — moved it to the {dest_label}.")
            else:
                self._safe_speak(f"I wasn't able to move that file, Sir. {msg}.")
            return

        if action == "rename":
            new_name = pending.get("new_name") or ""
            ok, msg = file_manager.rename_file(original_path, str(new_name))
            if ok:
                self._record_file_action("rename", msg, original_path)
                self._safe_speak(f"Done, Sir — renamed to {Path(msg).name}.")
            else:
                self._safe_speak(f"I wasn't able to rename that file, Sir. {msg}.")
            return

        self._safe_speak("I'm not sure what to do with that, Sir.")

    # ── Spinner ───────────────────────────────────────────────────────────────

    @staticmethod
    def _spinner(stop: threading.Event) -> None:
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not stop.is_set():
            sys.stdout.write(f"\r  Thinking {frames[i % len(frames)]}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()

    # ── Turn (LLM pipeline) ───────────────────────────────────────────────────

    def _speak_memory_ack(self) -> None:
        """No-op. Auto-detected facts are saved completely silently — the user
        should never hear Jarvis narrate its own memory operations. Kept as a
        no-op so existing call sites (_pending_memory_ack branches) compile
        and do no work."""
        return

    def _summarize_old_exchanges(self) -> None:
        """Level 3 memory: find conversation rows outside the active 20-turn
        window that have not yet been summarized, batch them into groups of 10,
        and use a silent internal LLM call to produce a 2-3 sentence summary
        for each batch. Runs on a background thread from the wake-mode hook."""
        if self._summarizing:
            return
        self._summarizing = True
        # Capture + clear the "first run" flag at the very top so even an
        # early return (nothing to summarize yet) counts as "the startup
        # attempt has happened". Without this, a fresh session with no
        # prior conversations would hit the early return below, leave the
        # flag True, and then speak "updating my memory" on the user's
        # FIRST return-to-wake-mode — exactly the bug Nicholas reported.
        was_first_summarization = self._first_summarization
        self._first_summarization = False
        try:
            unsummarized = memory.get_unsummarized_exchanges()
            batches = memory.batch_conversations_for_summary(unsummarized)
            if not batches:
                return

            print(
                f"[Memory] Summarizing {len(unsummarized)} older "
                f"exchange rows in {len(batches)} batch(es)…"
            )

            # Background memory work is silent. `was_first_summarization`
            # is still tracked for future use but Jarvis no longer speaks
            # "updating my memory" on the initial run.
            _ = was_first_summarization

            for batch in batches:
                try:
                    conversation_text = "\n".join(
                        f"{row['role'].upper()}: {row['content']}" for row in batch
                    )
                    summary_prompt = (
                        "Below is a short excerpt from a conversation between an AI "
                        "assistant named Jarvis and his user Nicholas. Write a 2-3 "
                        "sentence factual summary of what was discussed or learned. "
                        "Focus on facts, preferences, plans, or anything personally "
                        "relevant to Nicholas. Write in third person. Do not include "
                        "filler or commentary. Do not start with 'In this conversation' "
                        "or 'The user'.\n\n"
                        + conversation_text
                    )

                    # Silent non-streaming LLM call — no TTS, no history mutation.
                    # Held under _llm_lock so we don't collide with a chat
                    # turn or calendar worker running at the same time.
                    with self._llm_lock:
                        result = self._llm.create_chat_completion(
                            messages=[
                                {"role": "system", "content": "You are a concise summarizer. Reply with only the summary."},
                                {"role": "user", "content": summary_prompt},
                            ],
                            max_tokens=160,
                            temperature=0.3,
                            top_p=0.9,
                            stop=["<|eot_id|>"],
                            stream=False,
                        )
                    summary_text = (
                        result["choices"][0]["message"]["content"] or ""
                    ).strip()
                    summary_text = _clean(summary_text)

                    if not summary_text:
                        print("[Memory] Empty summary — skipping batch.")
                        continue

                    ids = [row["id"] for row in batch]
                    timestamps = [row["timestamp"] for row in batch if row.get("timestamp")]
                    date_from = min(timestamps) if timestamps else ""
                    date_to = max(timestamps) if timestamps else ""

                    memory.save_summary(
                        summary_text=summary_text,
                        conversation_ids=ids,
                        date_from=date_from,
                        date_to=date_to,
                    )
                    print(
                        f"[Memory] Summarized batch of {len(batch)} exchanges: "
                        f"{summary_text[:80]}{'…' if len(summary_text) > 80 else ''}"
                    )
                except Exception as e:
                    logger.error(f"[Memory] _summarize_old_exchanges batch error: {e}")
                    continue

            # Refresh the system prompt so new summaries are used immediately.
            self._rebuild_system_prompt()
            print("[Memory] Summarization complete.")

            # Level 3+: cleanup covered raw rows, then meta-summarize old
            # summaries. Order matters — summaries first, then cleanup, then
            # meta. All three are safe no-ops when there's nothing to do.
            memory.cleanup_summarized_exchanges()
            self._run_meta_summarization()
        except Exception as e:
            logger.exception(f"[Memory] _summarize_old_exchanges error: {e}")
        finally:
            self._summarizing = False
            # Safety net: if we ran from the wake-mode hook, make sure the UI
            # state is restored to "wake" when we finish.
            if not self._in_conversation:
                try:
                    ws_server.set_state("wake")
                except Exception:
                    pass

    def _run_meta_summarization(self) -> None:
        """Level 3+: compress individual summaries older than 30 days into a
        single rolling long-range memory paragraph. Silent — never speaks.
        If a meta-summary already exists, merges new candidates into it."""
        try:
            candidates = memory.get_summaries_needing_meta()
            if not candidates:
                return

            print(f"[Memory] Meta-summarizing {len(candidates)} old summaries.")

            combined = "\n".join(
                f"[{(s['date_from'] or '')[:10]} to {(s['date_to'] or '')[:10]}]: {s['summary_text']}"
                for s in candidates
            )

            existing = memory.get_latest_meta_summary()
            if existing:
                meta_prompt = (
                    "Below is an existing long-range memory paragraph about a person "
                    "named Nicholas, followed by new summary entries to incorporate.\n\n"
                    "EXISTING LONG-RANGE MEMORY:\n" + existing + "\n\n"
                    "NEW SUMMARIES TO INCORPORATE:\n" + combined + "\n\n"
                    "Rewrite the long-range memory as a single updated paragraph of "
                    "3-5 sentences. Merge old and new information. Preserve all "
                    "specific facts, preferences, names, and recurring themes. "
                    "Write in third person. No filler. No bullet points. "
                    "Do not start with 'Nicholas' as the first word."
                )
            else:
                meta_prompt = (
                    "Below are summary entries from conversations with a person "
                    "named Nicholas. Compress them into a single paragraph of "
                    "3-5 sentences capturing the most important facts, preferences, "
                    "recurring themes, and anything personally relevant. "
                    "Write in third person. No filler. No bullet points. "
                    "Do not start with 'Nicholas' as the first word.\n\n"
                    + combined
                )

            # Silent non-streaming LLM call — never TTS, never touches history.
            # Held under _llm_lock to serialize with other LLM callers.
            with self._llm_lock:
                result = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a concise long-range memory compressor. Reply with only the paragraph."},
                        {"role": "user", "content": meta_prompt},
                    ],
                    max_tokens=200,
                    temperature=0.3,
                    top_p=0.9,
                    stop=["<|eot_id|>"],
                    stream=False,
                )
            meta_text = (
                result["choices"][0]["message"]["content"] or ""
            ).strip()

            if not meta_text:
                print("[Memory] Meta-summarization returned empty result, skipping.")
                return

            meta_text = _clean(meta_text)
            if not meta_text:
                print("[Memory] Meta-summarization produced only filler, skipping.")
                return

            source_ids = [s["id"] for s in candidates]
            dates_from = [s["date_from"] for s in candidates if s.get("date_from")]
            dates_to = [s["date_to"] for s in candidates if s.get("date_to")]
            covers_from = min(dates_from) if dates_from else ""
            covers_to = max(dates_to) if dates_to else ""

            memory.save_meta_summary(meta_text, source_ids, covers_from, covers_to)
            print(
                f"[Memory] Meta-summary saved. Covers "
                f"{(covers_from or 'unknown')[:10]} to {(covers_to or 'unknown')[:10]}."
            )

            # Rebuild once more so the fresh meta-summary is in effect immediately.
            self._rebuild_system_prompt()
        except Exception as e:
            logger.exception(f"[Memory] _run_meta_summarization error: {e}")

    def _rebuild_system_prompt(self) -> None:
        """Rebuild the system prompt from the base prompt + current memory facts
        + rolling summaries of older conversations (Level 3).
        Called at startup and whenever facts or summaries change mid-session."""
        prompt = self._base_prompt
        facts_str = memory.format_facts_for_prompt()
        if facts_str:
            prompt = prompt + "\n\n" + facts_str
        # Long-range memory first (oldest context), then recent summaries.
        # LLMs build understanding best when context flows oldest-to-newest.
        meta_str = memory.format_meta_summary_for_prompt()
        if meta_str:
            prompt = prompt + "\n\n" + meta_str
        summaries_str = memory.format_summaries_for_prompt()
        if summaries_str:
            prompt = prompt + "\n\n" + summaries_str
        prompt = prompt + " When the user asks you to remember something, confirm it naturally with a brief phrase like 'Got it, I will keep that in mind' or 'Noted.' Do not repeat the fact back verbatim."
        self.system_prompt = prompt

    def _messages(self, history_pairs: Optional[int] = None,
                  system_override: Optional[str] = None) -> list[dict]:
        # Hard cap at 10 pairs regardless of config — keeps every inference
        # call's input tokens bounded so an old config with history_turns=20
        # can't silently blow up the context window we just tightened to 2048.
        if history_pairs is None:
            history_pairs = self._llm_cfg.get("history_turns", 10)
        history_pairs = min(history_pairs, 10)
        recent = self.history[-(history_pairs * 2):] if history_pairs > 0 else []
        system = system_override if system_override is not None else self.system_prompt
        return [{"role": "system", "content": system}] + recent

    def stream_sentences(self, user_text: str,
                         memory_context: Optional[str] = None,
                         fast: bool = False):
        # Always append the CLEAN user_text to history — never the injected
        # memory context. That's persisted only for this one LLM call.
        self.history.append({"role": "user", "content": user_text})

        if fast:
            # Fast conversational path. Strip memory facts/summaries from the
            # system prompt (they can cost hundreds of tokens) and use only
            # the last 3 pairs of history. Caps output at 150 tokens.
            messages = self._messages(
                history_pairs=3, system_override=self._base_prompt,
            )
            max_new_tokens = 150
        else:
            messages = self._messages()
            max_new_tokens = self._llm_cfg.get("max_new_tokens", 256)

        if memory_context and not fast:
            # Replace the last user message (the one we just appended) with
            # a prefixed version for this one call only. history stays clean.
            prefixed = memory_context + user_text
            messages = messages[:-1] + [{"role": "user", "content": prefixed}]

        # Held under _llm_lock for the ENTIRE generator iteration — a
        # concurrent LLM call from memory summarization or a calendar
        # worker would otherwise corrupt llama-cpp's internal state and
        # crash the process (SIGSEGV / exit code 11).
        buf  = ""
        full = ""
        with self._llm_lock:
            stream = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=self._llm_cfg.get("temperature", 0.7),
                top_p=self._llm_cfg.get("top_p", 0.9),
                stop=["<|eot_id|>", "\nUser:", "\nYou:"],
                stream=True,
            )

            for chunk in stream:
                delta: str = chunk["choices"][0]["delta"].get("content", "") or ""
                buf  += delta
                full += delta

                parts = SENTENCE_RE.split(buf)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        c = _clean(sentence)
                        if c:
                            yield c
                    buf = parts[-1]
                    continue

                if len(buf.split()) >= MIN_CLAUSE_WORDS:
                    clauses = CLAUSE_RE.split(buf)
                    if len(clauses) > 1:
                        for clause in clauses[:-1]:
                            c = _clean(clause)
                            if c:
                                yield c
                        buf = clauses[-1]

        if buf.strip():
            c = _clean(buf)
            if c:
                yield c

        self.history.append({"role": "assistant", "content": _clean(full)})

    def handle_turn(self, user_input: str, memory_context: Optional[str] = None,
                    fast: bool = False) -> None:
        """Three-thread pipeline: LLM → TTS → SeamlessPlayer (zero-gap audio).

        memory_context (optional) is a one-turn prefix injected into the LLM
        call for retrieved memory search results. It is NOT saved to history
        or to the database — only the clean user_input is persisted.

        fast=True enables the lightweight conversational path: base system
        prompt only, 3 history pairs, 150 max tokens, no memory context."""
        self._cancel_conversation_timer()  # Pause while processing + speaking
        self._stop_speak.clear()
        ws_server.set_state("thinking")

        sentence_q: queue.Queue[Optional[str]] = queue.Queue()
        player = SeamlessPlayer(sample_rate=TTS_RATE)
        player.start()

        first_audio_ready = threading.Event()
        display_parts: list[str] = []
        display_lock = threading.Lock()

        def _llm() -> None:
            for chunk in self.stream_sentences(
                user_input, memory_context=memory_context, fast=fast,
            ):
                sentence_q.put(chunk)
            sentence_q.put(None)

        def _tts() -> None:
            first = True
            while True:
                chunk = sentence_q.get()
                if chunk is None:
                    break
                if self._stop_speak.is_set():
                    break
                wav = self._synthesise(chunk)
                player.feed(wav)
                with display_lock:
                    display_parts.append(chunk)
                if first:
                    ws_server.set_state("speaking")
                    first_audio_ready.set()
                    first = False
            player.mark_done()

        llm_t = threading.Thread(target=_llm, daemon=True)
        tts_t = threading.Thread(target=_tts, daemon=True)

        stop_spin = threading.Event()
        spin_t    = threading.Thread(
            target=self._spinner, args=(stop_spin,), daemon=True
        )
        spin_t.start()
        llm_t.start()
        tts_t.start()

        first_audio_ready.wait(timeout=60)
        stop_spin.set()
        spin_t.join()

        tts_t.join()
        with display_lock:
            response_text = " ".join(display_parts)
        sys.stdout.write(f"Jarvis: {response_text}\n")
        sys.stdout.flush()

        memory.save_exchange(user_input, response_text)
        memory.rebuild_fts_indexes()

        player.wait()
        llm_t.join()
        # Resume conversation timer after LLM response finishes speaking
        if self._in_conversation:
            self._start_conversation_timer()
        ws_server.set_state("idle")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "═" * 58)
        print("  🟢  Jarvis ready — say 'Hey Jarvis' to begin")
        print("  Open http://localhost:3000 to see the UI")
        print("  Press Ctrl+C to quit")
        print("═" * 58 + "\n")

        self._init_wake_word()
        # Feature 7: start background notification monitor (timers already
        # speak via _timer_callback; this adds calendar + reminder due
        # notifications).
        self.start_notification_monitor()
        if STARTUP_MODE == "conversation":
            self._in_conversation = True
            self._start_conversation_timer()
            ws_server.set_state("listening")
            print("\n[STARTUP] Starting in conversation mode", flush=True)
        else:
            ws_server.set_state("wake")

        while True:
            try:
                if ws_server.is_muted():
                    ws_server.set_state("idle")
                    time.sleep(0.1)
                    continue

                if not self._in_conversation:
                    # Level 3: fire background summarization on wake-mode entry.
                    # Flag is set at __init__ (startup) and in _end_conversation
                    # (inactivity timeout + manual return-to-wake command).
                    if self._needs_wake_summarization:
                        self._needs_wake_summarization = False
                        threading.Thread(
                            target=self._summarize_old_exchanges,
                            daemon=True,
                        ).start()

                    # WAKE MODE: listen for wake word
                    detected = self._listen_for_wake_word()
                    if detected:
                        # NO sleep here — every millisecond we wait is
                        # a millisecond of the user's single-breath
                        # command that gets lost before the mic opens.
                        # Previous versions tried 0.3s, 1.0s, 0.2s; all
                        # of them swallowed "Hey Jarvis, tell me a joke"
                        # said without a pause. The echo-stripping logic
                        # below handles any wake-phrase tail that leaks
                        # into the transcript.
                        self._drain_q()
                        self._in_conversation = True
                        self._return_to_wake.clear()
                        # Mark this turn as the first post-wake utterance
                        # so the run loop can either strip a "hey jarvis"
                        # prefix from the first transcription or discard
                        # the transcript entirely if it's just the wake
                        # phrase echo.
                        self._just_woke = True
                        self._start_conversation_timer()
                        ws_server.set_state("listening")
                        print(
                            "\n[WAKE] Wake word detected — entering conversation mode",
                            flush=True
                        )
                    continue

                # Check if we should return to wake mode
                if self._return_to_wake.is_set():
                    self._return_to_wake.clear()
                    continue

                # ── Calendar worker guard ─────────────────────────────────
                # A background calendar worker is running. The worker is
                # responsible for speaking the result and restoring state.
                # Skip audio capture entirely until it finishes so we don't
                # record the assistant's own "one moment, Sir" output and
                # we don't race the worker's state updates.
                if self._calendar_working.is_set():
                    time.sleep(0.1)
                    continue

                # CONVERSATION MODE: full voice pipeline
                # Check pending Mac power confirmation first
                if (self.pending_confirmation is not None):
                    action = self.pending_confirmation.get('action')
                    audio = self.record_audio()
                    if not audio:
                        continue
                    response_text = self.transcribe(audio)
                    if not response_text:
                        continue
                    print(f"You: {response_text}")
                    t_conf = response_text.lower().strip()
                    if re.search(r"\b(?:yes|yeah|confirm|do\s+it|sure)\b", t_conf):
                        self.pending_confirmation = None
                        if action == "shutdown":
                            self.speak_direct("Shutting down your Mac now.")
                            subprocess.run(
                                ["osascript", "-e",
                                 'tell application "System Events" to shut down'],
                                check=False, timeout=10,
                            )
                        elif action == "restart":
                            self.speak_direct("Restarting your Mac now.")
                            subprocess.run(
                                ["osascript", "-e",
                                 'tell application "System Events" to restart'],
                                check=False, timeout=10,
                            )
                        elif action == "sleep":
                            self.speak_direct("Putting your Mac to sleep.")
                            subprocess.run(["pmset", "sleepnow"], check=False)
                    else:
                        self.pending_confirmation = None
                        self.speak_direct("Cancelled.")
                    self._start_conversation_timer()
                    continue

                audio = self.record_audio()
                if not audio:
                    continue  # Timer is already running — don't touch it

                user_input = self.transcribe(audio)
                if not user_input:
                    print("  (Didn't catch that — try again)\n")
                    continue  # Timer keeps running — don't touch it

                # Wake-phrase echo defense. If the first utterance after
                # the wake word starts with "hey jarvis" / "hi jarvis" /
                # "jarvis", we need to handle two cases:
                #
                #   a) The whole utterance IS just the wake phrase — the
                #      mic caught the tail of the user's own "Hey Jarvis"
                #      and there's no real command. Discard and keep
                #      listening for the next one.
                #
                #   b) The utterance is "hey jarvis <command>" said in one
                #      breath. Strip the wake prefix and use the remainder
                #      as the actual command.
                #
                # Only applies once per wake-up — the flag is cleared
                # immediately whether we keep, strip, or discard.
                if self._just_woke:
                    self._just_woke = False
                    _cleaned = re.sub(r"[^\w\s]", "", user_input).strip().lower()
                    # Prefix order matters: longer phrases must come first
                    # so "hey jarvis" is tried before the bare "jarvis"
                    # (otherwise the bare match would swallow the "hey"
                    # as part of the remainder).
                    _PREFIXES = (
                        "hey jarvis", "hi jarvis", "hello jarvis",
                        "hey jervis", "hey jarvus", "hey jarbis",
                        # Bare "jarvis" catches the common STT failure mode
                        # where faster-whisper drops the "Hey" and the
                        # transcription comes out as "Jarvis tomorrow I'm
                        # working from 7 to 5". Without this, the LLM
                        # extraction sees "Jarvis" as the first token and
                        # cheerfully uses it as the event title.
                        "jarvis",
                    )
                    _matched_prefix = None
                    for p in _PREFIXES:
                        if _cleaned.startswith(p):
                            _matched_prefix = p
                            break
                    if _matched_prefix is not None:
                        _remainder = _cleaned[len(_matched_prefix):].strip()
                        # Tolerate trailing "hey" or similar junk
                        _remainder = re.sub(r"^(hey|hi|hello)\s+", "", _remainder)
                        if not _remainder:
                            print(f"  (Ignoring wake-phrase echo: {user_input!r})\n")
                            continue
                        # The user said wake phrase + command in one breath.
                        # Replace user_input with just the command portion
                        # so the rest of the pipeline sees a clean command.
                        print(f"  (Stripped wake prefix: {user_input!r} -> {_remainder!r})")
                        user_input = _remainder

                # Confirmed speech — cancel timer while Jarvis processes and responds
                self._cancel_conversation_timer()

                print(f"You: {user_input}")

                # Pending calendar clarification (e.g. "Where are you
                # working, Sir?"). Route the user's answer back into the
                # pending action, don't treat it as a new command.
                if self._pending_calendar_action is not None:
                    self._resume_pending_calendar_action(user_input)
                    print()
                    continue

                # Pending file confirmation — the next utterance answers
                # "is this the file you want to move?" rather than being
                # a new command. Find/describe live in this state too,
                # but their waiting_for is None so the resume helper just
                # clears the preview and we fall through to handle the
                # new utterance normally.
                if self._pending_file_action is not None:
                    waiting_for = self._pending_file_action.get("waiting_for")
                    if waiting_for == "confirmation":
                        self._resume_pending_file_action(user_input)
                        print()
                        continue
                    # Read-only pending (find/describe) — dismiss preview
                    # and keep processing the new utterance below.
                    self._resume_pending_file_action(user_input)

                remember_fact = memory.detect_remember_command(user_input)
                if remember_fact:
                    fact_key = "user_note_" + str(int(time.time()))
                    memory.save_fact(fact_key, remember_fact)
                    print(f"[Memory] Saved fact: {remember_fact}")
                    self._rebuild_system_prompt()

                # Forget command (Level 2) — checked before auto-detect so
                # "forget that I like coffee" doesn't re-save the fact.
                forget_term = memory.detect_forget_command(user_input)
                if forget_term:
                    deleted = memory.delete_matching_facts(forget_term)
                    print(f"[Memory] Deleted {deleted} fact(s) matching: {forget_term}")
                    self._rebuild_system_prompt()
                    # Let the LLM respond naturally below

                # Level 4: Memory search — detect questions about past
                # context and retrieve matching rows via FTS5, optionally
                # filtered by a parsed date range. Runs after remember/forget
                # so those always take priority, and before auto-detect so
                # it can't trigger a fact save.
                search_spec = memory.detect_memory_search_query(user_input)
                memory_search_results: list = []
                if search_spec:
                    terms = search_spec.get("terms", "") or ""
                    dr = search_spec.get("date_range")
                    memory_search_results = memory.search_memory(
                        query=terms,
                        date_range=dr,
                    )
                    # Human-readable description for the terminal.
                    if dr and terms:
                        desc = f"terms='{terms}' in range {dr[0][:10]}..{dr[1][:10]}"
                    elif dr:
                        desc = f"range {dr[0][:10]}..{dr[1][:10]}"
                    else:
                        desc = f"terms='{terms}'"
                    if memory_search_results:
                        print(
                            f"[Memory] Search {desc} — "
                            f"{len(memory_search_results)} result(s) found."
                        )
                    else:
                        print(f"[Memory] Search {desc} — no results.")

                # Auto-detect a casually mentioned personal fact (Level 2)
                if not remember_fact and not forget_term:
                    detected = memory.auto_detect_fact(user_input)
                    if detected:
                        fact_key, fact_value = detected
                        existing_values = [v for k, v in memory.get_all_facts()]
                        if not memory.facts_are_similar(fact_value, existing_values):
                            memory.save_fact(fact_key, fact_value)
                            print(f"[Memory] Auto-saved fact: {fact_value}")
                            self._rebuild_system_prompt()
                            self._pending_memory_ack = True
                        else:
                            print(f"[Memory] Skipped duplicate: {fact_value}")

                # "What do you know about me?" readback (Level 2)
                readback_triggers = (
                    "what do you know about me",
                    "what have you remembered",
                    "what do you remember about me",
                    "tell me what you know about me",
                    "what facts do you have about me",
                )
                lowered_input = user_input.lower()
                is_readback = any(t in lowered_input for t in readback_triggers)

                if is_readback:
                    facts = memory.get_facts_for_readback()
                    if facts:
                        fact_list = "\n".join(f"- {f}" for f in facts)
                        readback_prompt = (
                            "The user has asked what you know about them. Here are the stored facts:\n"
                            f"{fact_list}\n"
                            "Respond in one natural, conversational paragraph. Do NOT use bullet points. "
                            "Weave them together as if recalling what you know about this person — warm, "
                            "organic, and intelligent. Do not start with 'Of course' or 'Certainly'. "
                            "Do not say you are reading from a list."
                        )
                        self.handle_turn(readback_prompt)
                    else:
                        self.handle_turn(
                            "The user asked what you know about them, but you don't have much stored yet. "
                            "Respond naturally and briefly, acknowledging you don't have much memory of them yet "
                            "and inviting them to tell you more."
                        )
                    if self._pending_memory_ack:
                        self._pending_memory_ack = False
                        self._speak_memory_ack()
                    print()
                    continue

                # Clipboard augmentation
                augmented_input, is_clipboard = self._try_augment_clipboard(user_input)

                # Fast conversational path. Short, command-keyword-free
                # utterances ("how are you?", "tell me a joke") skip the
                # full intent pipeline AND the memory-context injection, and
                # use a minimal 3-pair history window. This is the single
                # biggest win on simple turns — cuts the input-token count
                # by 5-10x on average, which directly reduces llama-cpp's
                # prefill time.
                if (
                    not is_clipboard
                    and is_simple_conversational_turn(user_input)
                ):
                    print("[Fast] simple conversational turn — skipping intent pipeline")
                    self.handle_turn(user_input, fast=True)
                    if self._pending_memory_ack:
                        self._pending_memory_ack = False
                        self._speak_memory_ack()
                    print()
                    continue

                # System command, calendar intent, file intent, screen
                # intent, or LLM
                sys_response = self._handle_system_command(user_input)
                cal_intent = None
                file_intent = None
                screen_intent = None
                if not sys_response:
                    # Only probe calendar intent when no system command
                    # matched — keeps timer-style reminders ("remind me in
                    # 5 minutes") on their existing fast path. Intent
                    # detection is strict regex only; casual chat falls
                    # through to None.
                    cal_intent = self._detect_calendar_intent(user_input)
                    if cal_intent is None:
                        file_intent = self._detect_file_intent(user_input)
                    if cal_intent is None and file_intent is None:
                        screen_intent = self._detect_screen_intent(user_input)

                if sys_response and sys_response != WAKE_MODE_SENTINEL:
                    print(f"System: {sys_response}")
                    self.speak_direct(sys_response)
                elif sys_response == WAKE_MODE_SENTINEL:
                    pass  # Already spoken and handled inside the command
                elif cal_intent is not None:
                    print(f"[Calendar] Intent: {cal_intent}")
                    # Kicks off a background worker and returns immediately.
                    # Main loop will idle on _calendar_working until the
                    # worker finishes.
                    self._handle_calendar_command(cal_intent, user_input)
                elif file_intent is not None:
                    print(f"[File] Intent: {file_intent}")
                    self._handle_file_command(file_intent, user_input)
                elif screen_intent is not None:
                    print(f"[Screen] Intent: {screen_intent}")
                    self._handle_screen_command(screen_intent, user_input)
                else:
                    # Build the one-turn memory context block if we have search hits.
                    memory_context: Optional[str] = None
                    if memory_search_results:
                        def _row_desc(r: dict) -> str:
                            if r.get("source") == "summary":
                                body = r.get("summary_text", "")
                                ts = (r.get("date_from", "") or "")[:10]
                            else:
                                body = r.get("content", "")
                                ts = (r.get("timestamp", "") or "")[:10]
                            return f"- {body} ({ts})"

                        memory_context = (
                            "[Retrieved memory context for this question:\n"
                            + "\n".join(_row_desc(r) for r in memory_search_results)
                            + "\nUse this context to answer the following question "
                            "naturally, as if recalling from memory. Do not say you "
                            "searched a database. Do not say 'according to my records'.]"
                            "\n\n"
                        )
                    self.handle_turn(augmented_input, memory_context=memory_context)

                # Copy LLM response to clipboard if requested
                if is_clipboard and self.history:
                    last = self.history[-1].get("content", "")
                    if last:
                        self._copy_to_clipboard(last)
                        print("📋  Response copied to clipboard.", flush=True)

                # Subtle in-character ack for auto-detected facts (Level 2)
                if self._pending_memory_ack:
                    self._pending_memory_ack = False
                    self._speak_memory_ack()

                print()

            except KeyboardInterrupt:
                print("\nGoodbye.")
                self._cancel_conversation_timer()
                self._cleanup_wake_word()
                ws_server.set_state("idle")
                break
            except JarvisPauseRequest:
                self._in_conversation = False
                self._cancel_conversation_timer()
                self._cleanup_wake_word()
                ws_server.set_state("idle")
                time.sleep(0.3)  # Let ws_server broadcast the idle state
                os._exit(0)      # Bypass Metal GPU cleanup to prevent crash


if __name__ == "__main__":
    # Configure logging so background-thread errors (notification monitor,
    # memory summarization, calendar/file workers) surface in the logs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # Start HTTP/WebSocket servers first — before heavy ML imports in VoiceAssistant.
    ws_server.start()
    assistant = VoiceAssistant()
    assistant.run()
