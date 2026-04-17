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
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
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

# Adaptive silence: short for quick commands, longer once you've been speaking a while
SILENCE_CUTOFF_SHORT_MS  = 520
SILENCE_CUTOFF_LONG_MS   = 950
LONG_SPEECH_THRESHOLD_MS = 2_500   # use long cutoff after 2.5 s of speech

PLAYER_BLOCKSIZE = 4_096

# ── Wake word ────────────────────────────────────────────────────────────────
WAKE_WORD_MODEL = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.75
CONVERSATION_TIMEOUT = 15.0
WAKE_CHUNK_SIZE = 1280
STARTUP_MODE = os.environ.get("JARVIS_STARTUP_MODE", "wake")
WAKE_MODE_SENTINEL = "__WAKE_MODE__"
NOISE_FLOOR_RMS = 150  # Minimum RMS energy to consider audio as speech

# Level 3 memory: speak a brief line when summarization starts? Set False for silent.
SPEAK_MEMORY_UPDATE = True

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

        self._load_llm()
        self._load_tts()
        self._load_stt()

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
            n_ctx=c.get("n_ctx", 4096),
            verbose=False,
        )
        self._llm_cfg = c
        print("[LLM] Ready  (Metal GPU layers active)")

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
            elif speaking:
                buf        += frame
                silence_ms += FRAME_MS
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
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 400,           # keep a bit of audio around speech edges
            },
        )
        return " ".join(s.text.strip() for s in segments).strip()

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _synthesise(self, text: str) -> np.ndarray:
        samples, _ = self._kokoro.create(
            text, voice=self._voice, speed=self._speed, lang="en-us"
        )
        return np.asarray(samples, dtype=np.float32)

    def speak_direct(self, text: str) -> None:
        """Speak text immediately via TTS — no LLM involved."""
        self._tts_speaking = True
        self._cancel_conversation_timer()  # Pause timer while speaking
        ws_server.set_state("speaking")
        try:
            wav    = self._synthesise(text)
            player = SeamlessPlayer(sample_rate=TTS_RATE)
            player.start()
            player.feed(wav)
            player.mark_done()
            player.wait()
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
        result = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True
        )
        return result.stdout.strip()

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
            check=False,
        )
        # Drain the audio queue so the mic doesn't pick up the notification sound
        time.sleep(0.3)
        self._drain_q()
        self.speak_direct(msg)
        # Drain again after speaking so Jarvis doesn't hear itself
        time.sleep(0.5)
        self._drain_q()

    # ── Wake word ─────────────────────────────────────────────────────────────

    def _init_wake_word(self) -> None:
        """Load the openwakeword model AND start the persistent audio stream.

        The persistent stream feeds self._audio_q continuously from here on.
        Both wake-word detection and command recording read from the same
        queue, so there's no stream close/reopen between modes and no gap
        where single-breath commands can get lost."""
        print("[WAKE] Loading hey_jarvis model...", flush=True)
        self._wake_model = WakeWordModel(
            wakeword_models=[WAKE_WORD_MODEL],
            inference_framework='onnx'
        )
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
            ], check=False)
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
        prompts never pollute the conversation history."""
        try:
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
        """Parse HH:MM (24h) into (hour, minute)."""
        if not t:
            return None
        s = str(t).strip()
        m = re.match(r"^(\d{1,2}):(\d{2})$", s)
        if not m:
            return None
        try:
            h, mi = int(m.group(1)), int(m.group(2))
            if 0 <= h <= 23 and 0 <= mi <= 59:
                return (h, mi)
        except Exception:
            pass
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
            else:
                print(f"[Calendar] unknown intent: {intent!r}")
        except RuntimeError as e:
            # Known failure from calendar_reminders (timeout, permission, etc.)
            print(f"[Calendar] AppleScript error: {e}")
            self._safe_speak(
                "I wasn't able to access your calendar, Sir — "
                "you may need to grant permission in System Settings."
            )
        except Exception as e:
            print(f"[Calendar] worker error: {e}")
            import traceback
            traceback.print_exc()
            self._safe_speak(
                "I ran into a problem handling that calendar request, Sir."
            )
        finally:
            # Defensive: if the worker crashed mid-flight, clear any pending
            # clarification so the next utterance isn't silently swallowed.
            if self._pending_calendar_action is None:
                pass  # nothing to clean up
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
            "assistant would say it out loud. Group related items and mention "
            "days, not full dates. Keep it to three to five sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(self._JARVIS_CAL_SYSTEM, prompt, max_tokens=260)
        if text:
            self._safe_speak(text)
        else:
            self._safe_speak("I had trouble summarizing your week, Sir.")

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
            "just how a sharp assistant would say them out loud. "
            "IMPORTANT: mention the reminders in the exact order they are "
            "given below (earliest due date first, latest last). Do not "
            "rearrange them. Keep it brief, two to four sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(self._JARVIS_CAL_SYSTEM, prompt, max_tokens=220)
        if text:
            self._safe_speak(text)
        else:
            self._safe_speak("I had trouble summarizing your reminders, Sir.")

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

        title = (data_json.get("title") or "Reminder").strip()
        notes = data_json.get("notes")
        date_field = data_json.get("date")
        start_time = data_json.get("start_time")

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
                print(f"[Calendar] resume finalize error: {e}")
                self._safe_speak(
                    "I wasn't able to finish creating that event, Sir."
                )
            finally:
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
        """Speak a subtle in-character acknowledgment after auto-saving a fact."""
        ack = random.choice([
            "Noted.",
            "Understood.",
            "Duly noted.",
            "I'll keep that in mind.",
            "Noted, Sir.",
            "Understood, Sir.",
            "Duly noted, Sir.",
        ])
        print(f"[Memory] {ack}")
        try:
            self.speak_direct(ack)
        except Exception as e:
            print(f"[Memory] ack speak error: {e}")

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

            # Only speak the memory-update line on the initial startup run.
            # Re-entries from inactivity timeout or manual return-to-wake stay silent.
            if SPEAK_MEMORY_UPDATE and was_first_summarization:
                try:
                    self.speak_direct("One moment, Sir. Updating my memory.")
                except Exception as e:
                    print(f"[Memory] memory-update speak error: {e}")
                finally:
                    # speak_direct leaves state on "idle" — restore to "wake"
                    # since we're summarizing from the wake-mode hook.
                    if not self._in_conversation:
                        ws_server.set_state("wake")

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
                    print(f"[Memory] batch summarize error: {e}")
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
            print(f"[Memory] Summarization error: {e}")
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
            print(f"[Memory] Meta-summarization error: {e}")

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

    def _messages(self) -> list[dict]:
        max_pairs = self._llm_cfg.get("history_turns", 10)
        recent    = self.history[-(max_pairs * 2):]
        return [{"role": "system", "content": self.system_prompt}] + recent

    def stream_sentences(self, user_text: str, memory_context: Optional[str] = None):
        # Always append the CLEAN user_text to history — never the injected
        # memory context. That's persisted only for this one LLM call.
        self.history.append({"role": "user", "content": user_text})

        messages = self._messages()
        if memory_context:
            # Replace the last user message (the one we just appended) with
            # a prefixed version for this one call only. history stays clean.
            prefixed = memory_context + user_text
            messages = messages[:-1] + [{"role": "user", "content": prefixed}]

        stream = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self._llm_cfg.get("max_new_tokens", 256),
            temperature=self._llm_cfg.get("temperature", 0.7),
            top_p=self._llm_cfg.get("top_p", 0.9),
            stop=["<|eot_id|>", "\nUser:", "\nYou:"],
            stream=True,
        )

        buf  = ""
        full = ""

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

    def handle_turn(self, user_input: str, memory_context: Optional[str] = None) -> None:
        """Three-thread pipeline: LLM → TTS → SeamlessPlayer (zero-gap audio).

        memory_context (optional) is a one-turn prefix injected into the LLM
        call for retrieved memory search results. It is NOT saved to history
        or to the database — only the clean user_input is persisted."""
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
            for chunk in self.stream_sentences(user_input, memory_context=memory_context):
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
                                check=False
                            )
                        elif action == "restart":
                            self.speak_direct("Restarting your Mac now.")
                            subprocess.run(
                                ["osascript", "-e",
                                 'tell application "System Events" to restart'],
                                check=False
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

                # System command, calendar intent, or LLM
                sys_response = self._handle_system_command(user_input)
                cal_intent = None
                if not sys_response:
                    # Only probe calendar intent when no system command
                    # matched — keeps timer-style reminders ("remind me in
                    # 5 minutes") on their existing fast path. Intent
                    # detection is strict regex only; casual chat falls
                    # through to None.
                    cal_intent = self._detect_calendar_intent(user_input)

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
    # Start HTTP/WebSocket servers first — before heavy ML imports in VoiceAssistant.
    ws_server.start()
    assistant = VoiceAssistant()
    assistant.run()
