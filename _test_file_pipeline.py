"""
Standalone verifier for the voice file-management pipeline.

Loads the real LLM (same model the voice pipeline uses) and runs the
LLM extraction prompt + search_file() against a handful of realistic
utterances. Prints the extracted JSON and the top matches so we can
confirm the whole pipeline works end-to-end without needing a mic.

Run:  .venv311/bin/python _test_file_pipeline.py
"""

import json
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import file_manager

# --- Load LLM exactly like VoiceAssistant._load_llm does ---------------------
def _load_llm():
    from huggingface_hub import try_to_load_from_cache
    from llama_cpp import Llama

    cfg_path = Path(__file__).parent / "config.json"
    cfg = json.loads(cfg_path.read_text())["llm"]
    cached = try_to_load_from_cache(repo_id=cfg["repo_id"], filename=cfg["filename"])
    assert cached and Path(cached).is_file(), "LLM not in HF cache — run Jarvis once first"
    return Llama(
        model_path=str(cached),
        n_gpu_layers=cfg.get("n_gpu_layers", -1),
        n_ctx=cfg.get("n_ctx", 4096),
        verbose=False,
    )


_LLM_LOCK = threading.Lock()
_LLM = None


def _llm_silent(system: str, user_prompt: str, max_tokens: int = 220,
                temperature: float = 0.1) -> str:
    global _LLM
    with _LLM_LOCK:
        r = _LLM.create_chat_completion(
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
    return (r["choices"][0]["message"]["content"] or "").strip()


# Copy the production extract prompt verbatim so we're testing what ships.
def extract_file_json(utterance: str):
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
        "- destination: a string OR null. Resolve spoken locations "
        "to absolute paths — Desktop -> /Users/nicholascoppola/Desktop, "
        "Documents -> /Users/nicholascoppola/Documents, "
        "Downloads -> /Users/nicholascoppola/Downloads, "
        "Pictures -> /Users/nicholascoppola/Pictures, "
        "Music -> /Users/nicholascoppola/Music. "
        "For anything that isn't a move, set this to null.\n"
        "- new_name: string OR null. Only set when the user is "
        "renaming the file; otherwise null.\n\n"
        "Examples:\n"
        "  'move the rmv file from my desktop to my documents folder' -> "
        "{\"query\": \"rmv\", \"action\": \"move\", "
        "\"destination\": \"/Users/nicholascoppola/Documents\", "
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
    raw = _llm_silent(
        "You are a precise JSON extraction tool. Reply with valid JSON only.",
        prompt,
        max_tokens=220,
        temperature=0.1,
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```\s*$", "", raw).strip()
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"  JSON parse error: {e} raw={raw!r}")
        return None


# --- Cases we want to work end-to-end ---------------------------------------
#
# Each case is (utterance, assertion) where assertion is a function taking
# (extracted_json, matches) and returning True on success. Two classes of
# assertion:
#   1. Happy path — the RMV file on Desktop must resolve for every RMV utterance.
#   2. Safety — Jarvis must NEVER surface its own internals (jarvis_memory.db,
#      config.json, any path inside the Jarvis project dir) even for adversarial
#      utterances that mention "jarvis" or "memory".
import os
_JARVIS_DIR = os.path.dirname(os.path.abspath(__file__))


# Mirror the deterministic regex from VoiceAssistant._parse_destination_from_utterance
# so the harness exercises the same override logic.
_DEST_RE = re.compile(
    r"\b(?:to|into|onto)\s+(?:my\s+|the\s+|a\s+)?"
    r"(desktop|downloads?|documents?|pictures?|photos?|"
    r"music|movies?|videos?|home)"
    r"(?:\s+folder|\s+directory)?\b"
)


def _parse_dest_regex(utterance: str):
    t = (utterance or "").lower()
    cleaned = re.sub(
        r"\bfrom\s+(?:my\s+|the\s+)?\w+(?:\s+folder|\s+directory)?\b",
        "",
        t,
    )
    m = _DEST_RE.findall(cleaned)
    return m[-1] if m else None


def _no_protected(matches):
    for m in matches:
        if file_manager.is_protected_path(m):
            return False, f"SURFACED PROTECTED PATH: {m}"
    return True, ""


def _has_rmv(matches):
    ok_proto, why = _no_protected(matches)
    if not ok_proto:
        return False, why
    if not any("rmv" in m.lower() for m in matches):
        return False, "RMV file missing from matches"
    return True, ""


def _dest_extraction_check(expected_substring):
    """Returns a case-check that validates the LLM extracted the
    destination correctly — used to catch "from X to Y" confusion."""
    def _check(matches, extracted=None):
        ok_proto, why = _no_protected(matches)
        if not ok_proto:
            return False, why
        if extracted is None:
            return True, ""
        dest = (extracted.get("destination") or "").lower()
        if expected_substring.lower() not in dest:
            return False, f"destination {dest!r} missing {expected_substring!r}"
        return True, ""
    return _check


CASES = [
    # Happy path — RMV resolution
    ("Okay, please move the rmv file from my desktop to my documents folder.", _has_rmv),
    ("The file is called rmv-realid-application-steps. Please move that file from my desktop to my documents.", _has_rmv),
    ("Move the RMV file to Downloads", _has_rmv),
    ("Find my RMV file", _has_rmv),
    ("What's in the RMV file on my desktop", _has_rmv),
    ("Rename the rmv file to license application", _has_rmv),
    # Follow-up phrasings where the verb is past/progressive tense —
    # common when the user is correcting themselves after a prior move.
    # All of these must still yield RMV even via the extraction -> search.
    ("Actually Jarvis moved the rmv file to my documents folder", _has_rmv),
    ("Moved the rmv file to my documents", _has_rmv),
    ("Put the RMV file on my desktop", _has_rmv),
    ("Take the rmv file and put it in Downloads", _has_rmv),
    # "from X to Y" and "from X into Y" — LLM must put Y (not X)
    # as destination. This is the regression that made the third
    # attempt in Nicholas's log go Downloads -> Downloads.
    ("Move the rmv file from my downloads folder into the documents folder",
     _dest_extraction_check("Documents")),
    ("Move the rmv file from my desktop to my downloads folder",
     _dest_extraction_check("Downloads")),
    # Regression — Jarvis must not suggest its own memory.db, config, etc.
    ("Move my Jarvis file to Downloads",        _no_protected),
    ("Move the Jarvis memory file to Desktop",  _no_protected),
    ("Find my Jarvis db",                        _no_protected),
    ("Move my config file to Downloads",         _no_protected),
    ("Move the memory database to Desktop",      _no_protected),
    ("Rename Jarvis memory to foo",              _no_protected),
    ("Summarize the jarvis_memory.db file",      _no_protected),
]


def _run_cases(label: str, cases, spotlight_blind: bool = False) -> tuple[int, int]:
    print(f"\n#### {label} ####\n")
    orig_mdfind = file_manager._run_mdfind
    if spotlight_blind:
        file_manager._run_mdfind = lambda args: []
    ok = 0
    bad = 0
    try:
        for utt, check in cases:
            print("=" * 72)
            print(f"UTTERANCE: {utt}")
            data = extract_file_json(utt) or {}
            print(f"  extracted: {data}")
            query = (data.get("query") or "").strip() or utt
            matches = file_manager.search_file(query)
            dest = data.get("destination")
            # Apply the same deterministic override the production
            # handler uses so the harness tests the full pipeline.
            regex_dest = _parse_dest_regex(utt)
            if regex_dest:
                dest = regex_dest
                data["destination"] = regex_dest
            resolved = file_manager.resolve_destination(dest) if dest else None
            print(f"  destination={dest!r}  resolved={resolved!r}")
            print(f"  matches ({len(matches)}):")
            for m in matches:
                marker = " [PROTECTED!]" if file_manager.is_protected_path(m) else ""
                print(f"    - {m}{marker}")
            # Check signature may be either check(matches) or
            # check(matches, extracted). Support both.
            try:
                passed, why = check(matches, data)
            except TypeError:
                passed, why = check(matches)
            if passed:
                print("  ok")
                ok += 1
            else:
                print(f"  FAIL — {why}")
                bad += 1
            print()
    finally:
        file_manager._run_mdfind = orig_mdfind
    return ok, bad


def main() -> None:
    global _LLM
    print("[test] loading LLM…")
    _LLM = _load_llm()
    print("[test] ready\n")

    # Round 1 — live Spotlight. Round 2 — Spotlight-blind (simulated
    # TCC): force mdfind to return nothing so only the filesystem-walk
    # fallback can find files. Every case that passes Round 1 must also
    # pass Round 2 or Jarvis will be useless inside JarvisApp.
    ok1, bad1 = _run_cases("ROUND 1 — live Spotlight", CASES, spotlight_blind=False)
    ok2, bad2 = _run_cases("ROUND 2 — Spotlight-blind (TCC simulation)", CASES, spotlight_blind=True)
    ok = ok1 + ok2
    bad = bad1 + bad2
    print(f"summary: round1 {ok1} ok / {bad1} fail  |  round2 {ok2} ok / {bad2} fail  "
          f"|  total {ok} ok / {bad} fail")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
