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
CASES = [
    "Okay, please move the rmv file from my desktop to my documents folder.",
    "The file is called rmv-realid-application-steps. Please move that file from my desktop to my documents.",
    "Move the RMV file to Downloads",
    "Find my RMV file",
    "What's in the RMV file on my desktop",
    "Rename the rmv file to license application",
]


def main() -> None:
    global _LLM
    print("[test] loading LLM…")
    _LLM = _load_llm()
    print("[test] ready\n")

    ok = 0
    bad = 0
    for utt in CASES:
        print("=" * 72)
        print(f"UTTERANCE: {utt}")
        data = extract_file_json(utt) or {}
        print(f"  extracted: {data}")
        query = (data.get("query") or "").strip() or utt
        matches = file_manager.search_file(query)
        dest = data.get("destination")
        resolved = file_manager.resolve_destination(dest) if dest else None
        print(f"  destination={dest!r}  resolved={resolved!r}")
        print(f"  matches ({len(matches)}):")
        for m in matches:
            print(f"    - {m}")
        # Pass/fail: the RMV file must show up as a match for every
        # utterance that references RMV.
        if "rmv" in utt.lower() and not any(
            "rmv" in m.lower() for m in matches
        ):
            print("  ❌ FAIL — RMV not found")
            bad += 1
        else:
            ok += 1
        print()

    print(f"summary: {ok} ok / {bad} fail")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
