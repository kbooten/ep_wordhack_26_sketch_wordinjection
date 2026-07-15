#!/usr/bin/env python3
# SKELETON VERSION: builds prompts like ___ dog ___ smile ___
"""Live fading-word park walk.

The browser records speech and POSTs it to /api/transcribe. Transcribed words
enter a fading weighted pool. /api/generate chooses up to ten unique words,
randomizes their order, turns them into a fixed sentence skeleton, and asks the
LLM to fill the gaps while continuing one coherent walk through a park.
"""

from __future__ import annotations

import math
import os
import random
import re
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

# Helps avoid a frequent CTranslate2/OpenMP conflict under Anaconda on macOS.
_base_prefix = str(getattr(sys, "base_prefix", sys.prefix)).lower()
_is_conda_python = (
    bool(os.getenv("CONDA_PREFIX"))
    or "anaconda" in _base_prefix
    or "miniconda" in _base_prefix
)
if sys.platform == "darwin" and _is_conda_python:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from faster_whisper import WhisperModel
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from werkzeug.exceptions import RequestEntityTooLarge


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


WORD_LIFETIME_SECONDS = max(2.0, env_float("WORD_LIFETIME_SECONDS", 30.0))
WORD_DECAY_TAU_SECONDS = max(0.5, env_float("WORD_DECAY_TAU_SECONDS", 11.0))
GENERATION_INTERVAL_SECONDS = max(
    4.0,
    env_float("GENERATION_INTERVAL_SECONDS", 30.0),
)
MAX_PROMPT_WORDS = min(10, max(0, env_int("MAX_PROMPT_WORDS", 10)))
MAX_STORY_LINES = max(20, env_int("MAX_STORY_LINES", 1000))
MAX_STORY_CONTEXT_CHARS = max(
    5_000,
    env_int("MAX_STORY_CONTEXT_CHARS", 60_000),
)
MAX_UPLOAD_MB = max(1, env_int("MAX_UPLOAD_MB", 25))
MAX_OUTPUT_TOKENS = max(80, env_int("MAX_OUTPUT_TOKENS", 700))

# tiny.en is much faster for short live utterances. Set WHISPER_MODEL=base.en
# in .env if you prefer somewhat better recognition over lower latency.
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en").strip() or None
WHISPER_VAD_FILTER = env_bool("WHISPER_VAD_FILTER", False)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError(f"OPENAI_API_KEY was not found or was empty in {ENV_PATH}")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
openai_client = OpenAI(api_key=OPENAI_API_KEY)

state_lock = threading.RLock()
whisper_load_lock = threading.Lock()
whisper_run_lock = threading.Lock()
generation_lock = threading.Lock()

word_occurrences: list[dict[str, Any]] = []
story_lines: list[str] = []
_whisper_model: WhisperModel | None = None

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


SKELETON_PROMPT = """Write exactly one new sentence continuing an ongoing walk through a park.

You are not being given a bag of words. You are being given a SENTENCE SKELETON. The underscores are gaps for you to fill. The visible fixed words are already embedded in the sentence and must remain in exactly that order.

SENTENCE SKELETON:
{skeleton}

Fill every gap with as few or as many words as needed to turn the skeleton into one clear, grammatical, meaningful sentence. Output the completed sentence only.

Rules for the skeleton:
- Keep every fixed word exactly as written and in exactly the displayed order.
- Do not delete, alter, inflect, reorder, quote, list, label, define, or discuss a fixed word.
- Do not output underscores.
- Do not treat the fixed words as arbitrary names merely to get rid of them.
- The sentence may be long when necessary, but every added phrase must do real grammatical or narrative work.

Rules for meaning:
- Continue directly from the walk so far. Preserve what has happened, where we are, and the consequences of earlier events.
- Write in first-person plural and present tense.
- Make one specific perception, action, event, encounter, bodily sensation, memory, association, or thought happen during the walk.
- Honestly try to make the fixed words make sense together. Prefer the most concrete and plausible relationship available.
- Do not hide incoherence behind vague abstraction, poetic filler, generic mood, grand claims, or decorative nonsense.
- A fixed word may enter through something we see, hear, touch, do, remember, imagine, compare, mistake, or think, but the connection must be intelligible and specific.
- Add only the amount of invention required by the fixed words. The stranger or more difficult the skeleton, the more freedom you have to introduce an event, memory, transformation, tonal shift, or change of genre.
- When the skeleton contains no fixed words, write the shortest, plainest, most predictable concrete continuation of the walk.
- Do not restart, summarize, repeat an earlier sentence, or merely paraphrase the last sentence. Move the walk one small step forward.

Example of the mechanism only:
Skeleton: ___ dog ___ smile ___
Possible completion: We pass a dog whose open mouth resembles a smile.

WALK SO FAR:
{story}

Return only the completed sentence."""


def get_whisper_model() -> WhisperModel:
    global _whisper_model

    if _whisper_model is None:
        with whisper_load_lock:
            if _whisper_model is None:
                print(
                    f"Loading faster-whisper {WHISPER_MODEL_NAME!r} "
                    f"on {WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}...",
                    flush=True,
                )
                _whisper_model = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
                print("Whisper model ready.", flush=True)

    return _whisper_model


def tokenize(text: str) -> list[str]:
    return [
        match.group(0).lower().replace("’", "'")
        for match in WORD_RE.finditer(text)
    ]


def prune_words_locked(now: float) -> None:
    cutoff = now - WORD_LIFETIME_SECONDS
    word_occurrences[:] = [
        item for item in word_occurrences if float(item["added_at"]) > cutoff
    ]


def add_words(words: list[str]) -> list[dict[str, Any]]:
    now = time.time()
    added: list[dict[str, Any]] = []

    with state_lock:
        prune_words_locked(now)
        for index, word in enumerate(words):
            item = {
                "id": uuid4().hex,
                "word": word,
                # Preserve tiny recency differences within one utterance.
                "added_at": now + index * 0.0001,
            }
            word_occurrences.append(item)
            added.append(item.copy())

    return added


def active_snapshot() -> tuple[float, list[dict[str, Any]], list[str]]:
    now = time.time()
    with state_lock:
        prune_words_locked(now)
        return now, [item.copy() for item in word_occurrences], list(story_lines)


def weighted_sample_without_replacement(
    weights: dict[str, float],
    sample_size: int,
) -> list[str]:
    """Choose unique items while favoring larger weights."""
    ranked: list[tuple[float, str]] = []
    for word, weight in weights.items():
        safe_weight = max(weight, 1e-12)
        key = random.random() ** (1.0 / safe_weight)
        ranked.append((key, word))

    ranked.sort(reverse=True)
    return [word for _, word in ranked[:sample_size]]


def select_live_words(now: float) -> tuple[list[str], dict[str, float]]:
    """Choose at most ten unique words, weighted by recency and repetition."""
    with state_lock:
        prune_words_locked(now)
        weights: dict[str, float] = defaultdict(float)

        for item in word_occurrences:
            word = str(item["word"])
            age = max(0.0, now - float(item["added_at"]))
            weights[word] += math.exp(-age / WORD_DECAY_TAU_SECONDS)

    if not weights or MAX_PROMPT_WORDS == 0:
        return [], dict(weights)

    unique_words = list(weights)
    if len(unique_words) <= MAX_PROMPT_WORDS:
        chosen = unique_words
    else:
        chosen = weighted_sample_without_replacement(
            dict(weights),
            MAX_PROMPT_WORDS,
        )

    # This order is the order the LLM must preserve in the final sentence.
    random.shuffle(chosen)
    return chosen, dict(weights)


def make_sentence_skeleton(words: list[str]) -> str:
    """Convert [dog, smile] into '___ dog ___ smile ___'."""
    if not words:
        return "___"
    return "___ " + " ___ ".join(words) + " ___"


def story_context() -> str:
    with state_lock:
        lines = list(story_lines)

    if not lines:
        return "(The walk has not begun yet.)"

    kept_reversed: list[str] = []
    used_chars = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if kept_reversed and used_chars + cost > MAX_STORY_CONTEXT_CHARS:
            break
        if not kept_reversed and cost > MAX_STORY_CONTEXT_CHARS:
            kept_reversed.append(line[-MAX_STORY_CONTEXT_CHARS:])
            break
        kept_reversed.append(line)
        used_chars += cost

    kept = list(reversed(kept_reversed))
    if len(kept) < len(lines):
        kept.insert(0, "(Earlier parts are omitted; preserve visible consequences.)")
    return "\n".join(kept)


def call_openai_once(prompt: str) -> str:
    """One request only: no validator, rejection, repair, or retry prompt."""
    if hasattr(openai_client, "responses"):
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        return getattr(response, "output_text", "") or ""

    kwargs = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        completion = openai_client.chat.completions.create(
            **kwargs,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
    except (TypeError, ValueError):
        completion = openai_client.chat.completions.create(
            **kwargs,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    return completion.choices[0].message.content or ""


def clean_model_output(text: str) -> str:
    """Remove presentation wrapping only; never evaluate or rewrite content."""
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == '"':
        cleaned = cleaned[1:-1].strip()
    return cleaned


def generate_sentence(words: list[str]) -> tuple[str, str]:
    skeleton = make_sentence_skeleton(words)
    prompt = SKELETON_PROMPT.format(
        skeleton=skeleton,
        story=story_context(),
    )
    sentence = clean_model_output(call_openai_once(prompt))
    if not sentence:
        raise RuntimeError("OpenAI returned an empty response.")
    return sentence, skeleton


def audio_suffix(mimetype: str) -> str:
    mime = (mimetype or "").lower()
    if "mp4" in mime or "m4a" in mime:
        return ".m4a"
    if "ogg" in mime:
        return ".ogg"
    if "wav" in mime:
        return ".wav"
    return ".webm"


@app.get("/")
def index():
    return render_template(
        "index.html",
        generation_interval=GENERATION_INTERVAL_SECONDS,
        word_lifetime=WORD_LIFETIME_SECONDS,
    )


@app.get("/api/state")
def api_state():
    now, words, lines = active_snapshot()
    return jsonify(
        {
            "now": now,
            "word_lifetime_seconds": WORD_LIFETIME_SECONDS,
            "generation_interval_seconds": GENERATION_INTERVAL_SECONDS,
            "words": words,
            "story": lines,
        }
    )


@app.post("/api/transcribe")
def api_transcribe():
    request_started = time.perf_counter()
    upload = request.files.get("audio")
    if upload is None or not upload.filename:
        return jsonify({"error": "No audio upload was received."}), 400

    temp_path: Path | None = None
    try:
        save_started = time.perf_counter()
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=audio_suffix(upload.mimetype),
        ) as temp_file:
            upload.save(temp_file)
            temp_path = Path(temp_file.name)
        save_seconds = time.perf_counter() - save_started

        model = get_whisper_model()
        transcription_started = time.perf_counter()

        # One model is shared. Requests are serialized so overlapping browser
        # uploads cannot make the model fight itself for CPU and memory.
        with whisper_run_lock:
            segments, info = model.transcribe(
                str(temp_path),
                language=WHISPER_LANGUAGE,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=WHISPER_VAD_FILTER,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            transcript = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

        transcription_seconds = time.perf_counter() - transcription_started
        words = tokenize(transcript)
        added = add_words(words)
        total_seconds = time.perf_counter() - request_started

        print(
            "Transcription timing: "
            f"save={save_seconds:.3f}s, "
            f"whisper={transcription_seconds:.3f}s, "
            f"request={total_seconds:.3f}s, "
            f"text={transcript!r}",
            flush=True,
        )

        return jsonify(
            {
                "transcript": transcript,
                "words": [item["word"] for item in added],
                "language": getattr(info, "language", WHISPER_LANGUAGE),
                "timing": {
                    "save_seconds": round(save_seconds, 3),
                    "whisper_seconds": round(transcription_seconds, 3),
                    "request_seconds": round(total_seconds, 3),
                },
            }
        )

    except Exception as exc:
        print(f"Transcription failed: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/api/generate")
def api_generate():
    if not generation_lock.acquire(blocking=False):
        return jsonify({"error": "A sentence is already being generated."}), 409

    try:
        selected_words, weights = select_live_words(time.time())
        sentence, skeleton = generate_sentence(selected_words)

        with state_lock:
            story_lines.append(sentence)
            if len(story_lines) > MAX_STORY_LINES:
                del story_lines[:-MAX_STORY_LINES]

        return jsonify(
            {
                "sentence": sentence,
                "selected_words": selected_words,
                "word_skeleton": skeleton,
                "weights": weights,
                # Kept so an older front end expecting this key does not fail.
                "energy": 0.0,
            }
        )
    except Exception as exc:
        print(f"Generation failed: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        generation_lock.release()


@app.post("/api/reset")
def api_reset():
    with state_lock:
        word_occurrences.clear()
        story_lines.clear()
    return jsonify({"ok": True})


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error: RequestEntityTooLarge):
    return jsonify({"error": f"Audio upload exceeds {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    # Preload before accepting requests so the first utterance does not also
    # pay model startup/download cost.
    get_whisper_model()
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=env_int("PORT", 5000),
        debug=env_bool("FLASK_DEBUG", False),
        use_reloader=False,
        threaded=True,
    )
