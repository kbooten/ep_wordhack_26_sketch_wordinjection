#!/usr/bin/env python3
"""Live fading-word story generator.

Hold Space in the browser to record. faster-whisper transcribes locally,
individual words enter a fading pool, and OpenAI writes one new sentence at a
fixed interval. We keep walking through a park: fresh/repeated words alter what
we directly see, feel, think, and do; when the pool empties, the walk becomes
calm again without erasing any concrete change that already occurred.
"""

from __future__ import annotations

import difflib
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

# Always load the .env file located beside this app.py.
# override=True prevents an older key exported in Terminal from taking priority.
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Configure OpenMP before importing faster-whisper/ctranslate2. Anaconda's
# NumPy stack and CTranslate2 can otherwise load two copies of libiomp5.dylib
# on macOS and abort during transcription.
_base_prefix = str(getattr(sys, "base_prefix", sys.prefix)).lower()
_is_conda_python = (
    bool(os.getenv("CONDA_PREFIX"))
    or "anaconda" in _base_prefix
    or "miniconda" in _base_prefix
)

if sys.platform == "darwin" and _is_conda_python:
    if "KMP_DUPLICATE_LIB_OK" not in os.environ:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        print(
            "Anaconda Python detected: enabling the macOS OpenMP compatibility workaround.",
            flush=True,
        )

    os.environ.setdefault("OMP_NUM_THREADS", "1")

from faster_whisper import WhisperModel
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from werkzeug.exceptions import RequestEntityTooLarge

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


WORD_LIFETIME_SECONDS = max(2.0, env_float("WORD_LIFETIME_SECONDS", 30.0))
WORD_DECAY_TAU_SECONDS = max(0.5, env_float("WORD_DECAY_TAU_SECONDS", 11.0))
GENERATION_INTERVAL_SECONDS = max(4.0, env_float("GENERATION_INTERVAL_SECONDS", 30.0))
MAX_PROMPT_WORDS = max(0, env_int("MAX_PROMPT_WORDS", 10))
MAX_INPUT_CHUNK_WORDS = max(1, env_int("MAX_INPUT_CHUNK_WORDS", 4))
WEIRD_DIRECTIVE_THRESHOLD = max(1, env_int("WEIRD_DIRECTIVE_THRESHOLD", 7))
WEIRD_DIRECTIVE_CHANCE = min(
    1.0, max(0.0, env_float("WEIRD_DIRECTIVE_CHANCE", 0.5))
)
MAX_STORY_LINES = max(20, env_int("MAX_STORY_LINES", 1000))
MAX_UPLOAD_MB = max(1, env_int("MAX_UPLOAD_MB", 25))

# The story keeps a compact continuity memo plus the newest lines. This avoids
# sending an indefinitely growing transcript while preserving consequences of
# high-energy turns after the audience words have vanished.
RECENT_CONTEXT_LINES = max(4, env_int("RECENT_CONTEXT_LINES", 12))
SUMMARY_BATCH_LINES = max(3, env_int("SUMMARY_BATCH_LINES", 8))
SUMMARY_MAX_WORDS = max(80, env_int("SUMMARY_MAX_WORDS", 180))

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en").strip() or None
WHISPER_BEAM_SIZE = max(1, env_int("WHISPER_BEAM_SIZE", 1))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError(
        f"OPENAI_API_KEY was not found or was empty in: {ENV_PATH}"
    )

print(
    f"Using API key from {ENV_PATH} ending in …{OPENAI_API_KEY[-8:]}",
    flush=True,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

openai_client = OpenAI(api_key=OPENAI_API_KEY)

state_lock = threading.RLock()
whisper_lock = threading.Lock()
generation_lock = threading.Lock()

# Repeated occurrences stay separate. Each contributes its own decaying weight,
# and the browser uses the number of living repetitions to enlarge the word.
word_occurrences: list[dict[str, Any]] = []
story_lines: list[str] = []
story_summary = ""
story_summary_upto = 0

_whisper_model: WhisperModel | None = None

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


CALM_PARK_FALLBACKS = [
    "We follow the path.",
    "We pass beneath leaves.",
    "We hear water nearby.",
    "We cross the shade.",
    "We stop beside the pond.",
    "We feel gravel underfoot.",
    "We walk through wet grass.",
]

OPENING_BEAT_DIRECTIVES = [
    "Begin with us already walking; notice one plain feature of the path, trees, grass, weather, pond, birds, or benches.",
    "Describe our steady movement through the park and one unremarkable sensory detail.",
    "Keep the line spare and calm: a path, a bodily sensation, and an ordinary thing we pass.",
]

ACTIVE_BEAT_DIRECTIVES = [
    "Let the required words alter something we directly see, hear, touch, smell, feel, remember, decide, or physically do.",
    "Make the next stretch of path disclose a concrete development through sensory experience rather than explanation.",
    "Let one required word become the cause of a visible action and another become its consequence.",
    "Change our route, bodies, attention, surroundings, or immediate purpose through a concrete occurrence.",
    "Allow an animal, object, plant, stranger, weather pattern, memory, or impossible thing in the park to act upon us.",
    "Turn an existing detail into the next bodily action or perception without stepping outside the walk to explain it.",
    "Let a previous oddity continue and use the new words to push it into a fresh, perceptible form.",
    "Move from one sensory fact to another until the unlikely words belong to the same physical moment.",
]

WEIRD_BEAT_DIRECTIVES = [
    "Use an Ashbery-like associative drift: let an ordinary park detail slide through several surprising but perceptible relations before the sentence lands.",
    "Use dense, Prynne-like compression: technical diction, sharp parataxis, unstable scale, and exact physical particulars, without quoting any poem.",
    "Make syntax behave like weather crossing the park: clauses change pressure, direction, and scale while every image remains bodily present.",
    "Let one concrete perception misread another, then let the mistake become physically true before the sentence ends.",
    "Use a lucid dream-logic in which each required word opens a new local law of matter, sensation, memory, or causality.",
    "Build the sentence through abrupt changes of register—botanical, domestic, anatomical, financial, mechanical—while keeping one continuous event visible.",
    "Let pronouns, distances, and sizes become briefly unstable, but keep our feet, hands, breath, and surroundings exact.",
    "Make the required words form a strange chain of causes whose final consequence is concrete enough to touch.",
]

AFTERMATH_BEAT_DIRECTIVES = [
    "State one small bodily action in plain language.",
    "Name one visible park detail and stop.",
    "Describe one quiet sensation in one short clause.",
    "Continue one existing task with a single simple action.",
    "Show one concrete thing remaining beside us.",
]


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        with whisper_lock:
            if _whisper_model is None:
                print(
                    f"Loading faster-whisper model {WHISPER_MODEL_NAME!r} "
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
    """Return individual spoken words, normalized for visible repetition."""
    return [match.group(0).lower().replace("’", "'") for match in WORD_RE.finditer(text)]


def prune_words_locked(now: float) -> None:
    cutoff = now - WORD_LIFETIME_SECONDS
    word_occurrences[:] = [item for item in word_occurrences if item["added_at"] > cutoff]


def split_input_chunks(words: list[str]) -> list[list[str]]:
    """Keep short spoken bursts intact; randomly partition longer ones.

    The original spoken order is preserved. A burst of four words or fewer is
    one indivisible prompt unit. Longer bursts are divided at random into
    chunks containing at most MAX_INPUT_CHUNK_WORDS words.
    """
    if not words:
        return []
    if len(words) <= MAX_INPUT_CHUNK_WORDS:
        return [list(words)]

    # Use the fewest chunks needed, but randomize where the boundaries fall.
    # Five words therefore become two chunks, nine words become three, etc.
    chunk_count = math.ceil(len(words) / MAX_INPUT_CHUNK_WORDS)
    chunk_sizes = [1] * chunk_count
    unassigned = len(words) - chunk_count
    while unassigned:
        available = [
            index
            for index, size in enumerate(chunk_sizes)
            if size < MAX_INPUT_CHUNK_WORDS
        ]
        chunk_sizes[random.choice(available)] += 1
        unassigned -= 1

    chunks: list[list[str]] = []
    cursor = 0
    for size in chunk_sizes:
        chunks.append(words[cursor : cursor + size])
        cursor += size
    return chunks


def add_words(words: list[str]) -> list[dict[str, Any]]:
    now = time.time()
    added: list[dict[str, Any]] = []
    chunks = split_input_chunks(words)
    spoken_index = 0

    with state_lock:
        prune_words_locked(now)
        for chunk_number, chunk in enumerate(chunks):
            chunk_id = uuid4().hex
            for position_in_chunk, word in enumerate(chunk):
                item = {
                    "id": uuid4().hex,
                    "word": word,
                    "chunk_id": chunk_id,
                    "chunk_number": chunk_number,
                    "position_in_chunk": position_in_chunk,
                    # Tiny offsets preserve spoken order without changing decay.
                    "added_at": now + spoken_index * 0.0001,
                }
                spoken_index += 1
                word_occurrences.append(item)
                added.append(item.copy())
    return added


def active_snapshot() -> tuple[float, list[dict[str, Any]], list[str]]:
    now = time.time()
    with state_lock:
        prune_words_locked(now)
        words = [item.copy() for item in word_occurrences]
        lines = list(story_lines)
    return now, words, lines


def weighted_sample_without_replacement(
    weights: dict[str, float], sample_size: int
) -> list[str]:
    """Efraimidis-Spirakis weighted sampling without replacement."""
    keyed: list[tuple[float, str]] = []
    for word, weight in weights.items():
        safe_weight = max(weight, 1e-9)
        key = random.random() ** (1.0 / safe_weight)
        keyed.append((key, word))
    keyed.sort(reverse=True)
    return [word for _, word in keyed[:sample_size]]


def weighted_sample_unique(now: float) -> tuple[list[str], float, dict[str, float]]:
    """Choose whole spoken chunks using repetition and freshness.

    A spoken burst of four words or fewer stays intact. Longer bursts were
    already partitioned into random chunks of at most four words by add_words().
    Selection happens at the chunk level, so words from one short burst are not
    torn apart merely because the prompt has limited space.
    """
    with state_lock:
        prune_words_locked(now)
        grouped: dict[str, float] = defaultdict(float)
        chunks: dict[str, dict[str, Any]] = {}

        for item in word_occurrences:
            age = max(0.0, now - float(item["added_at"]))
            occurrence_weight = math.exp(-age / WORD_DECAY_TAU_SECONDS)
            word = str(item["word"])
            grouped[word] += occurrence_weight

            chunk_id = str(item.get("chunk_id") or item["id"])
            chunk = chunks.setdefault(
                chunk_id,
                {"words": [], "local_weight": 0.0, "newest": 0.0},
            )
            if word not in chunk["words"]:
                chunk["words"].append(word)
            chunk["local_weight"] += occurrence_weight
            chunk["newest"] = max(chunk["newest"], float(item["added_at"]))

    if not grouped or not chunks or MAX_PROMPT_WORDS == 0:
        return [], 0.0, dict(grouped)

    total_fresh_weight = sum(grouped.values())
    raw_energy = 1.0 - math.exp(-total_fresh_weight / 5.5)
    desired_count = max(1, round(1 + raw_energy * (MAX_PROMPT_WORDS - 1)))

    # Repetition raises the score of chunks containing repeated words; local
    # freshness keeps recent spoken bursts ahead of old ones. Dividing by the
    # square root of chunk length avoids favoring a chunk merely for being long.
    chunk_weights: dict[str, float] = {}
    for chunk_id, chunk in chunks.items():
        words = chunk["words"]
        global_support = sum(grouped[word] for word in words)
        chunk_weights[chunk_id] = (
            global_support + float(chunk["local_weight"])
        ) / math.sqrt(max(1, len(words)))

    ordered_chunk_ids = weighted_sample_without_replacement(
        chunk_weights, len(chunk_weights)
    )

    selected: list[str] = []
    for chunk_id in ordered_chunk_ids:
        chunk_words = [word for word in chunks[chunk_id]["words"] if word not in selected]
        if not chunk_words:
            continue
        if len(selected) + len(chunk_words) > MAX_PROMPT_WORDS:
            continue
        selected.extend(chunk_words)
        if len(selected) >= desired_count:
            break

    if not selected:
        # MAX_INPUT_CHUNK_WORDS is normally below MAX_PROMPT_WORDS, so this is
        # only a defensive fallback for unusual environment settings.
        best_chunk_id = max(chunk_weights, key=chunk_weights.get)
        selected = chunks[best_chunk_id]["words"][:MAX_PROMPT_WORDS]

    coverage = len(selected) / max(1, MAX_PROMPT_WORDS)
    energy = min(1.0, 0.22 + 0.55 * raw_energy + 0.23 * coverage)
    return selected, energy, dict(grouped)


def join_words_for_sentence(words: list[str]) -> str:
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return ", ".join(words[:-1]) + f", and {words[-1]}"


def sentence_key(text: str) -> str:
    """Normalize a sentence for exact-repeat detection."""
    return " ".join(normalized_story_tokens(text))


def is_exact_repeat(sentence: str, previous_lines: list[str]) -> bool:
    key = sentence_key(sentence)
    return bool(key) and any(sentence_key(line) == key for line in previous_lines)


def local_fallback_sentence(selected: list[str], previous_lines: list[str]) -> str:
    """Return a concrete calm line that has never appeared before."""
    used = {sentence_key(line) for line in previous_lines}

    candidates = list(CALM_PARK_FALLBACKS)
    random.shuffle(candidates)
    for candidate in candidates:
        if sentence_key(candidate) not in used:
            return candidate

    actions = [
        "walk along the gravel",
        "pause beside the pond",
        "cross the damp grass",
        "stand beneath the sycamore",
        "follow the path",
        "sit on the bench",
        "step around a puddle",
        "hear a sparrow",
        "feel the wind",
        "watch the branches",
        "smell wet soil",
        "see pond light",
    ]
    for _ in range(100):
        candidate = f"We {random.choice(actions)}."
        if sentence_key(candidate) not in used:
            return candidate

    # The counter makes a literal duplicate impossible after a long run while
    # remaining a short, concrete physical action.
    return f"We take {len(previous_lines) + 1} steps."


def clean_one_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = cleaned.strip("`\"' ")
    if not cleaned:
        return ""

    # Keep one sentence if the model ignores the output instruction.
    match = re.match(r"^(.+?[.!?])(?:\s+|$)", cleaned)
    if match:
        cleaned = match.group(1)
    else:
        cleaned = cleaned.rstrip(";,:-") + "."

    return cleaned[:2600]


def contains_word(sentence: str, word: str) -> bool:
    return re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", sentence, re.I) is not None


def normalized_story_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def novelty_problem(sentence: str, previous_lines: list[str]) -> str | None:
    """Reject meta narration, wrong viewpoint, or any exact replay."""
    lowered = sentence.lower()
    stale_scaffolds = [
        "wrote down",
        "thought about the word",
        "considered the word",
        "the word made",
        "the situation",
        "the established reality",
        "the altered reality",
        "the new reality",
        "the current state",
        "new equilibrium",
        "what was already happening",
        "the same ordinary task",
        "returned to the same",
        "dealt with it in the most ordinary way",
        "continued unchanged",
        "the story",
        "this scene",
    ]
    for phrase in stale_scaffolds:
        if phrase in lowered:
            return f"The draft used the abstract or stale scaffold {phrase!r}."

    tokens = normalized_story_tokens(sentence)
    if is_exact_repeat(sentence, previous_lines):
        return "The draft exactly repeated a sentence that already appeared."
    if "alex" in tokens:
        return "The draft reintroduced Alex instead of using first-person plural."
    if not ({"we", "us", "our"} & set(tokens)):
        return "The draft left the first-person-plural park walk."

    candidate_norm = " ".join(tokens)
    if not tokens:
        return "The draft contained no usable language."

    # Ignore an initial 'we' when comparing openings, since the fixed viewpoint
    # should not itself count as repetition.
    candidate_stem = tokens[1:6] if tokens[:1] == ["we"] else tokens[:5]
    for line in previous_lines[-10:]:
        previous_tokens = normalized_story_tokens(line)
        previous_norm = " ".join(previous_tokens)
        if not previous_tokens:
            continue

        similarity = difflib.SequenceMatcher(None, candidate_norm, previous_norm).ratio()
        if similarity >= 0.70:
            return "The draft was too close in wording and action to a recent line."

        previous_stem = previous_tokens[1:6] if previous_tokens[:1] == ["we"] else previous_tokens[:5]
        if len(candidate_stem) >= 4 and candidate_stem[:4] == previous_stem[:4]:
            return "The draft repeated a recent sentence opening and grammatical frame."

        candidate_set = set(tokens)
        previous_set = set(previous_tokens)
        union = candidate_set | previous_set
        if union and len(candidate_set & previous_set) / len(union) >= 0.72:
            return "The draft reused too much of a recent line without a new perceptual beat."

    return None


def integration_problem(sentence: str, selected: list[str]) -> str | None:
    """Reject obvious token-as-character shortcuts.

    This is intentionally conservative: it catches the recurring failure where
    required words are merely coordinated into a list and made to "meet,"
    "gather," or otherwise behave as named beings instead of being used in
    ordinary grammatical phrases inside the scene.
    """
    if not selected:
        return None

    lowered = sentence.lower()
    alternatives = "|".join(
        sorted((re.escape(word.lower()) for word in selected), key=len, reverse=True)
    )
    token = rf"(?:{alternatives})"

    # "smile and humongous meet" / "pickle, simon, and crimp gather"
    coordinated_tokens = rf"{token}(?:\s*,\s*|\s+and\s+){token}(?:(?:\s*,\s*|\s+and\s+){token})*"
    actor_verbs = (
        r"meet|gather|assemble|arrive|appear|enter|wait|stand|sit|walk|move|"
        r"follow|lead|speak|talk|argue|agree|dance|watch|look|become|turn"
    )
    if re.search(rf"\b{coordinated_tokens}\s+(?:{actor_verbs})(?:s|ed|ing)?\b", lowered):
        return "The draft turned the required tokens into a list of characters or actors."

    # Explicitly discussing a token rather than instantiating its meaning.
    for word in selected:
        escaped = re.escape(word.lower())
        patterns = [
            rf"\bthe\s+word\s+['\"]?{escaped}\b",
            rf"\b{escaped}\s+(?:means|represents|symbolizes|stands\s+for)\b",
            rf"\b(?:say|saying|said|repeat|repeating|hear|hearing)\s+['\"]?{escaped}['\"]?\b",
        ]
        if any(re.search(pattern, lowered) for pattern in patterns):
            return f"The draft discussed {word!r} as a word instead of placing its meaning in the scene."

    return None


def call_openai_text(instructions: str, prompt: str, max_output_tokens: int) -> str:
    if openai_client is None:
        return ""

    if hasattr(openai_client, "responses"):
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        return getattr(response, "output_text", "") or ""

    chat_kwargs = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        completion = openai_client.chat.completions.create(
            **chat_kwargs,
            max_completion_tokens=max_output_tokens,
        )
    except (TypeError, ValueError) as exc:
        # Older 1.x clients call this field max_tokens instead. This path is
        # only used when client.responses is unavailable.
        if "max_completion_tokens" not in str(exc) and not isinstance(exc, TypeError):
            raise
        completion = openai_client.chat.completions.create(
            **chat_kwargs,
            max_tokens=max_output_tokens,
        )
    return completion.choices[0].message.content or ""


def maybe_update_story_summary() -> None:
    """Fold older lines into a compact continuity memo when needed."""
    global story_summary, story_summary_upto

    if openai_client is None:
        return

    with state_lock:
        target = max(0, len(story_lines) - RECENT_CONTEXT_LINES)
        if target - story_summary_upto < SUMMARY_BATCH_LINES:
            return
        old_upto = story_summary_upto
        old_summary = story_summary
        new_lines = list(story_lines[old_upto:target])

    instructions = (
        "Maintain a compact continuity memo for a first-person-plural walk through a park. Preserve only concrete "
        "facts future lines can perceive or act within: our location and route, visible objects and creatures, bodily "
        "states and injuries, things we carry or feed, people present, weather, transformations, ongoing tasks, and "
        "unresolved physical consequences. Treat bizarre developments as literal facts. Preserve concrete vocabulary "
        "even after the audience word that introduced it has vanished. Do not add interpretation, themes, abstract "
        "labels, prompts, audience words, generation, or writing instructions. Output only the memo."
    )
    prompt = f"""Existing continuity memo:
{old_summary or "(none yet)"}

New story lines to absorb:
{chr(10).join(new_lines)}

Rewrite the continuity memo in no more than {SUMMARY_MAX_WORDS} words. Keep details that future sentences need in order not to reset the story.
"""

    try:
        updated = re.sub(r"\s+", " ", call_openai_text(instructions, prompt, 300).strip())
        if not updated:
            return
        with state_lock:
            # Reset may have happened while the API request was in flight.
            if story_summary_upto == old_upto and len(story_lines) >= target:
                story_summary = updated
                story_summary_upto = target
    except Exception as exc:
        print(f"Story summary update failed: {exc}", flush=True)


def current_story_context() -> tuple[str, list[str]]:
    with state_lock:
        summary = story_summary
        upto = story_summary_upto
        lines = list(story_lines)

    # Normally the summary ends exactly where the recent tail begins. If a
    # summary request failed, include a wider unsummarized tail rather than lose
    # continuity altogether.
    start = max(upto, len(lines) - max(RECENT_CONTEXT_LINES, 24))
    return summary, lines[start:]


def energy_description(selected_count: int, energy: float, opening: bool = False) -> str:
    if opening:
        return "OPENING CALM: we are already walking through a park; give one spare, ordinary sensory line"
    if selected_count == 0:
        return "CALM CONTINUATION: no new rupture; directly perceive and manage the park exactly as it now is"
    if energy < 0.45:
        return "AWAKENED: even one human word must become a concrete new perception, gesture, object, feeling, or thought"
    if energy < 0.68:
        return "MOBILE: the words may redirect our route, attention, bodies, or immediate surroundings"
    if energy < 0.86:
        return "VOLATILE: permit a major sensory or causal shift, still experienced directly from inside the walk"
    return "MAXIMAL: the park and our bodies may become unbelievable, but every change must be seen, felt, heard, thought, or done"


def generate_with_openai(
    selected: list[str],
    energy: float,
    summary: str,
    recent_lines: list[str],
    all_story_lines: list[str],
) -> str:
    if openai_client is None:
        return local_fallback_sentence(selected, all_story_lines)

    selected_count = len(selected)
    opening = not summary and not recent_lines
    if selected_count == 0:
        min_words, max_words = 3, 9
    else:
        # Awkward combinations are allowed enough syntactic room to become one
        # meaningful perception or event rather than a compressed list.
        min_words = 22 + 8 * (selected_count - 1)
        max_words = 48 + 14 * (selected_count - 1)

    allow_rare_flourish = bool(selected) and random.random() < (0.10 + 0.25 * energy)
    rare_flourish_rule = (
        "After using every required word exactly, you may add one rarer related word if it arises naturally inside the perception."
        if allow_rare_flourish
        else "Do not substitute synonyms for the required words; use every one exactly as supplied."
    )

    continuity = summary or "(No older continuity memo yet.)"
    recent_story = "\n".join(recent_lines) or "(No lines yet; begin with us already walking through a park.)"
    selected_text = ", ".join(selected) if selected else "(none)"
    mode = energy_description(selected_count, energy, opening)
    if opening and not selected:
        beat_directive = random.choice(OPENING_BEAT_DIRECTIVES)
    elif not selected:
        beat_directive = random.choice(AFTERMATH_BEAT_DIRECTIVES)
    elif (
        selected_count >= WEIRD_DIRECTIVE_THRESHOLD
        and random.random() < WEIRD_DIRECTIVE_CHANCE
    ):
        beat_directive = random.choice(WEIRD_BEAT_DIRECTIVES)
    else:
        beat_directive = random.choice(ACTIVE_BEAT_DIRECTIVES)

    calm_syntax_rule = (
        " With no required words, use only one short independent clause of 3 to 9 words: plain subject, verb, and concrete object or sensation; no semicolon, dash, parenthesis, subordinate clause, or compound action."
        if not selected
        else ""
    )
    instructions = (
        "Write exactly one sentence in first-person plural, present tense, continuing our walk through a park. "
        "Output only the sentence: no label, quotation marks, notes, explanation, or second sentence. "
        "Remain completely inside immediate experience: what we see, hear, smell, touch, taste, physically feel, "
        "remember in a specific image, think in actual words, or do with our bodies. Never narrate the premise, "
        "state of the story, energy level, altered reality, situation, continuity, or meaning from above."
        + calm_syntax_rule
    )

    prompt = f"""CONTINUITY MEMO:
{continuity}

MOST RECENT LINES:
{recent_story}

REQUIRED LIVING WORDS FOR THIS TURN:
{selected_text}

ENERGY MODE:
{mode}

NEXT PERCEPTUAL IMPULSE:
{beat_directive}

TARGET LENGTH:
Approximately {min_words} to {max_words} words, all in ONE sentence. With no required words, write one laconic independent clause and stop. When many words are present, the sentence may grow very long through controlled clauses, semicolons, dashes, parentheses, and associative turns; do not cut it short merely to stay simple.

BASE FORM OF THE WALK:
- We are always somewhere in or immediately around a park, walking, pausing, crouching, sitting, climbing, carrying, feeding, bleeding, hiding, looking, listening, remembering, or otherwise bodily present.
- With no living words, use a grammatically simple, laconic line of 3 to 9 words: one independent clause, one action or sensation, no compound sentence, and no new development.
- Use plain concrete vocabulary in calm mode: path, grass, trees, water, weather, birds, benches, shade, breath, feet, hands, and whatever concrete oddities already remain.
- Calm does not mean abstract summary. Write the actual small thing: our shoe slips in mud; we wipe blood from a thumb; we count pennies into the dragon's mouth; leaves touch our sleeves.
- Vary the object, sensation, syntax, and next physical beat so consecutive lines do not become the same template.
- Never repeat any earlier sentence verbatim, even when the same required words recur; continue from the last line with a genuinely new perception, action, thought, or consequence.

THE WORD-COMBINATION GAME:
- When required words (or short phrases or chunks) are present, use ALL OF THEM in the same sentence, exactly as supplied.
- The selected words arrive in intact spoken chunks: do not omit part of a chunk. A short spoken burst of up to four words has been deliberately kept together; longer bursts were randomly divided into chunks of at most four before selection.
- You are judged first on complete exact-word coverage and then on how cleverly you make apparently incompatible words belong to one perceptible scene, action, mental state, or chain of events.
- The required words are raw language to be USED; they are not names of characters, labels, magical tokens, or topics for discussion.
- Silently decide an ordinary grammatical role and a concrete bearer for each word before writing. Do not output that plan.
- Adjectives must modify something we can identify; verbs must actually happen; nouns must name an object, being, substance, body part, place, feeling, thought-content, or other intelligible referent; adverbs must modify an action or quality.
- Ask silently: WHO smiles? WHAT is humongous? WHAT contains the money? WHO performs the verb? The sentence must answer those questions through its images and actions.
- Do not mention, quote, list, read, write down, define, discuss, or generically “think about” the required words.
- A thought must be the actual content of thought in the moment—an image, fear, memory, calculation, wish, or decision—not a report that we considered a vocabulary item.
- Embed every word inside natural syntax, with articles, prepositions, auxiliaries, possessives, and subordinate clauses as needed; the reader should be able to picture or feel what happened without ever thinking about a word game.
- Let the words alter what is seen and felt and, when their pressure is strong, change our route, bodies, companions, weather, scale, causality, or understanding of the park.
- When realism cannot contain the combination, use controlled poetic association: one concrete image transforms into the next, syntax may stretch, and the park may become impossible, but every required word must still perform a clear grammatical and perceptual function.
- This sentence can get weird and it can get long, but EVERY SINGLE WORD (or chunk) SUPPLIED MUST BE USED.  You fail if you don't use all of them.  If there are a lot, just make a really long rambling sentence that uses them all.
- {rare_flourish_rule}

CONTINUITY AND CALM:
- Preserve concrete consequences. If we began bleeding, we may still be bleeding after that screen word vanishes; if we are feeding a dragon pennies in a bush, later calm lines may quietly continue the feeding.
- Vanished audience words are not forbidden vocabulary. Use any previous word naturally when the ongoing physical world requires it.
- Living words inject novelty, action, and transformation. No living words means no fresh rupture, revelation, new creature, new danger, new magical rule, or abrupt relocation.
- In calm continuation, we may still walk, look, breathe, wipe, carry, feed, rest, adjust, or notice minor details within the changed baseline.
- Never use abstract placeholders such as “the situation,” “the altered reality,” “the established condition,” “the new equilibrium,” “what was already happening,” or “the ordinary way available.” Show the blood, coin, bush, dragon, mud, hand, sound, pressure, or movement instead.
- Never mention the audience, prompt, transcription, selected words, generation, game, rules, story, sentence, or energy inside the fiction.
"""

    try:
        raw_text = call_openai_text(instructions, prompt, 1200)
        sentence = clean_one_sentence(raw_text)
        if not sentence:
            return local_fallback_sentence(selected, all_story_lines)

        missing = [word for word in selected if not contains_word(sentence, word)]
        problem = novelty_problem(sentence, all_story_lines)
        integration = integration_problem(sentence, selected)
        if not missing and problem is None and integration is None:
            return sentence

        issues: list[str] = []
        if missing:
            issues.append(f"It omitted these exact required words: {', '.join(missing)}.")
        if problem:
            issues.append(problem)
        if integration:
            issues.append(integration)

        repair_prompt = f"""Replace this failed draft:
{sentence}

WHY IT FAILED:
{' '.join(issues)}

CONTINUITY MEMO:
{continuity}

RECENT WALK:
{recent_story}

ALL EXACT REQUIRED WORDS:
{selected_text}

Write a completely revised ONE-SENTENCE first-person-plural, present-tense next beat of approximately {min_words} to {max_words} words.
- If there are no required words, use one laconic independent clause of 3 to 9 words and no compound action.
- Use every required word exactly, in its ordinary grammatical role, inside what we see, feel, think, or do.
- Do not turn the required words into names or actors and do not coordinate them as a list; attach each adjective to a concrete noun, let each verb happen, and give each noun a specific referent.
- Silently identify the bearer of every quality and the agent or object of every action before writing.
- Stay in the park and inside immediate bodily perception.
- Do not explain the situation, reality, continuity, premise, or meaning.
- Change the opening, main action, imagery, and sentence architecture of the rejected draft.
- Preserve concrete ongoing facts; add a new rupture only when living words are present.
- Output only the replacement sentence.
"""
        repaired = clean_one_sentence(call_openai_text(instructions, repair_prompt, 1200))
        repaired_missing = [word for word in selected if not contains_word(repaired, word)]
        repaired_problem = novelty_problem(repaired, all_story_lines) if repaired else "The repair was empty."
        repaired_integration = integration_problem(repaired, selected) if repaired else "The repair was empty."
        if (
            repaired
            and not repaired_missing
            and repaired_problem is None
            and repaired_integration is None
        ):
            return repaired

        if not selected:
            return local_fallback_sentence(selected, all_story_lines)

        rescue_prompt = f"""Continue this first-person-plural walk through a park in exactly one long sentence:
{recent_story}

You MUST include every exact word below:
{selected_text}

The sentence may be up to {max_words + 40} words. First make a silent grammatical plan for each required word: identify what concrete noun an adjective modifies, who or what performs a verb, and what actual referent a noun names. Then write only the sentence. Do not turn the words into names or characters, do not coordinate them into a list, and do not discuss them as vocabulary. Make them function inside concrete scenery, bodily sensation, mental content, and events. Output only the sentence.
"""
        rescued = clean_one_sentence(call_openai_text(instructions, rescue_prompt, 1400))
        rescued_missing = [word for word in selected if not contains_word(rescued, word)]
        rescued_problem = novelty_problem(rescued, all_story_lines) if rescued else "The rescue was empty."
        rescued_integration = integration_problem(rescued, selected) if rescued else "The rescue was empty."
        if (
            rescued
            and not rescued_missing
            and rescued_problem is None
            and rescued_integration is None
        ):
            return rescued

        print(
            "Generation validation failed after all attempts; using a calm concrete fallback. "
            f"selected={selected!r}, initial_problem={problem!r}, integration={integration!r}, "
            f"repair_problem={repaired_problem!r}, repair_integration={repaired_integration!r}, "
            f"rescue_problem={rescued_problem!r}, rescue_integration={rescued_integration!r}",
            flush=True,
        )
        return local_fallback_sentence(selected, all_story_lines)
    except Exception as exc:
        print(f"OpenAI generation failed: {exc}", flush=True)
        return local_fallback_sentence(selected, all_story_lines)


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
    upload = request.files.get("audio")
    if upload is None or not upload.filename:
        return jsonify({"error": "No audio upload was received."}), 400

    suffix = audio_suffix(upload.mimetype)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            upload.save(temp_file)
            temp_path = Path(temp_file.name)

        model = get_whisper_model()
        with whisper_lock:
            segments, info = model.transcribe(
                str(temp_path),
                language=WHISPER_LANGUAGE,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()

        words = tokenize(transcript)
        added = add_words(words)
        return jsonify(
            {
                "transcript": transcript,
                "words": [item["word"] for item in added],
                "language": getattr(info, "language", WHISPER_LANGUAGE),
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
        maybe_update_story_summary()
        now = time.time()
        selected, energy, weights = weighted_sample_unique(now)
        summary, recent = current_story_context()
        with state_lock:
            all_story_lines = list(story_lines)

        sentence = generate_with_openai(
            selected, energy, summary, recent, all_story_lines
        )

        with state_lock:
            # Final server-side guarantee: no line can be appended twice, even
            # if an API repair fails or a fallback path is used.
            if is_exact_repeat(sentence, story_lines):
                sentence = local_fallback_sentence(selected, story_lines)
            story_lines.append(sentence)
            if len(story_lines) > MAX_STORY_LINES:
                # MAX_STORY_LINES is intentionally generous. The summary keeps
                # continuity, while this cap prevents an installation left on
                # for days from growing without bound.
                overflow = len(story_lines) - MAX_STORY_LINES
                del story_lines[:overflow]
                global story_summary_upto
                story_summary_upto = max(0, story_summary_upto - overflow)

        return jsonify(
            {
                "sentence": sentence,
                "selected_words": selected,
                "energy": energy,
                "weights": weights,
            }
        )
    finally:
        generation_lock.release()


@app.post("/api/reset")
def api_reset():
    global story_summary, story_summary_upto
    with state_lock:
        word_occurrences.clear()
        story_lines.clear()
        story_summary = ""
        story_summary_upto = 0
    return jsonify({"ok": True})


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error: RequestEntityTooLarge):
    return jsonify({"error": f"Audio upload exceeds {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    get_whisper_model()
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=env_int("PORT", 5000),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        use_reloader=False,
        threaded=True,
    )
