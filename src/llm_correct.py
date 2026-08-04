"""Stage 6: LLM post-correction via Synthetic.new (OpenAI-compatible API).

Uses gpt-oss-120b — returns content within token budget on Tibetan/Sanskrit
segments (unlike reasoning-first models that exhaust tokens on thinking).
Dynamic max_tokens via tiktoken estimation keeps us within model limits
regardless of segment length.

Token budgeting:
  - gpt-oss-120b has 128k context window
  - Estimate input tokens with tiktoken (cl100k_base approximation)
  - max_tokens = min(generous_cap, context_window - estimated_input - margin)
  - For a 5-hour lecture with 2000 segments at ~3s/segment: ~100 min total
"""
import json
import re
import logging

import tiktoken
from openai import OpenAI

from config import (
    SYNTHETIC_API_URL,
    SYNTHETIC_API_KEY,
    LLM_MODEL,
    LLM_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

MODEL_CONTEXT_WINDOW = 128_000  # gpt-oss-120b
SAFETY_MARGIN = 2_048  # buffer between input + output and context limit
DEFAULT_MAX_TOKENS = 4_096  # per-segment output cap (corrections are short)
MAX_MAX_TOKENS = 8_192  # absolute cap per segment

# cl100k_base is a reasonable approximation for non-OpenAI models.
# It overestimates slightly for non-Latin scripts, which is conservative.
_ENCODER = tiktoken.get_encoding("cl100k_base")

# --- System prompt ------------------------------------------------------------

# Core instructions — lean by design. The dharma knowledge base is injected
# as a separate user-context block only when relevant, not on every request.
# This prevents reasoning models from spiraling on cross-referencing.
SYSTEM_PROMPT = """\
You are a correction engine for Buddhist dharma teaching transcripts.
The transcripts contain English, Tibetan (bo), Sanskrit (sa), and occasionally Japanese (ja).

Correct transcription errors: misheard terms, garbled mantras, phonetic renderings.
Do NOT rephrase, add information, or restructure. Only fix specific errors.

Return ONLY a JSON object:
{"corrected": "...", "confidence": "high|low|none", "changes": ["was -> now", ...]}
"""

# Dharma domain reference — sent as context, not as instructions.
# This gives the model vocabulary without triggering exhaustive analysis.
DHARMA_REFERENCE = """\
Dharma vocabulary for reference:
Teachers: Garchen Rinpoche, Chogyal Namkhai Norbu, Lama Fede Andino, Tenga Rinpoche, Karmapa
Terms: bodhicitta, shunyata, vajra, mantra, mandala, empowerments, lungta, damaru, Sowa Rigpa
Sutras: Prajnaparamita, Vajracchedika, Heart Sutra
Lineage: Padmasambhava, Yeshe Tsogyal, Naropa, Tilopa, Marpa, Milarepa, Gampopa, Atisha, Tsongkhapa
Teachings: Dzogchen, Mahamudra, Three Jewels, Four Noble Truths, Eightfold Path
Regions: Amdo, Kham, Utsang
"""


# --- Token estimation --------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Estimate token count using cl100k_base encoding."""
    return len(_ENCODER.encode(text))


def _compute_max_tokens(*texts: str) -> int:
    """Compute safe max_tokens given input texts.

    Returns the token budget for model output, leaving room for input
    within the model's context window. For short segments this returns
    the default cap; for very long segments it scales down.
    """
    input_estimate = sum(_estimate_tokens(t) for t in texts)
    available = MODEL_CONTEXT_WINDOW - input_estimate - SAFETY_MARGIN
    return max(min(available, DEFAULT_MAX_TOKENS), 512)


# --- Client ------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy singleton — avoids constructing client at import time."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=SYNTHETIC_API_URL,
            api_key=SYNTHETIC_API_KEY,
        )
    return _client


# --- Core --------------------------------------------------------------------

def _parse_json_response(content: str) -> dict | None:
    """Extract a JSON object from an LLM response.

    Handles markdown code fences and extraneous text around the JSON.
    """
    if not content:
        return None
    clean = content.strip()
    # Strip markdown fences
    if clean.startswith("```"):
        parts = clean.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                clean = part
                break
    if clean.endswith("```"):
        clean = clean[:-3].strip()
    # Greedy match for outermost JSON object
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        clean = match.group(0)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


def correct_segment(
    segment: dict,
    prev_text: str = "",
    next_text: str = "",
) -> dict:
    """Send a single segment to the LLM for correction.

    Returns a dict with keys: corrected, confidence, changes, and
    optionally error. Never raises — errors are returned in the dict.
    """
    seg_text = segment.get("text", "")
    if not seg_text or len(seg_text.strip()) < 3:
        return {"corrected": seg_text, "confidence": "none", "changes": []}

    user_prompt = (
        f"{DHARMA_REFERENCE}\n"
        f"Previous: {prev_text[-200:] if prev_text else '(none)'}\n"
        f"Next: {next_text[:200] if next_text else '(none)'}\n\n"
        f"Segment:\n{seg_text}\n\n"
        f"Return JSON only."
    )

    max_tokens = _compute_max_tokens(SYSTEM_PROMPT, user_prompt)

    try:
        resp = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning("API error on segment: %s", e)
        return {"corrected": seg_text, "confidence": "none", "changes": [],
                "error": str(e)}

    msg = resp.choices[0].message
    content = msg.content

    if not content:
        finish = resp.choices[0].finish_reason
        return {"corrected": seg_text, "confidence": "none", "changes": [],
                "error": f"Empty content (finish={finish})"}

    result = _parse_json_response(content)
    if result is None:
        return {"corrected": seg_text, "confidence": "none", "changes": [],
                "error": f"JSON parse failed: {content[:200]}"}

    return result


def llm_correct_transcript(transcript: dict) -> dict:
    """Run LLM correction across all segments in a transcript.

    Dictionary corrections are applied BEFORE this stage (in output.py).
    High-confidence corrections are applied directly; low-confidence
    suggestions are stored for the review queue.
    """
    segments = transcript.get("segments", [])
    if not segments:
        return transcript

    logger.info("Starting LLM correction with %s (%d segments)",
                LLM_MODEL, len(segments))
    print(f"  [stage6] LLM correction with {LLM_MODEL} ({len(segments)} segments)...",
          flush=True)

    corrections_log = []
    high_conf = 0
    low_conf = 0
    errors = 0

    for i, seg in enumerate(segments):
        seg_text = seg.get("text", "")
        if len(seg_text.strip()) < 3:
            continue

        prev_text = segments[i - 1].get("text", "") if i > 0 else ""
        next_text = segments[i + 1].get("text", "") if i < len(segments) - 1 else ""

        result = correct_segment(seg, prev_text, next_text)

        corrected = result.get("corrected", seg_text)
        confidence = result.get("confidence", "none")
        changes = result.get("changes", [])

        if corrected != seg_text and confidence == "high":
            seg["text_pre_llm"] = seg_text
            seg["text"] = corrected
            seg["llm_corrected"] = True
            seg["llm_confidence"] = confidence
            high_conf += 1
            corrections_log.append({
                "segment_id": i, "original": seg_text,
                "corrected": corrected, "changes": changes,
                "confidence": confidence,
            })
        elif confidence == "low":
            seg["llm_suggestion"] = corrected
            seg["llm_confidence"] = confidence
            low_conf += 1
            corrections_log.append({
                "segment_id": i, "original": seg_text,
                "suggested": corrected, "changes": changes,
                "confidence": confidence,
            })
        elif "error" in result:
            errors += 1
            corrections_log.append({
                "segment_id": i, "original": seg_text,
                "error": result["error"],
            })

        if (i + 1) % 50 == 0:
            print(f"  [stage6] Processed {i + 1}/{len(segments)}...", flush=True)

    transcript["llm_correction"] = {
        "model": LLM_MODEL,
        "api": "redacted",
        "segments_processed": len(segments),
        "high_confidence_corrections": high_conf,
        "low_confidence_suggestions": low_conf,
        "errors": errors,
    }
    transcript["llm_corrections_log"] = corrections_log

    print(f"  [stage6] Done: {high_conf} high-conf, {low_conf} low-conf, {errors} errors",
          flush=True)

    return transcript
