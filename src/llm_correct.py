"""Stage 6: LLM post-correction via Qwen3.5-9B (local Ollama)."""
import json
import requests
from typing import List
from config import OLLAMA_HOST, LLM_MODEL, LLM_CONFIDENCE_THRESHOLD

DHARMA_SYSTEM_PROMPT = """You are a correction engine for Buddhist dharma teaching transcripts. The transcripts contain English, Tibetan (bo), Sanskrit (sa), and occasionally Japanese (ja).

Your task: correct transcription errors in the given segment. Common errors:
- Misheard Tibetan/Sanskrit terms phonetically rendered as English words
- Misheard teacher names, lineage names, or technical dharma terms
- Garbled mantra transliterations

Rules:
1. Only correct specific words you are confident are errors. Do NOT rephrase.
2. Do NOT add information not present in the audio.
3. Do NOT restructure or reformat the text.
4. If you are uncertain, say so — do not guess.
5. Preserve speaker labels and timestamps.
6. Output JSON: {"corrected": "...", "confidence": "high|low|none", "changes": ["was → now", ...]}

Dharma domain knowledge:
- Sowa Rigpa = Traditional Tibetan Medicine (NOT "Swahili")
- Yuthok Yöntan Gombo = Father of Tibetan Medicine
- Garchen Rinpoche, Chogyal Namkhai Norbu, Lama Fede Andino = teacher names
- Bodhicitta, shunyata, vajra, mantra, mandala, empowerments = key terms
- Lungta, damaru, bell, vajra = ritual objects
- Amdo, Kham, Utsang = Tibetan regions
- The Three Jewels, Four Noble Truths, Eightfold Path = core teachings
- Prajnaparamita, Vajracchedika, Heart Sutra = sutra names
- Avalokiteshvara, Manjushri, Vajrapani = bodhisattvas
- Padmasambhava, Yeshe Tsogyal = Nyingma lineage figures
- Naropa, Tilopa, Marpa, Milarepa, Gampopa = Kagyu lineage
- Atisha, Lamp for the Path to Enlightenment = Kadampa
- Tsongkhapa, Lamrim = Gelug
- Dzogchen, Mahamudra = highest teachings
- Tenga Rinpoche, Karmapa, Shamarpa, Sitrupa = Kagyu lineage holders
- Ngedon Drupa = Secret Mantra
- Semde, Longde, Manngagde = Dzogchen three series

Return ONLY the JSON. No prose."""


def correct_segment(segment: dict, prev_text: str = "", next_text: str = "") -> dict:
    """Send a segment to Qwen3.5-9B for correction via Ollama API."""
    seg_text = segment.get("text", "")
    if not seg_text or len(seg_text.strip()) < 3:
        return {"corrected": seg_text, "confidence": "none", "changes": []}

    user_prompt = f"""Correct this transcript segment.

Previous context: {prev_text[-200:] if prev_text else '(none)'}
Next context: {next_text[:200] if next_text else '(none)'}

Segment to correct:
{seg_text}

Return JSON only."""

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": DHARMA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 512,
                },
            },
            timeout=120,
        )

        if response.status_code != 200:
            return {"corrected": seg_text, "confidence": "none", "changes": [],
                    "error": f"HTTP {response.status_code}"}

        llm_response = response.json().get("message", {}).get("content", "")

        # Parse JSON from response (handle markdown code fences)
        llm_clean = llm_response.strip()
        if llm_clean.startswith("```"):
            llm_clean = llm_clean.split("```")[1]
            if llm_clean.startswith("json"):
                llm_clean = llm_clean[4:]
            llm_clean = llm_clean.strip()

        result = json.loads(llm_clean)
        return result

    except json.JSONDecodeError:
        return {"corrected": seg_text, "confidence": "none", "changes": [],
                "error": "JSON parse failed"}
    except requests.exceptions.RequestException as e:
        return {"corrected": seg_text, "confidence": "none", "changes": [],
                "error": str(e)}
    except Exception as e:
        return {"corrected": seg_text, "confidence": "none", "changes": [],
                "error": str(e)}


def llm_correct_transcript(transcript: dict) -> dict:
    """
    Run LLM correction on all segments in the transcript.
    Dictionary corrections are applied BEFORE this stage (in output.py).
    """
    segments = transcript.get("segments", [])
    if not segments:
        return transcript

    print(f"  [stage6] Starting LLM correction with {LLM_MODEL}...", flush=True)

    corrections_log = []
    high_conf_count = 0
    low_conf_count = 0
    error_count = 0

    for i, seg in enumerate(segments):
        seg_text = seg.get("text", "")

        # Skip very short segments
        if len(seg_text.strip()) < 3:
            continue

        # Context from adjacent segments
        prev_text = segments[i-1].get("text", "") if i > 0 else ""
        next_text = segments[i+1].get("text", "") if i < len(segments) - 1 else ""

        result = correct_segment(seg, prev_text, next_text)

        corrected = result.get("corrected", seg_text)
        confidence = result.get("confidence", "none")
        changes = result.get("changes", [])

        if corrected != seg_text and confidence == "high":
            seg["text_pre_llm"] = seg_text
            seg["text"] = corrected
            seg["llm_corrected"] = True
            seg["llm_confidence"] = confidence
            high_conf_count += 1
            corrections_log.append({
                "segment_id": i,
                "original": seg_text,
                "corrected": corrected,
                "changes": changes,
                "confidence": confidence,
            })
        elif confidence == "low":
            seg["llm_suggestion"] = corrected
            seg["llm_confidence"] = confidence
            low_conf_count += 1
            corrections_log.append({
                "segment_id": i,
                "original": seg_text,
                "suggested": corrected,
                "changes": changes,
                "confidence": confidence,
            })
        elif "error" in result:
            error_count += 1

        if (i + 1) % 50 == 0:
            print(f"  [stage6] Processed {i+1}/{len(segments)} segments...", flush=True)

    transcript["llm_correction"] = {
        "model": LLM_MODEL,
        "segments_processed": len(segments),
        "high_confidence_corrections": high_conf_count,
        "low_confidence_suggestions": low_conf_count,
        "errors": error_count,
    }
    transcript["llm_corrections_log"] = corrections_log

    print(f"  [stage6] Done: {high_conf_count} high-conf, {low_conf_count} low-conf, {error_count} errors", flush=True)

    return transcript
