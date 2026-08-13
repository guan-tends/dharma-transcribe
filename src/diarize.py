"""Stage 4: Speaker diarization via pyannote.

Uses WhisperX's DiarizationPipeline (pyannote-audio under the hood) to
identify speakers and assign word-level speaker labels.
"""
from gpu import flush_gpu, vram_free_mb
import config


def diarize_transcript(transcript: dict, wav_path: str, hf_token: str) -> dict:
    """Run speaker diarization and assign word-level speakers."""
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    if not hf_token:
        print("  [stage4] No HF token — skipping diarization", flush=True)
        return transcript

    print("  [stage4] Loading pyannote diarization model...", flush=True)

    try:
        pipeline = DiarizationPipeline(
            token=hf_token,
            device=config.DEVICE,
        )
    except Exception as e:
        print(f"  [stage4] Failed to load diarization pipeline: {e}", flush=True)
        return transcript

    print("  [stage4] Diarizing...", flush=True)

    try:
        diarize_df = pipeline(wav_path)

        # Assign speakers to words/segments via interval-tree overlap
        transcript = assign_word_speakers(
            diarize_df,
            transcript,
        )

        # Count speakers
        speakers = {seg.get("speaker") for seg in transcript["segments"] if seg.get("speaker")}
        transcript["speaker_count"] = len(speakers)
        transcript["diarized"] = True

        print(f"  [stage4] Found {len(speakers)} speakers: {sorted(speakers)}", flush=True)

    except Exception as e:
        print(f"  [stage4] Diarization failed: {e}", flush=True)
        transcript["diarized"] = False

    del pipeline
    flush_gpu()

    if config.DEVICE != "cpu":
        print(f"  [stage4] Diarization model flushed. VRAM free: {vram_free_mb()}MB", flush=True)

    return transcript
