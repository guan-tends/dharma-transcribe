"""Stage 4: Speaker diarization via pyannote."""
from gpu import flush_gpu, vram_free_mb
from config import DEVICE


def diarize_transcript(transcript: dict, wav_path: str, hf_token: str) -> dict:
    """
    Run speaker diarization and assign word-level speakers.
    """
    import whisperx

    if not hf_token:
        print("  [stage4] No HF token — skipping diarization", flush=True)
        return transcript

    print(f"  [stage4] Loading pyannote diarization model...", flush=True)

    try:
        diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=DEVICE,
        )
    except Exception as e:
        print(f"  [stage4] Failed to load diarization pipeline: {e}", flush=True)
        return transcript

    print(f"  [stage4] Diarizing...", flush=True)

    try:
        diarize_segments = diarize_pipeline(wav_path)

        # Assign speakers to words
        result = whisperx.assign_word_speakers(
            diarize_segments,
            {"segments": transcript["segments"]},
        )

        transcript["segments"] = result["segments"]
        transcript["diarized"] = True

        # Count speakers
        speakers = set()
        for seg in transcript["segments"]:
            spk = seg.get("speaker", "")
            if spk:
                speakers.add(spk)

        transcript["speaker_count"] = len(speakers)
        print(f"  [stage4] Found {len(speakers)} speakers: {sorted(speakers)}", flush=True)

    except Exception as e:
        print(f"  [stage4] Diarization failed: {e}", flush=True)
        transcript["diarized"] = False

    del diarize_pipeline
    flush_gpu()

    print(f"  [stage4] Diarization model flushed. VRAM free: {vram_free_mb()}MB", flush=True)

    return transcript
