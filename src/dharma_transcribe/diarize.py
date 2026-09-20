"""Stage 4: Speaker diarization via pyannote.

Uses WhisperX's DiarizationPipeline (pyannote-audio under the hood) to
identify speakers and assign word-level speaker labels.

The diarization pass runs in a SPAWNED SUBPROCESS guarded by a hard
timeout. pyannote's attention pooling can produce zero-length sequences
for certain audio segments, which triggers a std() warning and then hangs
the process indefinitely (no crash, no error). Running it in a killable
subprocess means a hang costs the timeout window instead of the whole
run -- see the 21-hour Black Manjushri hang, 2026-08-23/24.
"""

import multiprocessing as mp

from . import config
from .gpu import flush_gpu, vram_free_mb


def _diarize_worker(wav_path: str, hf_token: str, device: str, queue) -> None:
    """Subprocess entry point: load pyannote and diarize ``wav_path``.

    The resulting DataFrame is placed on ``queue`` as ``("ok", df)``. Any
    exception is reported as ``("error", repr)`` so the parent can degrade
    gracefully instead of crashing.

    Args:
        wav_path: Path to the extracted 16 kHz mono WAV.
        hf_token: HuggingFace token with pyannote gated-model access.
        device: Torch device string ("cuda" or "cpu").
        queue: Multiprocessing queue used to return the result.
    """
    try:
        from whisperx.diarize import DiarizationPipeline

        pipeline = DiarizationPipeline(token=hf_token, device=device)
        diarize_df = pipeline(wav_path)
        queue.put(("ok", diarize_df))
    except Exception as e:  # noqa: BLE001 - subprocess boundary reports all failures
        queue.put(("error", repr(e)))


def diarize_transcript(transcript: dict, wav_path: str, hf_token: str) -> dict:
    """Run speaker diarization and assign word-level speakers.

    The diarization pass runs in a spawned subprocess bounded by
    ``config.DIARIZE_TIMEOUT_SEC``. On timeout the subprocess is terminated
    and the transcript is returned un-diarized, so a pyannote hang costs
    the timeout window rather than the entire run.

    Args:
        transcript: Transcript dict with ``segments`` from earlier stages.
        wav_path: Path to the extracted WAV used for all audio stages.
        hf_token: HuggingFace token; empty string skips diarization.

    Returns:
        The transcript, with ``speaker`` labels added to words/segments when
        diarization succeeds, or ``diarized=False`` plus a ``diarize_note``
        when it is skipped, fails, or times out.
    """
    if not hf_token:
        print("  [stage4] No HF token — skipping diarization", flush=True)
        return transcript

    timeout = config.DIARIZE_TIMEOUT_SEC
    print(
        f"  [stage4] Diarizing in subprocess (hard timeout {timeout}s)...",
        flush=True,
    )

    # "spawn" (not fork): the parent already has CUDA initialised from
    # stages 2-3, and forking a CUDA context is unsafe. A fresh interpreter
    # also guarantees the child loads its own clean pyannote pipeline.
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_diarize_worker,
        args=(wav_path, hf_token, config.DEVICE, queue),
    )
    proc.start()
    proc.join(timeout)

    # --- Timeout path: kill the hung worker, keep the transcript ---
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        print(
            f"  [stage4] Diarization TIMED OUT after {timeout}s — killed. "
            "Transcript left un-diarized.",
            flush=True,
        )
        transcript["diarized"] = False
        transcript["diarize_note"] = f"timeout after {timeout}s"
        flush_gpu()
        return transcript

    # --- Crash-without-result path ---
    if queue.empty():
        print(
            "  [stage4] Diarization subprocess exited without a result "
            f"(exit code {proc.exitcode}) — transcript left un-diarized.",
            flush=True,
        )
        transcript["diarized"] = False
        transcript["diarize_note"] = f"subprocess exit {proc.exitcode}"
        flush_gpu()
        return transcript

    status, payload = queue.get()

    if status != "ok":
        print(f"  [stage4] Diarization failed: {payload}", flush=True)
        transcript["diarized"] = False
        transcript["diarize_note"] = str(payload)
        flush_gpu()
        return transcript

    # --- Success path: assign speaker labels in this process ---
    diarize_df = payload

    try:
        from whisperx.diarize import assign_word_speakers

        transcript = assign_word_speakers(diarize_df, transcript)

        speakers = {
            seg.get("speaker")
            for seg in transcript["segments"]
            if seg.get("speaker")
        }
        transcript["speaker_count"] = len(speakers)
        transcript["diarized"] = True

        print(
            f"  [stage4] Found {len(speakers)} speakers: {sorted(speakers)}",
            flush=True,
        )
    except Exception as e:
        print(f"  [stage4] Speaker assignment failed: {e}", flush=True)
        transcript["diarized"] = False
        transcript["diarize_note"] = str(e)

    flush_gpu()

    if config.DEVICE != "cpu":
        print(
            f"  [stage4] Diarization model flushed. VRAM free: {vram_free_mb()}MB",
            flush=True,
        )

    return transcript
