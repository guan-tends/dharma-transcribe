"""Main pipeline orchestrator — chains all 7 stages."""
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SOURCE_DIR, OUTPUT_DIR, MANIFEST_FILE, JSON_DIR,
    HF_TOKEN_ENV, LLM_MODEL, SYNTHETIC_API_URL
)
from gpu import flush_gpu, vram_free_mb, vram_total_mb
from ingest import ingest_file, find_media_files
from transcribe import transcribe_with_metadata
from align import align_transcript
from diarize import diarize_transcript
from tibetan_second_pass import tibetan_second_pass
from llm_correct import llm_correct_transcript
from output import generate_all_outputs


def load_manifest() -> dict:
    """Load or create the processing manifest."""
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text())
    return {"files": {}, "last_updated": None}


def save_manifest(manifest: dict):
    """Save the processing manifest."""
    manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def process_file(media_path: Path, hf_token: str = "", skip_llm: bool = False) -> dict:
    """
    Process a single media file through all 7 pipeline stages.
    Returns the manifest entry for this file.
    """
    source_name = media_path.name
    print(f"\n{'='*60}", flush=True)
    print(f"PROCESSING: {source_name}", flush=True)
    print(f"{'='*60}", flush=True)

    start_time = time.time()
    stages_completed = []

    # === STAGE 1: INGEST ===
    print(f"\n[STAGE 1: INGEST]", flush=True)
    metadata = ingest_file(media_path)
    if metadata is None:
        print(f"  FAILED: ingest returned None", flush=True)
        return {"status": "failed", "stage": "ingest", "error": "ingest failed"}

    print(f"  Duration: {metadata['duration']:.1f}s", flush=True)
    print(f"  WAV: {Path(metadata['wav_path']).name}", flush=True)
    stages_completed.append("ingest")

    # === STAGE 2: TRANSCRIPTION ===
    print(f"\n[STAGE 2: TRANSCRIPTION]", flush=True)
    transcript = transcribe_with_metadata(metadata["wav_path"], source_name)
    if transcript is None:
        return {"status": "failed", "stage": "transcribe", "error": "transcription returned None"}
    stages_completed.append("transcribe")

    # === STAGE 3: ALIGNMENT ===
    print(f"\n[STAGE 3: ALIGNMENT]", flush=True)
    transcript = align_transcript(transcript, metadata["wav_path"], hf_token)
    stages_completed.append("align")

    # === STAGE 4: DIARIZATION ===
    print(f"\n[STAGE 4: DIARIZATION]", flush=True)
    transcript = diarize_transcript(transcript, metadata["wav_path"], hf_token)
    stages_completed.append("diarize")

    # === STAGE 5: TIBETAN SECOND PASS ===
    print(f"\n[STAGE 5: TIBETAN SECOND PASS]", flush=True)
    transcript = tibetan_second_pass(transcript, metadata["wav_path"])
    stages_completed.append("tibetan_second_pass")

    # === STAGE 6: LLM CORRECTION ===
    if not skip_llm:
        print(f"\n[STAGE 6: LLM CORRECTION]", flush=True)
        transcript = llm_correct_transcript(transcript)
        stages_completed.append("llm_correct")
    else:
        print(f"\n[STAGE 6: LLM CORRECTION] SKIPPED", flush=True)

    # === STAGE 7: OUTPUT ===
    print(f"\n[STAGE 7: OUTPUT]", flush=True)
    outputs = generate_all_outputs(transcript, source_name)
    stages_completed.append("output")
    print(f"  JSON: {outputs.get('json', 'N/A')}", flush=True)
    print(f"  SRT:  {outputs.get('srt', 'N/A')}", flush=True)
    print(f"  VTT:  {outputs.get('vtt', 'N/A')}", flush=True)
    print(f"  TXT:  {outputs.get('txt', 'N/A')}", flush=True)
    if "review" in outputs:
        print(f"  REVIEW: {outputs['review']}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n  TOTAL TIME: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    return {
        "status": "completed",
        "stages": stages_completed,
        "source_file": source_name,
        "checksum": metadata["checksum"],
        "duration": metadata["duration"],
        "processing_time": round(elapsed, 1),
        "outputs": outputs,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dharma Audio Transcription Pipeline")
    parser.add_argument("input", nargs="?", help="Single file or directory to process")
    parser.add_argument("--source-dir", default=str(SOURCE_DIR), help="Source directory")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM correction stage")
    parser.add_argument("--hf-token", default=os.environ.get(HF_TOKEN_ENV, ""), help="HuggingFace token")
    args = parser.parse_args()

    print(f"Dharma Transcription Pipeline", flush=True)
    print(f"GPU: {vram_total_mb()}MB total, {vram_free_mb()}MB free", flush=True)
    print(f"LLM: {LLM_MODEL} via Synthetic API", flush=True)
    print(f"HF Token: {'provided' if args.hf_token else 'MISSING (diarization will be skipped)'}", flush=True)

    # Determine input files
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            media_files = [input_path]
        elif input_path.is_dir():
            media_files = find_media_files(input_path)
        else:
            print(f"Error: {args.input} not found", file=sys.stderr)
            sys.exit(1)
    else:
        media_files = find_media_files(Path(args.source_dir))

    if not media_files:
        print(f"No media files found", flush=True)
        sys.exit(0)

    print(f"\nFound {len(media_files)} media file(s) to process", flush=True)

    # Load manifest for idempotency
    manifest = load_manifest()

    for media_path in media_files:
        checksum_key = media_path.name  # simplified; could use checksum

        # Check if already processed
        existing = manifest["files"].get(checksum_key, {})
        if existing.get("status") == "completed":
            print(f"\nSKIP (already processed): {media_path.name}", flush=True)
            continue

        # Process the file
        result = process_file(media_path, args.hf_token, args.skip_llm)

        # Update manifest
        manifest["files"][checksum_key] = result
        save_manifest(manifest)

        # Flush GPU between files
        flush_gpu()

    print(f"\n{'='*60}", flush=True)
    print(f"PIPELINE COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    completed = sum(1 for f in manifest["files"].values() if f.get("status") == "completed")
    failed = sum(1 for f in manifest["files"].values() if f.get("status") == "failed")
    print(f"  Completed: {completed}", flush=True)
    print(f"  Failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
