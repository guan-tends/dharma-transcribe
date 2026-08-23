"""Command-line interface for the dharma transcription pipeline.

Entry point: dharma-transcribe (installed via pip) or python -m dharma_transcribe.cli
"""

import os
import sys

# Force CPU mode before any torch/whisperx imports if --device cpu is specified.
# This must happen before ANY import that touches CUDA, otherwise PyTorch
# initializes the GPU context even when models are later moved to CPU.
if "--device" in sys.argv:
    _idx = sys.argv.index("--device")
    if _idx + 1 < len(sys.argv) and sys.argv[_idx + 1] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from pathlib import Path

from . import config
from .config import DEVICE as DEFAULT_DEVICE
from .config import HF_TOKEN_ENV
from .gpu import flush_gpu, vram_free_mb, vram_total_mb
from .ingest import find_media_files
from .pipeline import load_manifest, process_file, save_manifest


def main():
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Dharma Audio Transcription Pipeline — "
        "multilingual transcription for Buddhist teachings",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Single file or directory to process",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Default source directory (overrides DHARMA_SOURCE_DIR env var)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM correction stage (ASR transcription only)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get(HF_TOKEN_ENV, ""),
        help="HuggingFace token (for pyannote diarization + model downloads)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help="Compute device: cuda (GPU) or cpu",
    )
    args = parser.parse_args()

    # Override device at runtime so all modules pick it up
    config.DEVICE = args.device

    # Print startup info
    print("Dharma Transcription Pipeline", flush=True)
    print(f"Device: {args.device}", flush=True)
    if args.device == "cpu":
        print("GPU: disabled (CPU mode)", flush=True)
    else:
        print(f"GPU: {vram_total_mb()}MB total, {vram_free_mb()}MB free", flush=True)
    print(f"LLM: {config.LLM_MODEL or 'disabled'}", flush=True)
    print(
        f"HF Token: {'provided' if args.hf_token else 'MISSING (diarization will be skipped)'}",
        flush=True,
    )

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
    elif args.source_dir:
        media_files = find_media_files(Path(args.source_dir))
    else:
        print(
            "Error: no input specified. Provide a file/directory path or --source-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not media_files:
        print("No media files found", flush=True)
        sys.exit(0)

    print(f"\nFound {len(media_files)} media file(s) to process", flush=True)

    # Load manifest for idempotency
    manifest = load_manifest()

    for media_path in media_files:
        # Use filename as manifest key (checksum-based key could be added)
        manifest_key = media_path.name

        # Skip already-processed files
        existing = manifest["files"].get(manifest_key, {})
        if existing.get("status") == "completed":
            print(f"\nSKIP (already processed): {media_path.name}", flush=True)
            continue

        result = process_file(media_path, args.hf_token, args.skip_llm)

        manifest["files"][manifest_key] = result
        save_manifest(manifest)

        # Flush GPU between files
        flush_gpu()

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("PIPELINE COMPLETE", flush=True)
    print(f"{'=' * 60}", flush=True)
    completed = sum(1 for f in manifest["files"].values() if f.get("status") == "completed")
    failed = sum(1 for f in manifest["files"].values() if f.get("status") == "failed")
    print(f"  Completed: {completed}", flush=True)
    print(f"  Failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
