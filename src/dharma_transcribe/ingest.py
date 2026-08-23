"""Stage 1: Ingest — extract audio from media files, compute metadata."""

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

from .config import FFMPEG_CHANNELS, FFMPEG_CODEC, FFMPEG_SR, MEDIA_EXTENSIONS, OUTPUT_DIR


def compute_checksum(filepath: Path) -> str:
    """SHA256 checksum of a file (streaming, memory-safe)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_duration(filepath: Path) -> float:
    """Get media duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_audio(input_path: Path, output_path: Path) -> bool:
    """Extract audio to 16kHz mono WAV via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(FFMPEG_SR),
        "-ac",
        str(FFMPEG_CHANNELS),
        "-c:a",
        FFMPEG_CODEC,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    return result.returncode == 0


def ingest_file(media_path: Path) -> dict | None:
    """Process a single media file: checksum, extract audio, metadata."""
    if not media_path.exists():
        return None

    checksum = compute_checksum(media_path)
    wav_path = OUTPUT_DIR / "wav" / f"{media_path.stem}_{checksum[:8]}.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if wav_path.exists():
        # Already extracted
        duration = get_duration(wav_path)
    else:
        if not extract_audio(media_path, wav_path):
            return None
        duration = get_duration(wav_path)

    return {
        "source_file": str(media_path),
        "source_name": media_path.name,
        "checksum": checksum,
        "wav_path": str(wav_path),
        "duration": round(duration, 2),
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }


def find_media_files(source_dir: Path) -> list[Path]:
    """Walk directory recursively, find all audio/video files."""
    files = []
    for f in source_dir.rglob("*"):
        if f.suffix.lower() in MEDIA_EXTENSIONS and f.is_file():
            files.append(f)
    return sorted(files)
