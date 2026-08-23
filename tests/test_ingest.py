"""Unit tests for ingest.py — checksum, duration, file discovery."""

import hashlib
from pathlib import Path
from unittest.mock import patch

from dharma_transcribe import ingest


def test_compute_checksum(tmp_path):
    """Checksum should produce a stable SHA256 hash."""
    test_file = tmp_path / "test.wav"
    test_file.write_bytes(b"test content for hashing")

    result = ingest.compute_checksum(test_file)
    expected = hashlib.sha256(b"test content for hashing").hexdigest()
    assert result == expected
    assert len(result) == 64  # SHA256 hex length


def test_compute_checksum_streaming(tmp_path):
    """Checksum should handle large files without loading entirely into memory."""
    test_file = tmp_path / "large.wav"
    test_file.write_bytes(b"x" * (5 * 1024 * 1024))  # 5 MB

    result = ingest.compute_checksum(test_file)
    assert len(result) == 64


def test_get_duration_no_ffprobe():
    """get_duration should return 0.0 when ffprobe fails."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("ffprobe not found")
        result = ingest.get_duration(Path("/nonexistent/file.mp4"))
    assert result == 0.0


def test_find_media_files(tmp_path):
    """find_media_files should discover audio/video files recursively."""
    # Create test files
    (tmp_path / "teaching.mp4").touch()
    (tmp_path / "audio.mp3").touch()
    (tmp_path / "notes.txt").touch()  # should be ignored
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "recording.wav").touch()

    result = ingest.find_media_files(tmp_path)
    names = [f.name for f in result]

    assert "teaching.mp4" in names
    assert "audio.mp3" in names
    assert "recording.wav" in names
    assert "notes.txt" not in names
    assert len(result) == 3


def test_find_media_files_empty_dir(tmp_path):
    """find_media_files should return empty list for directory with no media."""
    (tmp_path / "readme.txt").touch()
    result = ingest.find_media_files(tmp_path)
    assert result == []


def test_ingest_file_nonexistent():
    """ingest_file should return None for non-existent files."""
    result = ingest.ingest_file(Path("/nonexistent/file.mp4"))
    assert result is None
