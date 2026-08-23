"""Unit tests for gpu.py — VRAM utilities and flushing."""

from unittest.mock import MagicMock, patch

from dharma_transcribe import gpu


def test_vram_free_mb_no_gpu():
    """vram_free_mb should return 0 when nvidia-smi fails."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        result = gpu.vram_free_mb()
    assert result == 0


def test_vram_total_mb_no_gpu():
    """vram_total_mb should return 0 when nvidia-smi fails."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        result = gpu.vram_total_mb()
    assert result == 0


def test_vram_free_mb_with_gpu():
    """vram_free_mb should parse nvidia-smi output correctly."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "5500\n"
    with patch("subprocess.run", return_value=mock_result):
        result = gpu.vram_free_mb()
    assert result == 5500


def test_vram_total_mb_with_gpu():
    """vram_total_mb should parse nvidia-smi output correctly."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "6144\n"
    with patch("subprocess.run", return_value=mock_result):
        result = gpu.vram_total_mb()
    assert result == 6144


def test_flush_gpu_no_torch():
    """flush_gpu should not crash when torch is not available."""
    # flush_gpu catches ImportError internally
    gpu.flush_gpu()  # should not raise
