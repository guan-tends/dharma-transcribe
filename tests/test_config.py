"""Unit tests for config.py — environment variable loading and defaults."""

import importlib
import os


def test_config_loads_defaults(monkeypatch):
    """Config should load with sensible defaults when no env vars are set."""
    # Clear all DHARMA_ env vars
    for key in list(os.environ):
        if key.startswith("DHARMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    from dharma_transcribe import config

    importlib.reload(config)

    assert config.WHISPER_MODEL == "large-v3"
    assert config.COMPUTE_TYPE == "int8"
    assert config.BATCH_SIZE == 4
    assert config.DEVICE == "cuda"
    assert config.FFMPEG_SR == 16000
    assert config.FFMPEG_CHANNELS == 1
    assert config.FFMPEG_CODEC == "pcm_s16le"
    assert config.LLM_API_URL == ""
    assert config.LLM_API_KEY == ""
    assert config.LLM_MODEL == ""
    assert config.HF_TOKEN_ENV == "HF_TOKEN"


def test_config_reads_env_vars(monkeypatch):
    """Config should read all values from environment variables."""
    monkeypatch.setenv("DHARMA_WHISPER_MODEL", "medium")
    monkeypatch.setenv("DHARMA_COMPUTE_TYPE", "float16")
    monkeypatch.setenv("DHARMA_BATCH_SIZE", "8")
    monkeypatch.setenv("DHARMA_DEVICE", "cpu")
    monkeypatch.setenv("DHARMA_LLM_API_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("DHARMA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("DHARMA_LLM_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("DHARMA_LLM_CONFIDENCE_THRESHOLD", "0.8")
    monkeypatch.setenv("DHARMA_CONFIDENCE_THRESHOLD", "0.7")

    from dharma_transcribe import config

    importlib.reload(config)

    assert config.WHISPER_MODEL == "medium"
    assert config.COMPUTE_TYPE == "float16"
    assert config.BATCH_SIZE == 8
    assert config.DEVICE == "cpu"
    assert config.LLM_API_URL == "http://localhost:11434/v1"
    assert config.LLM_API_KEY == "test-key"
    assert config.LLM_MODEL == "qwen3.5:9b"
    assert config.LLM_CONFIDENCE_THRESHOLD == 0.8
    assert config.CONFIDENCE_THRESHOLD == 0.7


def test_config_media_extensions():
    """Media extensions set should include common audio/video formats."""
    from dharma_transcribe import config

    expected = {".mp4", ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
    assert expected.issubset(config.MEDIA_EXTENSIONS)


def test_config_align_models():
    """Alignment models dict should map known languages."""
    from dharma_transcribe import config

    assert "en" in config.ALIGN_MODELS
    assert "ja" in config.ALIGN_MODELS
    assert "bo" in config.ALIGN_MODELS
    assert "sa" in config.ALIGN_MODELS
    assert config.ALIGN_MODELS["en"] is None  # built-in


def test_config_tibetan_model_default(monkeypatch):
    """Tibetan model should default to OpenPecha dharma-trained model."""
    monkeypatch.delenv("DHARMA_TIBETAN_MODEL", raising=False)
    from dharma_transcribe import config

    importlib.reload(config)

    assert config.TIBETAN_MODEL_HF == "openpecha/op-whisper_small-ft-v2"


def test_config_no_hardcoded_secrets(monkeypatch):
    """Config should never contain hardcoded API keys or URLs."""
    monkeypatch.delenv("DHARMA_LLM_API_URL", raising=False)
    monkeypatch.delenv("DHARMA_LLM_API_KEY", raising=False)

    from dharma_transcribe import config

    importlib.reload(config)

    assert not config.LLM_API_URL
    assert not config.LLM_API_KEY
    # Ensure no example.com or syn_ patterns in the module
    import inspect

    source = inspect.getsource(config)
    assert "syn_" not in source
    assert "example.com" not in source
