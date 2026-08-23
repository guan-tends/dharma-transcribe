"""Shared pytest fixtures for the dharma transcription pipeline tests."""

import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_transcript():
    """A minimal transcript dict for testing output and correction stages."""
    return {
        "source_file": "test_teaching.mp4",
        "wav_path": "/tmp/test.wav",
        "transcribed_at": "2026-01-01T00:00:00Z",
        "model": "whisperx-large-v3",
        "detected_language": "en",
        "segment_count": 3,
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Welcome to the teaching on bodhicitta.",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Welcome", "start": 0.0, "end": 0.5, "score": 0.95},
                    {"word": "to", "start": 0.5, "end": 0.7, "score": 0.98},
                    {"word": "the", "start": 0.7, "end": 0.9, "score": 0.97},
                    {"word": "teaching", "start": 0.9, "end": 1.3, "score": 0.92},
                    {"word": "on", "start": 1.3, "end": 1.5, "score": 0.96},
                    {"word": "bodhicitta.", "start": 1.5, "end": 2.0, "score": 0.3},
                ],
            },
            {
                "start": 5.0,
                "end": 10.0,
                "text": "The nature of shunyata is emptiness.",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "The", "start": 5.0, "end": 5.2, "score": 0.99},
                    {"word": "nature", "start": 5.2, "end": 5.6, "score": 0.94},
                    {"word": "of", "start": 5.6, "end": 5.8, "score": 0.97},
                    {"word": "shunyata", "start": 5.8, "end": 6.3, "score": 0.4},
                    {"word": "is", "start": 6.3, "end": 6.5, "score": 0.98},
                    {"word": "emptiness.", "start": 6.5, "end": 7.0, "score": 0.91},
                ],
            },
            {
                "start": 10.0,
                "end": 15.0,
                "text": "བོད་སྐད་ཀྱི་མཚོན་ཆ།",
                "speaker": "SPEAKER_01",
                "language": "bo",
                "words": [
                    {"word": "བོད་སྐད་", "start": 10.0, "end": 11.0, "score": 0.85},
                    {"word": "ཀྱི་", "start": 11.0, "end": 12.0, "score": 0.80},
                    {"word": "མཚོན་ཆ།", "start": 12.0, "end": 13.0, "score": 0.75},
                ],
            },
        ],
    }


@pytest.fixture
def sample_metadata():
    """Metadata dict as produced by the ingest stage."""
    return {
        "source_file": "/path/to/test.mp4",
        "source_name": "test.mp4",
        "checksum": "abc123def456",
        "wav_path": "/tmp/test_abc123de.wav",
        "duration": 15.0,
        "ingested_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for LLM correction tests."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "corrected": "Welcome to the teaching on bodhicitta.",
            "confidence": "high",
            "changes": ["bodichitta -> bodhicitta"],
        }
    )
    mock_choice.finish_reason = "stop"

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


@pytest.fixture
def temp_output_dir(tmp_path, monkeypatch):
    """Redirect output directories to a temp directory for isolated tests."""
    output = tmp_path / "output"
    output.mkdir()
    for subdir in ["json", "srt", "vtt", "txt", "review", "corrections", "wav"]:
        (output / subdir).mkdir()

    monkeypatch.setenv("DHARMA_OUTPUT_DIR", str(output))

    # Reimport config to pick up new env vars
    import importlib

    import config

    importlib.reload(config)
    yield output
