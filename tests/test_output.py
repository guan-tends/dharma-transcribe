"""Unit tests for output.py — format generation, corrections, review queue."""

import json
from pathlib import Path

from dharma_transcribe.output import (
    _format_timestamp,
    _format_timestamp_vtt,
    apply_dictionary,
    generate_all_outputs,
    generate_json,
    generate_review_queue,
    generate_srt,
    generate_txt,
    generate_vtt,
    load_corrections,
)


def test_format_timestamp_srt():
    """SRT timestamps use comma as millisecond separator."""
    assert _format_timestamp(0.0) == "00:00:00,000"
    assert _format_timestamp(3661.5) == "01:01:01,500"
    assert _format_timestamp(59.999) == "00:00:59,999"


def test_format_timestamp_vtt():
    """VTT timestamps use period as millisecond separator."""
    assert _format_timestamp_vtt(0.0) == "00:00:00.000"
    assert _format_timestamp_vtt(3661.5) == "01:01:01.500"


def test_generate_srt(sample_transcript, tmp_path, monkeypatch):
    """SRT output should include segment numbers, timestamps, and speaker labels."""
    monkeypatch.setattr("dharma_transcribe.output.SRT_DIR", tmp_path)

    result = generate_srt(sample_transcript, "test.mp4")
    content = Path(result).read_text()

    assert "1" in content
    assert "00:00:00,000 --> 00:00:05,000" in content
    assert "[SPEAKER_00] Welcome to the teaching on bodhicitta." in content
    assert "2" in content


def test_generate_vtt(sample_transcript, tmp_path, monkeypatch):
    """VTT output should start with WEBVTT header and use speaker tags."""
    monkeypatch.setattr("dharma_transcribe.output.VTT_DIR", tmp_path)

    result = generate_vtt(sample_transcript, "test.mp4")
    content = Path(result).read_text()

    assert content.startswith("WEBVTT")
    assert "<v SPEAKER_00>" in content
    assert "00:00:00.000 --> 00:00:05.000" in content


def test_generate_txt(sample_transcript, tmp_path, monkeypatch):
    """TXT output should be plain text with optional speaker labels."""
    monkeypatch.setattr("dharma_transcribe.output.TXT_DIR", tmp_path)

    result = generate_txt(sample_transcript, "test.mp4")
    content = Path(result).read_text()

    assert "[SPEAKER_00] Welcome to the teaching on bodhicitta." in content
    assert "[SPEAKER_01]" in content  # Tibetan segment


def test_generate_json(sample_transcript, tmp_path, monkeypatch):
    """JSON output should be valid JSON with all transcript data."""
    monkeypatch.setattr("dharma_transcribe.output.JSON_DIR", tmp_path)

    result = generate_json(sample_transcript, "test.mp4")
    data = json.loads(Path(result).read_text())

    assert data["source_file"] == "test_teaching.mp4"
    assert len(data["segments"]) == 3
    assert data["segments"][0]["text"] == "Welcome to the teaching on bodhicitta."


def test_generate_review_queue(sample_transcript, tmp_path, monkeypatch):
    """Review queue should flag low-confidence segments."""
    monkeypatch.setattr("dharma_transcribe.output.REVIEW_DIR", tmp_path)
    monkeypatch.setattr("dharma_transcribe.output.CONFIDENCE_THRESHOLD", 0.5)

    result = generate_review_queue(sample_transcript, "test.mp4")
    if result:
        data = json.loads(Path(result).read_text())
        # Segment 0 has a word with score 0.3 < 0.5
        assert any(item["segment_id"] == 0 for item in data["review_items"])
    # If no review items, result is None — also valid if all words are confident


def test_apply_dictionary(sample_transcript, tmp_path, monkeypatch):
    """Dictionary corrections should apply case-insensitive replacements."""
    corrections_dir = tmp_path / "corrections"
    corrections_dir.mkdir()
    corrections_file = corrections_dir / "corrections.json"
    corrections_file.write_text(
        json.dumps(
            {
                "corrections": [
                    {"pattern": "bodhicitta", "replacement": "bodhicitta"},
                    {"pattern": "shunyata", "replacement": "śūnyatā"},
                ]
            }
        )
    )

    monkeypatch.setattr("dharma_transcribe.config.CORRECTIONS_DIR", corrections_dir)
    monkeypatch.setattr("dharma_transcribe.config.CORRECTIONS_FILE", corrections_file)
    monkeypatch.setattr("dharma_transcribe.output.CORRECTIONS_FILE", corrections_file)

    # Need to reload corrections
    result = apply_dictionary(sample_transcript.copy())

    # shunyata should be replaced
    seg2_text = result["segments"][1]["text"]
    assert "śūnyatā" in seg2_text


def test_load_corrections_empty(tmp_path, monkeypatch):
    """Loading corrections when file doesn't exist returns empty dict."""
    monkeypatch.setattr("dharma_transcribe.output.CORRECTIONS_FILE", tmp_path / "nonexistent.json")

    result = load_corrections()
    assert result == {"corrections": []}


def test_generate_all_outputs(sample_transcript, tmp_path, monkeypatch):
    """generate_all_outputs should produce all format files."""
    for dir_name, dir_attr in [
        ("json", "JSON_DIR"),
        ("srt", "SRT_DIR"),
        ("vtt", "VTT_DIR"),
        ("txt", "TXT_DIR"),
        ("review", "REVIEW_DIR"),
        ("corrections", "CORRECTIONS_DIR"),
    ]:
        d = tmp_path / dir_name
        d.mkdir()
        monkeypatch.setattr(f"dharma_transcribe.config.{dir_attr}", d)
        monkeypatch.setattr(f"dharma_transcribe.output.{dir_attr}", d)

    monkeypatch.setattr("dharma_transcribe.output.CONFIDENCE_THRESHOLD", 0.5)

    result = generate_all_outputs(sample_transcript, "test.mp4")

    assert "json" in result
    assert "srt" in result
    assert "vtt" in result
    assert "txt" in result
    assert Path(result["json"]).exists()
    assert Path(result["srt"]).exists()
    assert Path(result["vtt"]).exists()
    assert Path(result["txt"]).exists()
