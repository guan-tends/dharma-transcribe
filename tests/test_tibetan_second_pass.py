"""Unit tests for tibetan_second_pass.py — heuristics, detection, selection logic."""

from dharma_transcribe.tibetan_second_pass import _has_tibetan_script, _looks_tibetan


def test_looks_tibetan_with_tibetan_text():
    """Should detect text with high proportion of Tibetan characters."""
    text = "བོད་སྐད་ཀྱི་མཚོན་ཆ།"
    assert _looks_tibetan(text) is True


def test_looks_tibetan_with_english_text():
    """Should not detect English text as Tibetan."""
    text = "Welcome to the teaching on bodhicitta."
    assert _looks_tibetan(text) is False


def test_looks_tibetan_mixed_text():
    """Should detect Tibetan even in mixed-language text (>10% Tibetan chars)."""
    text = "The teacher said བོད་སྐད་ཀྱི་ and then continued in English."
    assert _looks_tibetan(text) is True


def test_looks_tibetan_empty():
    """Empty text should return False."""
    assert _looks_tibetan("") is False
    assert _looks_tibetan(None) is False


def test_has_tibetan_script_true():
    """Should detect presence of any Tibetan Unicode character."""
    text = "This has one Tibetan syllable: བོད"
    assert _has_tibetan_script(text) is True


def test_has_tibetan_script_false():
    """Should return False for text with no Tibetan characters."""
    text = "Pure English text with no Tibetan."
    assert _has_tibetan_script(text) is False


def test_has_tibetan_script_empty():
    """Empty string should return False."""
    assert _has_tibetan_script("") is False
