# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI pipeline (lint, typecheck, test on Python 3.10/3.11/3.12)
- PyPI publishing workflow (trusted publishing via OIDC)
- Pre-commit hooks configuration (ruff + mypy)
- `.env.example` with all configurable environment variables
- `CONTRIBUTING.md` with development setup and testing guide
- `tests/fixtures/` directory with README for test audio samples

## [0.1.0] - 2026-08-23

### Added
- 7-stage pipeline architecture: Ingest → WhisperX → Alignment → Diarization → Tibetan Second-Pass → LLM Correction → Output
- WhisperX large-v3 primary transcription with automatic language detection
- Per-language forced alignment (wav2vec2 for en, ja, bo, sa)
- Speaker diarization via pyannote speaker-diarization-community-1
- Tibetan second-pass transcription using OpenPecha op-whisper_small-ft-v2 (dharma-trained)
- LLM post-correction via any OpenAI-compatible API (cloud or local)
- Output formats: JSON, SRT, VTT, TXT, review queue
- Corrections dictionary for deterministic pre-LLM fixes
- Idempotent manifest for batch processing
- VRAM management with model flushing between stages (6 GB consumer GPU support)
- `--device cpu` flag for CPU-only mode
- `--skip-llm` flag to skip LLM correction
- Environment-variable-based configuration (no hardcoded secrets)
- `src/` package layout with `pyproject.toml` (hatchling build backend)
- 54 unit tests covering all modules
- MIT license

### Security
- Git history scrubbed of all API keys, URLs, and private paths
- All sensitive values read from environment variables
- No credentials in source code or commit history
