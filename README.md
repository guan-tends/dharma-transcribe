# Dharma Audio Transcription Pipeline

Headless Python pipeline for transcribing 1000+ hours of Buddhist dharma teachings.
Mixed-language: English, Tibetan (bo), Sanskrit (sa), Japanese (ja).

## Architecture — 7 Stages (Serial, GPU flushing between each)

1. **Ingest** — ffmpeg extract to 16kHz mono WAV, checksum idempotency
2. **Transcription** — WhisperX large-v3, int8, auto language detect
3. **Alignment** — wav2vec2 per-language (en/ja built-in, bo/sa custom)
4. **Diarization** — pyannote speaker identification
5. **Tibetan Second-Pass** — OpenPecha whisper-small (dharma-trained) on bo segments
6. **LLM Correction** — Qwen3.5-9B (local Ollama, 4-bit) with dharma domain prompt
7. **Output** — JSON/SRT/VTT/TXT + review queue + corrections dictionary

## Setup

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install torch (CUDA 12.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install WhisperX + deps
pip install whisperx requests transformers

# Ollama + Qwen3.5-9B for LLM correction
ollama pull qwen3.5:9b

# Set HuggingFace token (for pyannote + model downloads)
export HF_TOKEN=your_token_here
```

## Usage

```bash
# Single file
python src/pipeline.py /path/to/audio.mp4 --hf-token $HF_TOKEN

# Directory (batch)
python src/pipeline.py /path/to/recordings/ --hf-token $HF_TOKEN

# Skip LLM correction (testing ASR only)
python src/pipeline.py /path/to/audio.mp4 --skip-llm
```

## VRAM Management

the GPU 6GB — serial stage execution with model flushing between stages:
- Stage 2: large-v3 int8 (~4GB) → FLUSH
- Stage 3: alignment model (~1-2GB) → FLUSH
- Stage 4: pyannote (~1GB) → FLUSH
- Stage 5: OpenPecha whisper-small (~1GB) → FLUSH
- Stage 6: Qwen3.5-9B 4-bit (~5-6.6GB) → FLUSH

## Corrections Dictionary

`output/corrections/corrections.json` — case-insensitive string replacement.
Applied before LLM pass. Grows from manual review.

## Samaya

AI Horde rejected. Sacred dharma teachings stay local. Qwen3.5-9B runs locally only.
