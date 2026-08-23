"""Global configuration for the dharma transcription pipeline.

All sensitive values (API keys, tokens) are read from environment variables.
No credentials are hardcoded. See .env.example for required variables.
"""

import os
from pathlib import Path

# --- Paths --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(
    os.environ.get(
        "DHARMA_OUTPUT_DIR",
        str(PROJECT_ROOT / "output"),
    )
)
LOG_DIR = Path(
    os.environ.get(
        "DHARMA_LOG_DIR",
        str(PROJECT_ROOT / "logs"),
    )
)
MODEL_CACHE = Path(
    os.environ.get(
        "DHARMA_MODEL_CACHE",
        str(PROJECT_ROOT / "models"),
    )
)

# Output subdirectories
JSON_DIR = OUTPUT_DIR / "json"
SRT_DIR = OUTPUT_DIR / "srt"
VTT_DIR = OUTPUT_DIR / "vtt"
TXT_DIR = OUTPUT_DIR / "txt"
REVIEW_DIR = OUTPUT_DIR / "review"
CORRECTIONS_DIR = OUTPUT_DIR / "corrections"
WAV_DIR = OUTPUT_DIR / "wav"

# --- WhisperX settings --------------------------------------------------------

WHISPER_MODEL = os.environ.get("DHARMA_WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("DHARMA_COMPUTE_TYPE", "int8")
BATCH_SIZE = int(os.environ.get("DHARMA_BATCH_SIZE", "4"))
DEVICE = os.environ.get("DHARMA_DEVICE", "cuda")
VAD_FILTER = True
CONDITION_ON_PREV_TEXT = False

# --- Tibetan second-pass model ------------------------------------------------

# OpenPecha model fine-tuned on Garchen Rinpoche's dharma teachings.
TIBETAN_MODEL_HF = os.environ.get(
    "DHARMA_TIBETAN_MODEL",
    "openpecha/op-whisper_small-ft-v2",
)

# --- Alignment models (wav2vec2) ----------------------------------------------

# None = use WhisperX/torchaudio built-in for that language.
ALIGN_MODELS: dict[str, str | None] = {
    "en": None,
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "bo": "openpecha/wav2vec2_run8",
    "sa": "addy88/wav2vec2-sanskrit-stt",
}

# --- LLM correction (OpenAI-compatible API) -----------------------------------

# Any OpenAI-compatible endpoint works: Synthetic, OpenAI, Ollama, vLLM,
# LM Studio, etc. Set DHARMA_LLM_API_URL and DHARMA_LLM_API_KEY to enable.
# Use --skip-llm to disable at runtime.
LLM_API_URL = os.environ.get("DHARMA_LLM_API_URL", "")
LLM_API_KEY = os.environ.get("DHARMA_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("DHARMA_LLM_MODEL", "")
LLM_CONFIDENCE_THRESHOLD = float(os.environ.get("DHARMA_LLM_CONFIDENCE_THRESHOLD", "0.5"))

# --- Review queue --------------------------------------------------------------

CONFIDENCE_THRESHOLD = float(os.environ.get("DHARMA_CONFIDENCE_THRESHOLD", "0.5"))

# --- Corrections dictionary ---------------------------------------------------

CORRECTIONS_FILE = CORRECTIONS_DIR / "corrections.json"

# --- Manifest -----------------------------------------------------------------

MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

# --- HuggingFace token --------------------------------------------------------

# Required for pyannote diarization and model downloads.
HF_TOKEN_ENV = "HF_TOKEN"

# --- ffmpeg output settings ----------------------------------------------------

FFMPEG_SR = 16000
FFMPEG_CHANNELS = 1
FFMPEG_CODEC = "pcm_s16le"

# --- Supported media extensions -----------------------------------------------

MEDIA_EXTENSIONS: set[str] = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".wma",
    ".wv",
    ".opus",
}

# --- Directory initialization --------------------------------------------------

for _d in [
    JSON_DIR,
    SRT_DIR,
    VTT_DIR,
    TXT_DIR,
    REVIEW_DIR,
    CORRECTIONS_DIR,
    LOG_DIR,
    MODEL_CACHE,
    WAV_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)
