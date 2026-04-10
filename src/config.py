from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env file if present (before reading env vars)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_path)


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Hero Dealership Voice Agent")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # --- LLM (Groq) ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_model_fast: str = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
    llm_timeout_sec: float = float(os.getenv("LLM_TIMEOUT_SEC", "3.5"))

    # --- Telephony (Exotel) ---
    exotel_api_key: str = os.getenv("EXOTEL_API_KEY", "")
    exotel_api_token: str = os.getenv("EXOTEL_API_TOKEN", "")
    exotel_sid: str = os.getenv("EXOTEL_SID", "")
    exotel_exophone: str = os.getenv("EXOTEL_EXOPHONE", "")

    # --- Speech (Sarvam) ---
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    sarvam_tts_url: str = os.getenv(
        "SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech"
    )
    sarvam_tts_speaker: str = os.getenv("SARVAM_TTS_SPEAKER", "anushka")
    sarvam_tts_language: str = os.getenv("SARVAM_TTS_LANGUAGE", "hi-IN")

    # --- STT ---
    stt_timeout_sec: float = float(os.getenv("STT_TIMEOUT_SEC", "2.0"))
    tts_timeout_sec: float = float(os.getenv("TTS_TIMEOUT_SEC", "2.0"))

    # --- End-of-speech ---
    eos_silence_ms: int = int(os.getenv("EOS_SILENCE_MS", "650"))
    eos_min_utterance_ms: int = int(os.getenv("EOS_MIN_UTTERANCE_MS", "400"))

    # --- Learning / RAG ---
    learning_store_dir: str = os.getenv("LEARNING_STORE_DIR", "data/learning")
    docs_dir: str = os.getenv("DOCS_DIR", "data/uploads")

    # --- Deployment ---
    base_url: str = os.getenv("BASE_URL", "")  # public callback URL for Exotel


settings = Settings()
