from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Hero Dealership Voice Agent")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm_timeout_sec: float = float(os.getenv("LLM_TIMEOUT_SEC", "3.5"))

    stt_timeout_sec: float = float(os.getenv("STT_TIMEOUT_SEC", "2.0"))
    tts_timeout_sec: float = float(os.getenv("TTS_TIMEOUT_SEC", "2.0"))

    eos_silence_ms: int = int(os.getenv("EOS_SILENCE_MS", "650"))
    eos_min_utterance_ms: int = int(os.getenv("EOS_MIN_UTTERANCE_MS", "400"))

    learning_store_dir: str = os.getenv("LEARNING_STORE_DIR", "data/learning")
    docs_dir: str = os.getenv("DOCS_DIR", "data/uploads")


settings = Settings()
