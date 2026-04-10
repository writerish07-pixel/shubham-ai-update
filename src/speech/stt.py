from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

import httpx

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text-translate"


@dataclass
class TranscriptChunk:
    text: str
    is_final: bool


class StreamingSTT:
    """Adapter layer for streaming STT.

    When SARVAM_API_KEY is configured, audio bytes are sent to the Sarvam
    speech-to-text API.  When running locally without an API key the
    text-frame passthrough mode is used (tokens arrive as text strings).
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._audio_buffer: bytearray = bytearray()
        self._enabled = bool(settings.sarvam_api_key)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.stt_timeout_sec, connect=2.0),
            )
        return self._client

    # ---- text-frame interface (used by QA / WebSocket text mode) ----

    async def feed_text_frame(self, frame: str, is_final: bool = False) -> TranscriptChunk:
        await asyncio.sleep(0)
        if frame.strip():
            self._buffer.append(frame.strip())
        text = " ".join(self._buffer)
        if is_final:
            self._buffer.clear()
        logger.debug("STT chunk final=%s text=%s", is_final, text)
        return TranscriptChunk(text=text, is_final=is_final)

    # ---- audio-bytes interface (used by real telephony) ----

    async def feed_audio(self, audio_bytes: bytes, is_final: bool = False) -> TranscriptChunk:
        """Accept raw audio bytes. When *is_final* is True, send to STT API."""
        self._audio_buffer.extend(audio_bytes)
        if not is_final:
            return TranscriptChunk(text="", is_final=False)

        if not self._enabled:
            logger.debug("STT disabled — returning empty transcript for %d bytes", len(self._audio_buffer))
            self._audio_buffer.clear()
            return TranscriptChunk(text="", is_final=True)

        transcript = await self._transcribe(bytes(self._audio_buffer))
        self._audio_buffer.clear()
        return TranscriptChunk(text=transcript, is_final=True)

    async def _transcribe(self, audio: bytes) -> str:
        try:
            client = await self._get_client()
            audio_b64 = base64.b64encode(audio).decode("ascii")
            resp = await client.post(
                SARVAM_STT_URL,
                headers={
                    "api-subscription-key": settings.sarvam_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "input": audio_b64,
                    "model": "saarika:v2",
                    "language_code": "hi-IN",
                    "with_timestamps": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("transcript", "")
        except httpx.TimeoutException:
            logger.warning("Sarvam STT timeout")
            return ""
        except Exception as exc:
            logger.warning("Sarvam STT error: %s", exc)
            return ""

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
