from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptChunk:
    text: str
    is_final: bool


class StreamingSTT:
    """Adapter layer for streaming STT.

    The default implementation is intentionally provider-agnostic and can be
    swapped with Deepgram/Sarvam/Whisper live streaming APIs.
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []

    async def feed_text_frame(self, frame: str, is_final: bool = False) -> TranscriptChunk:
        # In production, frame is bytes and this method sends audio to STT provider.
        await asyncio.sleep(0)
        if frame.strip():
            self._buffer.append(frame.strip())
        text = " ".join(self._buffer)
        if is_final:
            self._buffer.clear()
        logger.debug("STT chunk final=%s text=%s", is_final, text)
        return TranscriptChunk(text=text, is_final=is_final)
