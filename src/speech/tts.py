from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict

import httpx

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StreamingTTS:
    """Text-to-speech streamer backed by Sarvam AI API.

    When SARVAM_API_KEY is set, real audio bytes are returned.
    Otherwise a lightweight UTF-8 text fallback is used so the rest
    of the pipeline keeps working during local development.

    Latency optimisations
    ---------------------
    * Short text (≤ 25 words) is synthesised in a **single API call**
      so there is no segment-splitting overhead.
    * Longer text is split into sentence-sized segments.  All segments
      are fired **concurrently**; the stream yields each segment's
      audio **as soon as it is ready** (in order) so playback of the
      first segment can start while later segments are still being
      synthesised.
    * An LRU cache (256 entries) avoids repeat calls for frequently
      used phrases (greeting, scripted responses).
    * ``warm_cache`` pre-synthesises high-frequency phrases at startup.
    """

    _MAX_CACHE_SIZE = 256

    def __init__(self) -> None:
        self._enabled = bool(settings.sarvam_api_key)
        self._client: httpx.AsyncClient | None = None
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.tts_timeout_sec, connect=2.0),
            )
        return self._client

    async def stream(self, text: str):
        """Yield audio byte chunks for *text*.

        Short replies (≤ 25 words) are synthesised in one shot.
        Longer replies are split into sentence-sized segments,
        all fired concurrently, and yielded in order as each
        segment completes — so playback starts as soon as the
        first segment is ready.
        """
        if not text or not text.strip():
            return

        if not self._enabled:
            async for chunk in self._fallback_stream(text):
                yield chunk
            return

        words = text.split()
        if len(words) <= 25:
            # Short text -> single API call (fastest path)
            audio = await self._synthesise(text.strip())
            if audio:
                yield audio
        else:
            # Longer text -> fire all segments concurrently,
            # yield each in order as it completes (pipelined).
            segments = self._split_segments(text)
            tasks = [asyncio.ensure_future(self._synthesise(seg)) for seg in segments]
            for task in tasks:
                audio = await task
                if audio:
                    yield audio

    async def _synthesise(self, text: str, _retries: int = 1) -> bytes | None:
        cache_key = text.strip().lower()
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        for attempt in range(_retries + 1):
            try:
                client = await self._get_client()
                resp = await client.post(
                    settings.sarvam_tts_url,
                    headers={
                        "api-subscription-key": settings.sarvam_api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": [text],
                        "target_language_code": settings.sarvam_tts_language,
                        "speaker": settings.sarvam_tts_speaker,
                        "model": "bulbul:v2",
                        "enable_preprocessing": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                audios = data.get("audios")
                if audios and audios[0]:
                    audio_bytes = base64.b64decode(audios[0])
                    self._cache[cache_key] = audio_bytes
                    if len(self._cache) > self._MAX_CACHE_SIZE:
                        self._cache.popitem(last=False)
                    return audio_bytes
                logger.warning("Sarvam TTS returned empty audio for: %s", text[:40])
                return None
            except httpx.TimeoutException:
                logger.warning("Sarvam TTS timeout for: %s", text[:40])
                return None
            except Exception as exc:
                if attempt < _retries:
                    logger.debug("Sarvam TTS transient error, retrying: %s", exc)
                    await asyncio.sleep(0.3)
                    continue
                logger.warning("Sarvam TTS error: %s", exc)
                return None
        return None

    @staticmethod
    def _split_segments(text: str, max_words: int = 15) -> list[str]:
        """Split into sentence fragments for concurrent TTS."""
        segments: list[str] = []
        buf: list[str] = []
        for word in text.split():
            buf.append(word)
            if word.endswith((".", "?", "!")) or len(buf) >= max_words:
                segments.append(" ".join(buf))
                buf.clear()
        if buf:
            segments.append(" ".join(buf))
        return segments

    @staticmethod
    async def _fallback_stream(text: str):
        """UTF-8 text chunks when no TTS API is configured."""
        words = text.split()
        chunk: list[str] = []
        for word in words:
            chunk.append(word)
            if len(chunk) >= 5:
                await asyncio.sleep(0)
                yield (" ".join(chunk) + " ").encode("utf-8")
                chunk.clear()
        if chunk:
            await asyncio.sleep(0)
            yield " ".join(chunk).encode("utf-8")

    async def warm_cache(self, texts: list[str]) -> None:
        """Pre-synthesise commonly used phrases into the cache.

        Called once at startup so the first caller gets instant audio
        for the greeting and other high-frequency phrases.
        """
        if not self._enabled:
            return
        results = await asyncio.gather(
            *(self._synthesise(t) for t in texts),
            return_exceptions=True,
        )
        cached = sum(1 for r in results if isinstance(r, bytes))
        logger.info("TTS warm cache: %d/%d phrases pre-cached", cached, len(texts))

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
