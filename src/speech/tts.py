from __future__ import annotations

import asyncio


class StreamingTTS:
    """Text to audio streamer.

    Returns byte chunks so playback can start before full synthesis is done.
    """

    async def stream(self, text: str):
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
            yield (" ".join(chunk)).encode("utf-8")
