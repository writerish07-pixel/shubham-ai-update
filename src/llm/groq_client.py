from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from groq import AsyncGroq

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GroqClient:
    def __init__(self) -> None:
        self._enabled = bool(settings.groq_api_key)
        self._client = AsyncGroq(api_key=settings.groq_api_key) if self._enabled else None

    async def stream_reply(self, prompt: str, context: str = "") -> AsyncGenerator[str, None]:
        if not self._enabled:
            fallback = (
                "Sir, is topic par main aapko best offer ke saath madad karunga. "
                "Kya aap model name confirm karenge?"
            )
            for token in fallback.split():
                await asyncio.sleep(0)
                yield token + " "
            return

        try:
            stream = await self._client.chat.completions.create(
                model=settings.groq_model,
                temperature=0.4,
                max_tokens=180,
                stream=True,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a top Hero dealership sales advisor. "
                            "Speak concise Hinglish, complete sentences, polite and consultative."
                        ),
                    },
                    {"role": "system", "content": f"Relevant dealership memory: {context}"},
                    {"role": "user", "content": prompt},
                ],
            )
            async for part in stream:
                delta = part.choices[0].delta.content or ""
                if delta:
                    yield delta
        except Exception as exc:
            logger.exception("Groq streaming failed: %s", exc)
            yield "Maaf kijiye, network issue hua hai. Kya aap model aur budget dubara batayenge?"
