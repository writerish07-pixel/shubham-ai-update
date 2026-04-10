from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import httpx

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are Priya, a friendly and professional Hero MotoCorp dealership sales advisor. "
    "You speak concise Hinglish (mix of Hindi and English). Always use complete sentences. "
    "Be polite, consultative, and helpful. Keep replies to 2 sentences max (~25 words). "
    "Use female Hindi grammar (e.g., 'main karti hoon', 'main batati hoon'). "
    "Always end with a question or call-to-action to keep the conversation going."
)


class GroqClient:
    def __init__(self) -> None:
        self._enabled = bool(settings.groq_api_key)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.llm_timeout_sec, connect=2.0),
            )
        return self._client

    async def stream_reply(
        self,
        prompt: str,
        context: str = "",
        fast: bool = False,
    ) -> AsyncGenerator[str, None]:
        if not self._enabled:
            fallback = (
                "Sir, is topic par main aapko best offer ke saath madad karungi. "
                "Kya aap model name confirm karenge?"
            )
            for token in fallback.split():
                await asyncio.sleep(0)
                yield token + " "
            return

        model = settings.groq_model_fast if fast else settings.groq_model

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if context:
            messages.append(
                {"role": "system", "content": f"Relevant dealership memory:\n{context}"}
            )
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "temperature": 0.4,
            "max_tokens": 100,
            "stream": True,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        try:
            client = await self._get_client()
            async with client.stream(
                "POST", GROQ_API_URL, json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk_str = line[len("data: "):]
                    if chunk_str.strip() == "[DONE]":
                        break
                    try:
                        import json

                        chunk = json.loads(chunk_str)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield token
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
        except httpx.TimeoutException:
            logger.warning("Groq request timed out (model=%s)", model)
            yield "Maaf kijiye, thoda time lag raha hai. Kya aap apna sawaal dubara bolenge?"
        except httpx.HTTPStatusError as exc:
            logger.warning("Groq HTTP error %s: %s", exc.response.status_code, exc)
            yield "Maaf kijiye, network issue hua hai. Kya aap model aur budget dubara batayenge?"
        except Exception as exc:
            logger.warning("Groq reply failed: %s", exc)
            yield "Maaf kijiye, network issue hua hai. Kya aap model aur budget dubara batayenge?"

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
