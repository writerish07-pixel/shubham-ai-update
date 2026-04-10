from __future__ import annotations

import asyncio
import re

from src.agent.models import ConversationState
from src.hybrid.router import HybridRouter
from src.learning.learner import CallLearner
from src.learning.rag import RAGRetriever
from src.llm.groq_client import GroqClient
from src.speech.tts import StreamingTTS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationEngine:
    def __init__(self, learner: CallLearner, rag: RAGRetriever) -> None:
        self.router = HybridRouter()
        self.llm = GroqClient()
        self.tts = StreamingTTS()
        self.learner = learner
        self.rag = rag

    async def generate_response(self, state: ConversationState, user_text: str):
        cleaned_input = (user_text or "").strip()
        if not cleaned_input:
            reply = "Ji, aap aaraam se batayein, aapko kis type ki bike chahiye?"
            state.add_agent(reply)
            return reply, None, self.tts.stream(reply)

        state.add_user(cleaned_input)

        try:
            decision = self.router.route(cleaned_input)
        except Exception as exc:
            logger.warning("Router error: %s", exc)
            reply = self._enforce_sales_style("")
            state.add_agent(reply)
            return reply, None, self.tts.stream(reply)

        if decision.source == "script":
            reply = self._enforce_sales_style(decision.text)
            state.add_agent(reply)
            asyncio.create_task(
                self._safe_learn(state.call_id, cleaned_input, reply, decision.intent)
            )
            return reply, decision.intent, self.tts.stream(reply)

        context = self.rag.context_for(cleaned_input)

        # --- Streaming LLM-to-TTS: start first-sentence TTS while LLM
        # is still generating the rest so latencies overlap. ---
        first_sentence: str | None = None
        first_tts_task: asyncio.Task | None = None
        buffer = ""
        _SENT_RE = re.compile(r'(?<=[!?])\s+|(?<!\d)(?<=[.])\s+')

        chunks: list[str] = []
        try:
            async for token in self.llm.stream_reply(cleaned_input, context=context):
                chunks.append(token)
                # Detect first complete sentence boundary while streaming
                if first_sentence is None:
                    buffer += token
                    parts = _SENT_RE.split(buffer, maxsplit=1)
                    if len(parts) > 1 and parts[0].strip():
                        first_sentence = parts[0].strip()
                        # Fire TTS for first sentence immediately (overlaps
                        # with remaining LLM generation).
                        first_tts_task = asyncio.ensure_future(
                            self.tts._synthesise(first_sentence)
                        )
        except Exception as exc:
            logger.warning("LLM stream error: %s", exc)

        reply = self._enforce_sales_style("".join(chunks).strip())

        # 70/30 user-agent talk ratio: trim to first two sentences if agent is talking too much
        _user_ratio, agent_ratio = state.talk_ratio
        if agent_ratio > 0.35:
            # Split on sentence-ending punctuation + space; negative lookbehind
            # for digits preserves decimals like "1.5 lakh" or "9.9%".
            sentences = [s.strip() for s in re.split(r'(?<=[!?])\s+|(?<!\d)(?<=[.])\s+', reply) if s.strip()]
            if len(sentences) > 2:
                reply = " ".join(sentences[:2])
                if not reply.endswith((".", "?", "!")):
                    reply += "."

        state.add_agent(reply)
        asyncio.create_task(
            self._safe_learn(state.call_id, cleaned_input, reply, decision.intent)
        )

        # Build an optimised TTS stream that re-uses the pre-fetched first
        # sentence audio (if available) so playback starts sooner.
        audio_stream = self._tts_with_prefetch(reply, first_sentence, first_tts_task)
        return reply, decision.intent, audio_stream

    async def _tts_with_prefetch(
        self,
        reply: str,
        prefetched_sentence: str | None,
        prefetch_task: asyncio.Task | None,
    ):
        """Yield TTS audio, re-using a pre-fetched first-sentence result.

        If *prefetched_sentence* matches the start of *reply* and the
        pre-fetch task succeeded, its audio is yielded immediately
        (saving one API round-trip).  The remainder of *reply* is then
        synthesised via the normal ``tts.stream`` path.
        """
        used_prefetch = False
        if (
            prefetched_sentence
            and prefetch_task is not None
            and reply.startswith(prefetched_sentence)
        ):
            try:
                audio = await prefetch_task
                if audio:
                    yield audio
                    used_prefetch = True
                    # Synthesise the rest (after the first sentence)
                    remainder = reply[len(prefetched_sentence):].strip()
                    if remainder:
                        async for chunk in self.tts.stream(remainder):
                            yield chunk
                    return
            except Exception:
                pass  # fall through to normal path

        # Cancel unused prefetch to avoid orphaned coroutines
        if prefetch_task and not prefetch_task.done():
            prefetch_task.cancel()

        # Normal path (no prefetch or prefetch failed)
        async for chunk in self.tts.stream(reply):
            yield chunk

    async def _safe_learn(
        self, call_id: str, user_text: str, agent_text: str, intent: str | None
    ) -> None:
        """Fire-and-forget learning that never crashes the call."""
        try:
            await self.learner.learn_from_turn(call_id, user_text, agent_text, intent)
        except Exception as exc:
            logger.warning("Background learning failed (non-fatal): %s", exc)

    @staticmethod
    def _enforce_sales_style(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            cleaned = "Main aapki madad ke liye hoon. Aap model, budget, ya usage share kijiye."
        if not cleaned.endswith((".", "?", "!")):
            cleaned += "."
        if len(cleaned.split()) < 6:
            cleaned += " Sir, aap apni exact requirement batayein, main best option suggest karti hoon."
        return cleaned
