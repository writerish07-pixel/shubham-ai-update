from __future__ import annotations

import asyncio

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
        chunks: list[str] = []
        try:
            async for token in self.llm.stream_reply(cleaned_input, context=context):
                chunks.append(token)
        except Exception as exc:
            logger.warning("LLM stream error: %s", exc)

        reply = self._enforce_sales_style("".join(chunks).strip())

        # 70/30 user-agent talk ratio: trim to first two sentences if agent is talking too much
        _user_ratio, agent_ratio = state.talk_ratio
        if agent_ratio > 0.35:
            sentences = [s.strip() for s in reply.replace("?", "?.").replace("!", "!.").split(".") if s.strip()]
            if len(sentences) > 2:
                reply = " ".join(
                    s if s.endswith(("?", "!")) else s + "."
                    for s in sentences[:2]
                )

        state.add_agent(reply)
        asyncio.create_task(
            self._safe_learn(state.call_id, cleaned_input, reply, decision.intent)
        )
        return reply, decision.intent, self.tts.stream(reply)

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
