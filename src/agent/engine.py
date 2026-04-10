from __future__ import annotations

import asyncio

from src.agent.models import ConversationState
from src.hybrid.router import HybridRouter
from src.learning.learner import CallLearner
from src.learning.rag import RAGRetriever
from src.llm.groq_client import GroqClient
from src.speech.tts import StreamingTTS


class ConversationEngine:
    def __init__(self, learner: CallLearner, rag: RAGRetriever) -> None:
        self.router = HybridRouter()
        self.llm = GroqClient()
        self.tts = StreamingTTS()
        self.learner = learner
        self.rag = rag

    async def generate_response(self, state: ConversationState, user_text: str):
        state.add_user(user_text)
        decision = self.router.route(user_text)

        if decision.source == "script":
            reply = self._enforce_sales_style(decision.text)
            state.add_agent(reply)
            asyncio.create_task(self.learner.learn_from_turn(state.call_id, user_text, reply, decision.intent))
            return reply, decision.intent, self.tts.stream(reply)

        context = self.rag.context_for(user_text)
        chunks = []
        async for token in self.llm.stream_reply(user_text, context=context):
            chunks.append(token)
        reply = self._enforce_sales_style("".join(chunks).strip())

        # 70/30 user-agent balance by keeping assistant concise when needed.
        user_ratio, agent_ratio = state.talk_ratio
        if agent_ratio > 0.32:
            reply = reply.split(".")[0].strip() + "."

        state.add_agent(reply)
        asyncio.create_task(self.learner.learn_from_turn(state.call_id, user_text, reply, decision.intent))
        return reply, decision.intent, self.tts.stream(reply)

    @staticmethod
    def _enforce_sales_style(text: str) -> str:
        cleaned = text.strip()
        if not cleaned.endswith((".", "?", "!")):
            cleaned += "."
        if len(cleaned.split()) < 6:
            cleaned += " Sir, aap apni exact requirement batayein, main best option suggest karta hoon."
        return cleaned
