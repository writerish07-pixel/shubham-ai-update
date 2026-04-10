from __future__ import annotations

import asyncio
import time
import uuid

from src.learning.vector_store import VectorMemory


class CallLearner:
    def __init__(self, memory: VectorMemory) -> None:
        self.memory = memory

    async def learn_from_turn(self, call_id: str, user_text: str, agent_text: str, intent: str | None) -> None:
        await asyncio.sleep(0)
        summary = (
            f"intent={intent or 'unknown'} | customer='{user_text}' | agent='{agent_text}'"
        )
        self.memory.add(
            item_id=f"{call_id}-{uuid.uuid4()}",
            text=summary,
            metadata={"call_id": call_id, "intent": intent or "unknown", "ts": int(time.time())},
        )

    async def extract_call_signals(self, transcript: str) -> dict:
        await asyncio.sleep(0)
        t = transcript.lower()
        objections = [phrase for phrase in ["mehenga", "expensive", "later", "soch ke batata"] if phrase in t]
        lost_reason = "price" if any(x in t for x in ["mehenga", "expensive"]) else "unknown"
        return {
            "intent": "finance" if "emi" in t else "general",
            "objections": objections,
            "lost_reason": lost_reason,
        }
