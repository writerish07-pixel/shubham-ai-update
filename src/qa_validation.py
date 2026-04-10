from __future__ import annotations

import asyncio
import os
import time

from src.agent.engine import ConversationEngine
from src.agent.models import ConversationState
from src.learning.learner import CallLearner
from src.learning.rag import RAGRetriever
from src.learning.vector_store import VectorMemory
from src.speech.eos import EndOfSpeechDetector
from src.speech.stt import StreamingSTT


def check_env() -> dict:
    required = ["EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "SARVAM_API_KEY", "GROQ_API_KEY"]
    return {k: bool(os.getenv(k)) for k in required}


async def run_case(engine: ConversationEngine, text: str, case_id: int):
    state = ConversationState(call_id=f"case-{case_id}")
    stt = StreamingSTT()
    eos = EndOfSpeechDetector(silence_ms_threshold=600, min_utterance_ms=200)

    for token in text.split():
        await stt.feed_text_frame(token, is_final=False)
        eos.ingest_frame(is_speech=True)

    for _ in range(40):
        done = eos.ingest_frame(is_speech=False)
        if done:
            break

    final = await stt.feed_text_frame("", is_final=True)
    start = time.perf_counter()
    reply, intent, tts_stream = await engine.generate_response(state, final.text)
    audio = bytearray()
    async for chunk in tts_stream:
        audio.extend(chunk)
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "input": text,
        "intent": intent,
        "reply": reply,
        "latency_ms": round(elapsed, 2),
        "audio_bytes": len(audio),
    }


async def main() -> None:
    memory = VectorMemory()
    learner = CallLearner(memory)
    rag = RAGRetriever(memory)
    engine = ConversationEngine(learner, rag)

    print("ENV_CHECK", check_env())

    cases = [
        "Splendor ka price kya hai?",
        "Mileage kitna deti hai?",
        "Finance available hai?",
        "Best bike under 1 lakh",
        "Aaj mausam badhiya hai but mujhe city ride ke liye reliable option chahiye",
    ]

    results = []
    for idx, case in enumerate(cases, start=1):
        result = await run_case(engine, case, idx)
        results.append(result)
        print("CASE", idx, result)

    p95 = sorted(r["latency_ms"] for r in results)[int(len(results) * 0.95) - 1]
    print("PERF", {"avg_ms": round(sum(r['latency_ms'] for r in results) / len(results), 2), "p95_ms": p95})


if __name__ == "__main__":
    asyncio.run(main())
