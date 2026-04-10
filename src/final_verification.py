from __future__ import annotations

import asyncio
import time
from statistics import mean

from src.agent.engine import ConversationEngine
from src.agent.models import ConversationState
from src.learning.learner import CallLearner
from src.learning.rag import RAGRetriever
from src.learning.vector_store import VectorMemory
from src.llm.groq_client import GroqClient
from src.speech.eos import EndOfSpeechDetector
from src.speech.stt import StreamingSTT


async def run_pipeline(engine: ConversationEngine, state: ConversationState, text: str, final: bool = True):
    stt = StreamingSTT()
    eos = EndOfSpeechDetector(silence_ms_threshold=600, min_utterance_ms=200)

    for token in text.split():
        await stt.feed_text_frame(token, is_final=False)
        eos.ingest_frame(is_speech=True)

    if final:
        for _ in range(40):
            if eos.ingest_frame(is_speech=False):
                break

    final_chunk = await stt.feed_text_frame("", is_final=True)
    utterance = final_chunk.text if final else text

    start = time.perf_counter()
    reply, intent, stream = await engine.generate_response(state, utterance)
    first_chunk_ms = None
    audio = bytearray()
    async for chunk in stream:
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - start) * 1000
        audio.extend(chunk)
    total_ms = (time.perf_counter() - start) * 1000
    return {
        "intent": intent,
        "reply": reply,
        "first_chunk_ms": round(first_chunk_ms or total_ms, 2),
        "total_ms": round(total_ms, 2),
        "audio_bytes": len(audio),
    }


async def test_interruption(engine: ConversationEngine):
    state = ConversationState(call_id="interrupt")
    reply, _, stream = await engine.generate_response(state, "Best bike under 1 lakh")
    task = asyncio.create_task(_consume(stream))
    await asyncio.sleep(0)
    task.cancel()
    cancelled = False
    try:
        await task
    except asyncio.CancelledError:
        cancelled = True
    return {"reply": reply, "cancelled": cancelled}


async def _consume(stream):
    async for _ in stream:
        await asyncio.sleep(0.001)


async def test_missing_api_response():
    client = GroqClient()
    original = client._enabled
    client._enabled = True

    async def _collect():
        parts = []
        async for token in client.stream_reply("hello"):
            parts.append(token)
        return "".join(parts)

    text = await _collect()
    client._enabled = original
    return "network issue" in text.lower() or "madad" in text.lower()


async def main():
    memory = VectorMemory()
    learner = CallLearner(memory)
    rag = RAGRetriever(memory)
    engine = ConversationEngine(learner, rag)

    scenarios = [
        ("clear_query", "Splendor ka price kya hai?", True),
        ("long_sentence", "Mujhe ek aisi bike chahiye jo mileage accha de aur price bhi kam ho", True),
        ("incomplete_speech", "Mujhe ek", False),
        ("fast_speaker", "".join(["mileage "] * 30).strip(), True),
        ("random_conversation", "Hello kya chal raha hai?", True),
        ("confusing_query", "Best bike but budget thoda flexible hai aur mileage bhi important hai", True),
        ("empty_input", "", True),
        ("noise_input", "asdf ### !!!", True),
    ]

    results = {}
    for name, text, final in scenarios:
        state = ConversationState(call_id=name)
        results[name] = await run_pipeline(engine, state, text, final=final)

    interruption = await test_interruption(engine)
    api_fallback_ok = await test_missing_api_response()

    stress_latencies = []
    for i in range(100):
        state = ConversationState(call_id=f"stress-{i}")
        out = await run_pipeline(engine, state, "Splendor ka price kya hai?", True)
        stress_latencies.append(out["total_ms"])

    learning_records = len(memory._items)

    print("FINAL_RESULTS", results)
    print("INTERRUPTION", interruption)
    print("API_FALLBACK_OK", api_fallback_ok)
    print(
        "STRESS",
        {
            "runs": len(stress_latencies),
            "avg_ms": round(mean(stress_latencies), 2),
            "p95_ms": sorted(stress_latencies)[94],
            "max_ms": max(stress_latencies),
        },
    )
    print("LEARNING_RECORDS", learning_records)


if __name__ == "__main__":
    asyncio.run(main())
