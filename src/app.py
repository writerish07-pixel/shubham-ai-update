from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.agent.engine import ConversationEngine
from src.agent.models import ConversationState
from src.config import settings
from src.learning.ingest import FileLearner
from src.learning.learner import CallLearner
from src.learning.rag import RAGRetriever
from src.learning.vector_store import VectorMemory
from src.speech.eos import EndOfSpeechDetector
from src.speech.stt import StreamingSTT

app = FastAPI(title=settings.app_name, version="3.0.0")

memory = VectorMemory()
learner = CallLearner(memory)
rag = RAGRetriever(memory)
file_learner = FileLearner(memory)
engine = ConversationEngine(learner, rag)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.post("/learning/upload")
async def learning_upload(file: UploadFile = File(...)):
    Path(settings.docs_dir).mkdir(parents=True, exist_ok=True)
    destination = Path(settings.docs_dir) / file.filename
    destination.write_bytes(await file.read())
    text = file_learner.ingest(str(destination))
    return JSONResponse({"stored": file.filename, "chars": len(text)})


@app.websocket("/ws/call/{call_id}")
async def ws_call(websocket: WebSocket, call_id: str):
    await websocket.accept()
    state = ConversationState(call_id=call_id)
    eos = EndOfSpeechDetector(settings.eos_silence_ms, settings.eos_min_utterance_ms)
    stt = StreamingSTT()
    tts_task: asyncio.Task | None = None

    try:
        await websocket.send_json({
            "event": "agent_ready",
            "message": "Namaste sir, Hero dealership se bol raha hoon. Aap kaunsi bike explore karna chahenge?",
        })

        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            if event == "user_interrupt":
                state.interrupted = True
                if tts_task and not tts_task.done():
                    tts_task.cancel()
                await websocket.send_json({"event": "interrupt_ack"})
                continue

            if event != "stt_frame":
                continue

            text_frame = str(data.get("text", ""))
            is_speech = bool(text_frame.strip())
            frame = await stt.feed_text_frame(text_frame, is_final=False)

            if eos.ingest_frame(is_speech=is_speech) or data.get("final", False):
                final_chunk = await stt.feed_text_frame("", is_final=True)
                utterance = final_chunk.text.strip()
                if not utterance:
                    continue

                reply, intent, audio_stream = await engine.generate_response(state, utterance)
                await websocket.send_json({"event": "agent_text", "text": reply, "intent": intent})

                async def _push_audio():
                    async for audio_chunk in audio_stream:
                        await websocket.send_bytes(audio_chunk)
                    await websocket.send_json({"event": "agent_audio_end"})

                tts_task = asyncio.create_task(_push_audio())

    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"event": "error", "message": str(exc)})
    finally:
        if tts_task and not tts_task.done():
            tts_task.cancel()
