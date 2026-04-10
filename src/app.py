from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from src.agent.engine import ConversationEngine
from src.agent.models import ConversationState
from src.config import settings
from src.learning.ingest import FileLearner
from src.learning.learner import CallLearner
from src.learning.rag import RAGRetriever
from src.learning.vector_store import VectorMemory
from src.speech.eos import EndOfSpeechDetector
from src.speech.stt import StreamingSTT
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---- Shared singletons (initialised in lifespan) ----
memory: VectorMemory | None = None
learner: CallLearner | None = None
rag: RAGRetriever | None = None
file_learner: FileLearner | None = None
engine: ConversationEngine | None = None

GREETING = (
    "Namaste! Main Priya bol rahi hoon, Hero dealership se. "
    "Aapko kis bike ke baare mein jaankari chahiye?"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory, learner, rag, file_learner, engine
    memory = VectorMemory()
    learner = CallLearner(memory)
    rag = RAGRetriever(memory)
    file_learner = FileLearner(memory)
    engine = ConversationEngine(learner, rag)
    logger.info("Voice agent started — modules initialised")
    yield
    # Graceful shutdown: close HTTP clients
    if engine:
        await engine.llm.close()
        await engine.tts.close()
    logger.info("Voice agent shut down cleanly")


app = FastAPI(title=settings.app_name, version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== Health =====================


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "groq_configured": bool(settings.groq_api_key),
        "sarvam_configured": bool(settings.sarvam_api_key),
        "exotel_configured": bool(settings.exotel_api_key),
    }


# ===================== Learning Upload =====================


@app.post("/learning/upload")
async def learning_upload(file: UploadFile = File(...)):
    assert file_learner is not None
    Path(settings.docs_dir).mkdir(parents=True, exist_ok=True)
    destination = Path(settings.docs_dir) / (file.filename or "upload.bin")
    destination.write_bytes(await file.read())
    text = file_learner.ingest(str(destination))
    return JSONResponse({"stored": file.filename, "chars": len(text)})


# ===================== Exotel Webhooks =====================


@app.post("/exotel/answer")
async def exotel_answer(request: Request):
    """Exotel ExoML webhook — returns TwiML-style XML to connect the call to a WebSocket stream."""
    callback_base = settings.base_url or str(request.base_url).rstrip("/")
    form = await request.form()
    call_sid = str(form.get("CallSid", "unknown"))
    ws_url = callback_base.replace("http", "ws", 1) + f"/ws/call/{call_sid}"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'  <Connect><Stream url="{ws_url}" /></Connect>'
        "</Response>"
    )
    return Response(content=xml, media_type="application/xml")


@app.post("/exotel/status")
async def exotel_status(request: Request):
    """Exotel call-status callback — triggers post-call learning."""
    form = await request.form()
    call_sid = str(form.get("CallSid", "unknown"))
    status = str(form.get("Status", ""))
    logger.info("Exotel status call_sid=%s status=%s", call_sid, status)
    return JSONResponse({"received": True})


# ===================== WebSocket Call =====================


@app.websocket("/ws/call/{call_id}")
async def ws_call(websocket: WebSocket, call_id: str):
    assert engine is not None
    await websocket.accept()
    state = ConversationState(call_id=call_id)
    eos = EndOfSpeechDetector(settings.eos_silence_ms, settings.eos_min_utterance_ms)
    stt = StreamingSTT()
    tts_task: asyncio.Task | None = None

    try:
        await websocket.send_json({
            "event": "agent_ready",
            "message": GREETING,
        })

        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            # --- Interrupt handling ---
            if event == "user_interrupt":
                state.interrupted = True
                if tts_task and not tts_task.done():
                    tts_task.cancel()
                eos.reset()
                await websocket.send_json({"event": "interrupt_ack"})
                continue

            if event != "stt_frame":
                continue

            text_frame = str(data.get("text", ""))
            is_speech = bool(text_frame.strip())
            await stt.feed_text_frame(text_frame, is_final=False)

            if eos.ingest_frame(is_speech=is_speech) or data.get("final", False):
                final_chunk = await stt.feed_text_frame("", is_final=True)
                utterance = final_chunk.text.strip()
                if not utterance:
                    continue

                state.interrupted = False
                reply, intent, audio_stream = await engine.generate_response(state, utterance)
                await websocket.send_json({"event": "agent_text", "text": reply, "intent": intent})

                async def _push_audio(stream=audio_stream):
                    try:
                        async for audio_chunk in stream:
                            if state.interrupted:
                                break
                            await websocket.send_bytes(audio_chunk)
                        await websocket.send_json({"event": "agent_audio_end"})
                    except asyncio.CancelledError:
                        pass

                if tts_task and not tts_task.done():
                    tts_task.cancel()
                tts_task = asyncio.create_task(_push_audio())

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: call_id=%s", call_id)
    except Exception as exc:
        logger.warning("WebSocket error call_id=%s: %s", call_id, exc)
        try:
            await websocket.send_json({"event": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if tts_task and not tts_task.done():
            tts_task.cancel()
        await stt.close()
