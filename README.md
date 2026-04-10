# Hero Dealership Production Voice AI (v3)

Low-latency, hybrid-script + LLM, self-learning voice agent for two-wheeler dealership calls.

## Architecture

```
/src
  /speech     # streaming STT/TTS + end-of-speech detection
  /agent      # conversation engine + state
  /hybrid     # intent classifier + script-first router
  /llm        # Groq integration client (HTTP)
  /learning   # memory store, call learner, RAG retriever, file ingestion
  /utils      # logger/helper utilities
```

## Key Features

- Streaming pipeline orchestration
- End-of-speech detection with configurable 500–800ms silence window
- Interrupt control (`user_interrupt` event cancels ongoing TTS)
- Hybrid router with script-first responses for high-frequency intents
- Async self-learning pipeline to avoid call-path latency impact
- RAG retrieval from past calls and uploaded docs
- PDF/JPEG ingestion via `/learning/upload`

## Run

```bash
python main.py
```

- If `fastapi` + `uvicorn` are installed, server mode starts.
- If not installed, app auto-runs QA simulation mode for end-to-end validation.

## Environment Keys

- `EXOTEL_API_KEY`
- `EXOTEL_API_TOKEN`
- `SARVAM_API_KEY`
- `GROQ_API_KEY`
- `GROQ_MODEL` (default `llama-3.3-70b-versatile`)
- `EOS_SILENCE_MS` (default `650`)
- `EOS_MIN_UTTERANCE_MS` (default `400`)
- `LEARNING_STORE_DIR` (default `data/learning`)
- `DOCS_DIR` (default `data/uploads`)
