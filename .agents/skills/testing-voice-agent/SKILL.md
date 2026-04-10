# Testing the Voice Agent

## Prerequisites
- Python 3.10+
- `pip install -r requirements.txt`
- `pip install websockets` (for WebSocket testing)

## Devin Secrets Needed
- `GROQ_API_KEY` — for real LLM streaming (optional, fallback works without it)
- `SARVAM_API_KEY` — for real TTS/STT audio (optional, text fallback works without it)
- `EXOTEL_API_KEY` and `EXOTEL_API_TOKEN` — for telephony integration (optional)

## Starting the Server
```bash
cd /home/ubuntu/repos/shubham-ai-update
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
The server initializes all singletons (VectorMemory, CallLearner, RAGRetriever, FileLearner, ConversationEngine) in the FastAPI lifespan handler.

## Key Endpoints
| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Returns status + which API keys are configured |
| `/learning/upload` | POST | Upload PDF/image files for learning ingestion |
| `/exotel/answer` | POST | Exotel ExoML webhook — returns XML with WebSocket URL |
| `/exotel/status` | POST | Exotel call status callback |
| `/ws/call/{call_id}` | WS | Main call WebSocket endpoint |

## WebSocket Protocol
1. Connect to `ws://localhost:8000/ws/call/{any-call-id}`
2. First message received: `{"event": "agent_ready", "message": "Namaste! Main Priya..."}`
3. Send STT frames: `{"event": "stt_frame", "text": "user text here", "final": true}`
4. Receive: `{"event": "agent_text", "text": "...", "intent": "price"|null}`
5. For interrupts: send `{"event": "user_interrupt"}`, expect `{"event": "interrupt_ack"}`

## Scripted Intents (Script-First Routing)
These intents return canned responses without needing LLM:
- `price` — keywords: price, on road, kitne ki, cost, quotation, kya price, kitna hai, rate
- `mileage` — keywords: mileage, kmpl, average, kitna deti, fuel, petrol
- `offers` — keywords: offer, discount, exchange, cashback, deal, scheme
- `finance` — keywords: finance, emi, loan, down payment, installment
- `availability` — keywords: available, delivery, stock, color available, kab milegi
- `comparison` — keywords: compare, vs, better, difference, konsi acchi
- `booking` — keywords: book, booking, advance, token, register
- `test_ride` — keywords: test ride, test drive, chalake dekhna, ride
- `service` — keywords: service, warranty, maintenance, servicing

Anything not matching these goes to LLM fallback.

## Testing Without API Keys
All endpoints work in degraded mode:
- LLM: returns empty string, `_enforce_sales_style` provides a generic fallback response
- TTS: streams UTF-8 text chunks instead of audio bytes
- STT: text frame passthrough works (no audio transcription)
- Health endpoint shows `groq_configured: false` etc.

## Common Test Patterns
```python
import asyncio, json, websockets

async def test_price():
    async with websockets.connect('ws://localhost:8000/ws/call/test-1') as ws:
        greeting = json.loads(await ws.recv())
        assert greeting['event'] == 'agent_ready'
        await ws.send(json.dumps({"event": "stt_frame", "text": "Splendor ka price kya hai?", "final": True}))
        resp = json.loads(await ws.recv())
        assert resp['intent'] == 'price'
        assert 'on-road' in resp['text'].lower()
```

## Known Limitations
- Port 8000 may be occupied by stale processes. Use `fuser -k 8000/tcp` or `ss -tlnp | grep 8000` to find and kill them before starting.
- Without GROQ_API_KEY, the LLM fallback response is generic (not contextual). This is expected.
- The talk-ratio trimming threshold (agent_ratio > 0.35) triggers from the first exchange. This is aggressive but intentional.
- The sentence splitting regex preserves decimals ("1.5 lakh") but may still split on some abbreviations like "Rs." — this is a known minor limitation.
- Empty WebSocket text frames are silently skipped (no response sent back). This is correct behavior per the EOS detector logic.
