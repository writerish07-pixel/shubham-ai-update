"""
voice.py
Handles Speech-to-Text (Sarvam primary, Deepgram fallback) and Text-to-Speech (Sarvam).

FIXES:
- Sarvam TTS payload uses "inputs" (list) not "text" (string)
- Sarvam TTS model is "bulbul:v1" not "bulbul:v3"
- Sarvam TTS response key is "audios" (list) not "audio" (string)
- synthesize_speech() now accepts both "hi-IN" style codes and "hindi"/"hinglish" labels
"""
import io, base64, re, requests
import config

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


# ── LANGUAGE NORMALISATION ────────────────────────────────────────────────────

def _lang_to_code(language: str) -> str:
    """
    Accept either a friendly label ("hindi", "hinglish", "english") or an
    IETF tag ("hi-IN", "en-IN") and return the Sarvam language code.
    """
    language = (language or "").lower().strip()
    mapping = {
        # friendly labels
        "hindi":       "hi-IN",
        "hinglish":    "hi-IN",
        "rajasthani":  "hi-IN",  # Sarvam doesn't have Rajasthani; use Hindi
        "english":     "en-IN",
        # IETF tags (pass-through)
        "hi-in":       "hi-IN",
        "en-in":       "en-IN",
        "hi":          "hi-IN",
        "en":          "en-IN",
    }
    return mapping.get(language, "hi-IN")


def _normalize_lang(code: str) -> str:
    """Convert an IETF language code to a friendly label used internally."""
    code = code.lower()
    if "en" in code:
        return "english"
    if "hi" in code:
        return "hindi"
    if "raj" in code:
        return "rajasthani"
    return "hinglish"


def _detect_audio_mime(audio_bytes: bytes) -> str:
    """Detect audio MIME type from magic bytes. Exotel recordings can be WAV or MP3."""
    if len(audio_bytes) >= 4 and audio_bytes[:4] == b'RIFF':
        return "audio/wav"
    if len(audio_bytes) >= 3 and audio_bytes[:3] == b'ID3':
        return "audio/mpeg"
    if len(audio_bytes) >= 2 and audio_bytes[:2] in (
        b'\xff\xfb', b'\xff\xfa', b'\xff\xf3', b'\xff\xf2'
    ):
        return "audio/mpeg"
    return "audio/wav"   # safe default


# ── SPEECH TO TEXT ────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, language_hint: str = "hi-IN") -> dict:
    """
    Convert audio bytes → text.
    Returns {"text": "...", "language": "hindi/english/hinglish", "confidence": float}
    Tries Sarvam first (better for Hindi/Hinglish), falls back to Deepgram.
    """
    try:
        result = _sarvam_stt(audio_bytes, _lang_to_code(language_hint))
        if result.get("text"):
            return result
    except Exception as e:
        print(f"[Voice] Sarvam STT failed: {e}, trying Deepgram")

    try:
        return _deepgram_stt(audio_bytes)
    except Exception as e:
        print(f"[Voice] Deepgram STT failed: {e}")
        return {"text": "", "language": "unknown", "confidence": 0.0}


def _sarvam_stt(audio_bytes: bytes, language: str = "hi-IN") -> dict:
    if not config.SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not configured")

    mime = _detect_audio_mime(audio_bytes)
    ext  = "wav" if mime == "audio/wav" else "mp3"

    headers = {"api-subscription-key": config.SARVAM_API_KEY}
    files   = {"file": (f"audio.{ext}", io.BytesIO(audio_bytes), mime)}
    data    = {
        "model":           "saarika:v2.5",
        "language_code":   language,
        "with_timestamps": "false",
    }

    print(f"[STT] Sending to Sarvam: {len(audio_bytes)} bytes, mime={mime}, lang={language}, first4={audio_bytes[:4]}")
    r = requests.post(SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=15)
    
    if not r.ok:
        print(f"[Voice] Sarvam STT failed: {r.status_code}, response: {r.text[:300]}")
        raise Exception(f"Sarvam STT {r.status_code}: {r.text[:200]}")

    result = r.json()
    transcript = result.get("transcript", "")
    lang_code  = result.get("language_code", "hi-IN")

    return {
        "text":       transcript,
        "language":   _normalize_lang(lang_code),
        "confidence": 0.9,
    }

def _deepgram_stt(audio_bytes: bytes) -> dict:
    if not config.DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY not configured")

    mime_type = _detect_audio_mime(audio_bytes)
    headers   = {
        "Authorization": f"Token {config.DEEPGRAM_API_KEY}",
        "Content-Type":  mime_type,
    }
    params = {
        "model":           "nova-2",
        "language":        "hi",
        "detect_language": "true",
        "smart_format":    "true",
        "punctuate":       "true",
    }

    r = requests.post(
        "https://api.deepgram.com/v1/listen",
        headers=headers, params=params,
        data=audio_bytes, timeout=15,
    )
    r.raise_for_status()

    data     = r.json()
    channels = data.get("results", {}).get("channels", [{}])
    alts     = channels[0].get("alternatives", [{}])
    lang     = channels[0].get("detected_language", "hi")

    return {
        "text":       alts[0].get("transcript", ""),
        "language":   _normalize_lang(lang),
        "confidence": alts[0].get("confidence", 0.8),
    }


# ── TEXT TO SPEECH ────────────────────────────────────────────────────────────

def synthesize_speech(text: str, language: str = "hinglish") -> bytes:
    """
    Convert text → MP3 audio bytes via Sarvam AI.

    `language` can be a friendly label ("hindi", "hinglish", "english")
    OR an IETF tag ("hi-IN", "en-IN") — both are handled correctly.

    Returns b"" on any failure so callers can fall back to <Say>.
    """
    # Clean markdown / JSON blocks that should never be spoken
    text = re.sub(r'\{[^}]+\}', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```',  '', text, flags=re.DOTALL)
    text = text.strip()

    if not text:
        print("[Voice] TTS skipped — empty text after cleaning")
        return b""

    if not config.SARVAM_API_KEY:
        print("[Voice] SARVAM_API_KEY not set — skipping TTS")
        return b""

    lang_code = _lang_to_code(language)

    try:
        return _sarvam_tts(text, lang_code)
    except Exception as e:
        print(f"[Voice] Sarvam TTS failed: {e}")
        return b""


def _sarvam_tts(text: str, language: str = "hi-IN") -> bytes:
    """
    Call Sarvam /text-to-speech.

    Correct payload format (as of Sarvam API v1):
      - "inputs"  : list of strings  (NOT "text")
      - "model"   : "bulbul:v1"      (NOT "bulbul:v3")
      - "speaker" : "meera" / "pavithra" / "maitreyi" / "arvind" / "amol" / "amartya"
                    For female Hindi voice use "meera" (Priya is not a valid speaker name)
      - Response  : {"audios": ["<base64>", ...]}  (NOT "audio")
    """
    # Sarvam TTS has a 500-char limit per request — split if needed
    chunks = _split_text(text, max_chars=490)
    all_audio = b""

    headers = {
        "api-subscription-key": config.SARVAM_API_KEY,
        "Content-Type":         "application/json",
    }

    for chunk in chunks:
        payload = {
            "inputs":               [chunk],
            "target_language_code": language,
            "speaker":              "anushka",   # warm female Hindi voice (valid Sarvam speaker)
            "model":                "bulbul:v2",
            "pitch":                0,
            "pace":                 1.1,        # slightly faster for natural phone speech
            "loudness":             1.5,
            "enable_preprocessing": True,
        }

        r = requests.post(SARVAM_TTS_URL, headers=headers, json=payload, timeout=20)

        print(f"[Voice] Sarvam TTS status: {r.status_code}")
        if r.status_code != 200:
            print(f"[Voice] Sarvam TTS error body: {r.text[:400]}")
            r.raise_for_status()

        data = r.json()

        # Response is {"audios": ["base64string", ...]}
        audios = data.get("audios") or data.get("audio")
        if not audios:
            print(f"[Voice] Sarvam TTS returned no audio. Keys: {list(data.keys())}")
            raise ValueError("No audio in Sarvam TTS response")

        # audios can be a list (one per input string) or a bare string
        audio_b64 = audios[0] if isinstance(audios, list) else audios
        all_audio += base64.b64decode(audio_b64)

    return all_audio


def _split_text(text: str, max_chars: int = 490) -> list[str]:
    """
    Split long text into chunks ≤ max_chars, breaking on sentence boundaries.
    Sarvam TTS silently truncates inputs over ~500 chars.
    """
    if len(text) <= max_chars:
        return [text]

    chunks, current = [], ""
    for sentence in re.split(r'(?<=[।.!?])\s+', text):
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence is still too long, hard-split it
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence

    if current:
        chunks.append(current)

    return chunks