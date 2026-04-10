"""
phrase_cache.py
Pre-generates TTS audio for common Priya phrases at startup.
"""
import logging
from difflib import SequenceMatcher
from voice import synthesize_speech
from audio_utils import _mp3_to_pcm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("shubham-ai.phrase_cache")

CACHED_PHRASES = [
    "Showroom aa jaaiye, Lal Kothi Tonk Road, Jaipur. Subah 9 se shaam 7 baje tak khula hai.",
    "Test ride bilkul free hai, koi commitment nahi. Aap kab aa sakte hain?",
    "Main manager se confirm karke bata deti hoon.",
    "Aapka WhatsApp number kya hai? Main details bhej deti hoon.",
    "Bilkul ji! Kab tak decide karenge — main tab call karungi?",
    "Koi baat nahi ji! Kabhi bhi zaroorat ho toh call karein. Dhanyavaad!",
    "Koi baat nahi ji! Main kab call karoon — aapko kab convenient rahega?",
    "Ji, ek second — main check karti hoon.",
    "EMI sirf 1,800 rupaye se shuru hoti hai — aaj test ride karein?",
    "Hamare showroom mein aaiye, main personally dikhaungi.",
]

_cache: dict[str, bytes] = {}  # phrase → PCM bytes
SIMILARITY_THRESHOLD = 0.82    # 82% match = cache hit


def build_cache() -> None:
    """Generate PCM audio for all cached phrases. Call at startup."""
    success = 0
    for phrase in CACHED_PHRASES:
        try:
            audio = synthesize_speech(phrase, "hinglish")
            if audio:
                pcm = _mp3_to_pcm(audio)
                if pcm:
                    _cache[phrase] = pcm
                    success += 1
                    log.info(f"[PhraseCache] Cached: '{phrase[:50]}' ({len(pcm)} bytes)")
        except Exception as e:
            log.warning(f"[PhraseCache] Failed to cache '{phrase[:40]}': {e}")
    log.info(f"[PhraseCache] Built {success}/{len(CACHED_PHRASES)} phrases")


def get_cached_audio(text: str) -> bytes | None:
    """
    Return cached PCM if text is a close match to a cached phrase.
    Returns None if no match — caller should then use Sarvam TTS.
    """
    text_clean = text.strip().lower()

    # 1. Exact match first (fastest)
    for phrase, pcm in _cache.items():
        if text_clean == phrase.lower():
            log.info(f"[PhraseCache] Exact hit: '{text[:50]}'")
            return pcm

    # 2. Fuzzy match
    best_ratio = 0.0
    best_pcm = None
    for phrase, pcm in _cache.items():
        ratio = SequenceMatcher(None, text_clean, phrase.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_pcm = pcm

    if best_ratio >= SIMILARITY_THRESHOLD:
        log.info(f"[PhraseCache] Fuzzy hit ({best_ratio:.2f}): '{text[:50]}'")
        return best_pcm

    return None