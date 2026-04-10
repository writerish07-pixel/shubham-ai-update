from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EndOfSpeechDetector:
    silence_ms_threshold: int = 650
    min_utterance_ms: int = 400

    _silence_ms: int = 0
    _speech_ms: int = 0

    def reset(self) -> None:
        self._silence_ms = 0
        self._speech_ms = 0

    def ingest_frame(self, is_speech: bool, frame_ms: int = 20) -> bool:
        if is_speech:
            self._speech_ms += frame_ms
            self._silence_ms = 0
            return False

        self._silence_ms += frame_ms
        if self._speech_ms >= self.min_utterance_ms and self._silence_ms >= self.silence_ms_threshold:
            self.reset()
            return True
        return False
