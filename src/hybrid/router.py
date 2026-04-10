from __future__ import annotations

from dataclasses import dataclass

from src.hybrid.intents import IntentClassifier
from src.hybrid.scripts import SCRIPT_RESPONSES


@dataclass
class HybridDecision:
    source: str
    text: str
    intent: str | None = None


class HybridRouter:
    def __init__(self) -> None:
        self.classifier = IntentClassifier()

    def route(self, transcript: str) -> HybridDecision:
        result = self.classifier.classify(transcript)
        if result.intent and result.intent in SCRIPT_RESPONSES:
            return HybridDecision(source="script", text=SCRIPT_RESPONSES[result.intent], intent=result.intent)
        return HybridDecision(source="llm", text=transcript, intent=result.intent)
