from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher


INTENT_PATTERNS = {
    "price": ["price", "on road", "kitne ki", "cost", "quotation"],
    "mileage": ["mileage", "kmpl", "average", "kitna deti"],
    "offers": ["offer", "discount", "exchange", "cashback"],
    "finance": ["finance", "emi", "loan", "down payment"],
    "availability": ["available", "delivery", "stock", "color available"],
}


@dataclass
class IntentResult:
    intent: str | None
    confidence: float


class IntentClassifier:
    @staticmethod
    def _score(query: str, pattern: str) -> float:
        if pattern in query:
            return 1.0
        return SequenceMatcher(a=query, b=pattern).ratio()

    def classify(self, text: str) -> IntentResult:
        query = text.lower().strip()
        best_intent, best_score = None, 0.0
        for intent, patterns in INTENT_PATTERNS.items():
            for p in patterns:
                score = self._score(query, p)
                if score > best_score:
                    best_intent, best_score = intent, score
        return IntentResult(intent=best_intent if best_score >= 0.62 else None, confidence=best_score)
