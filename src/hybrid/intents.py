from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz


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
    def classify(self, text: str) -> IntentResult:
        query = text.lower().strip()
        best_intent, best_score = None, 0.0
        for intent, patterns in INTENT_PATTERNS.items():
            for p in patterns:
                score = fuzz.partial_ratio(query, p) / 100.0
                if score > best_score:
                    best_intent, best_score = intent, score
        return IntentResult(intent=best_intent if best_score >= 0.78 else None, confidence=best_score)
