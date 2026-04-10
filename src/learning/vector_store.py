from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from src.config import settings


@dataclass
class MemoryItem:
    id: str
    text: str
    metadata: dict


class VectorMemory:
    """Dependency-free vector-like memory with JSON persistence.

    Uses token overlap scoring to keep runtime robust even when vector DB packages
    are unavailable in restricted environments.
    """

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []
        os.makedirs(settings.learning_store_dir, exist_ok=True)
        self._db_file = Path(settings.learning_store_dir) / "memory.json"
        self._load()

    def _load(self) -> None:
        if not self._db_file.exists():
            return
        data = json.loads(self._db_file.read_text(encoding="utf-8"))
        self._items = [MemoryItem(**item) for item in data]

    def _save(self) -> None:
        data = [asdict(item) for item in self._items]
        self._db_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def add(self, item_id: str, text: str, metadata: dict) -> None:
        self._items.append(MemoryItem(id=item_id, text=text, metadata=metadata))
        self._save()

    def search(self, query: str, k: int = 3) -> list[str]:
        query_terms = set(query.lower().split())
        scored = []
        for item in self._items:
            overlap = len(query_terms.intersection(set(item.text.lower().split())))
            scored.append((overlap, item.text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:k] if score > 0]
