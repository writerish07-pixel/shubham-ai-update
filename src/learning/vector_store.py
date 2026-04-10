from __future__ import annotations

import os
from dataclasses import dataclass

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryItem:
    text: str
    metadata: dict


class VectorMemory:
    def __init__(self) -> None:
        self._in_memory: list[MemoryItem] = []
        self._mode = "memory"
        self._collection = None

        os.makedirs(settings.learning_store_dir, exist_ok=True)
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            client = chromadb.PersistentClient(path=settings.learning_store_dir)
            ef = embedding_functions.DefaultEmbeddingFunction()
            self._collection = client.get_or_create_collection("calls", embedding_function=ef)
            self._mode = "chroma"
        except Exception as exc:
            logger.warning("Chroma unavailable, using in-memory store: %s", exc)

    def add(self, item_id: str, text: str, metadata: dict) -> None:
        if self._mode == "chroma":
            self._collection.add(ids=[item_id], documents=[text], metadatas=[metadata])
            return
        self._in_memory.append(MemoryItem(text=text, metadata=metadata))

    def search(self, query: str, k: int = 3) -> list[str]:
        if self._mode == "chroma":
            result = self._collection.query(query_texts=[query], n_results=k)
            return result.get("documents", [[]])[0]

        query_terms = set(query.lower().split())
        scored = []
        for item in self._in_memory:
            overlap = len(query_terms.intersection(set(item.text.lower().split())))
            scored.append((overlap, item.text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:k] if score > 0]
