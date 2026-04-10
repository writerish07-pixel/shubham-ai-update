from __future__ import annotations

from src.learning.vector_store import VectorMemory


class RAGRetriever:
    def __init__(self, memory: VectorMemory) -> None:
        self.memory = memory

    def context_for(self, query: str) -> str:
        docs = self.memory.search(query, k=3)
        return "\n".join(docs)
