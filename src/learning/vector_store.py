from __future__ import annotations

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryItem:
    id: str
    text: str
    metadata: dict


class VectorMemory:
    """Dependency-free vector-like memory with JSON persistence.

    Uses token overlap scoring to keep runtime robust even when vector DB packages
    are unavailable in restricted environments.  Persistence is done on a background
    thread so that ``add()`` never blocks the event loop.
    """

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []
        self._lock = threading.Lock()
        self._write_executor = ThreadPoolExecutor(max_workers=1)
        try:
            os.makedirs(settings.learning_store_dir, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create learning store dir: %s", exc)
        self._db_file = Path(settings.learning_store_dir) / "memory.json"
        self._load()

    def _load(self) -> None:
        if not self._db_file.exists():
            return
        try:
            data = json.loads(self._db_file.read_text(encoding="utf-8"))
            self._items = [MemoryItem(**item) for item in data]
            logger.info("Loaded %d memory items from %s", len(self._items), self._db_file)
        except Exception as exc:
            logger.warning("Failed to load memory store: %s", exc)

    def _save(self) -> None:
        try:
            with self._lock:
                data = [asdict(item) for item in self._items]
            self._db_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to persist memory store: %s", exc)

    def _save_async(self) -> None:
        """Schedule persistence on a single-thread executor so writes are serialised."""
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(self._write_executor, self._save)
        except RuntimeError:
            self._save()

    def add(self, item_id: str, text: str, metadata: dict) -> None:
        with self._lock:
            self._items.append(MemoryItem(id=item_id, text=text, metadata=metadata))
        self._save_async()

    def search(self, query: str, k: int = 3) -> list[str]:
        query_terms = set(query.lower().split())
        with self._lock:
            scored = []
            for item in self._items:
                overlap = len(query_terms.intersection(set(item.text.lower().split())))
                scored.append((overlap, item.text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:k] if score > 0]

    @property
    def count(self) -> int:
        return len(self._items)
