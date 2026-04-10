from __future__ import annotations

import importlib
import os
import uuid
from pathlib import Path

from src.learning.vector_store import VectorMemory


class FileLearner:
    def __init__(self, memory: VectorMemory) -> None:
        self.memory = memory

    def ingest(self, filepath: str) -> str:
        path = Path(filepath)
        text = ""

        if path.suffix.lower() == ".pdf":
            pdfplumber = importlib.import_module("pdfplumber") if importlib.util.find_spec("pdfplumber") else None
            if pdfplumber:
                with pdfplumber.open(path) as pdf:
                    text = "\n".join((page.extract_text() or "") for page in pdf.pages)
            else:
                text = f"PDF file indexed: {path.name}"
        elif path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            pil_mod = importlib.import_module("PIL.Image") if importlib.util.find_spec("PIL") else None
            tess_mod = importlib.import_module("pytesseract") if importlib.util.find_spec("pytesseract") else None
            if pil_mod and tess_mod:
                img = pil_mod.open(path)
                text = tess_mod.image_to_string(img)
            else:
                text = f"Image file indexed: {path.name}"
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        normalized = " ".join(text.split())
        self.memory.add(str(uuid.uuid4()), normalized, {"source": os.path.basename(filepath)})
        return normalized
