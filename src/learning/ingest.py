from __future__ import annotations

import os
import uuid
from pathlib import Path

import pdfplumber
import pytesseract
from PIL import Image

from src.learning.vector_store import VectorMemory


class FileLearner:
    def __init__(self, memory: VectorMemory) -> None:
        self.memory = memory

    def ingest(self, filepath: str) -> str:
        path = Path(filepath)
        text = ""
        if path.suffix.lower() == ".pdf":
            with pdfplumber.open(path) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        elif path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        normalized = " ".join(text.split())
        self.memory.add(str(uuid.uuid4()), normalized, {"source": os.path.basename(filepath)})
        return normalized
