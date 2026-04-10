from __future__ import annotations

import importlib.util

from src.config import settings


if __name__ == "__main__":
    has_fastapi = importlib.util.find_spec("fastapi") is not None
    has_uvicorn = importlib.util.find_spec("uvicorn") is not None

    if has_fastapi and has_uvicorn:
        import uvicorn
        from src.app import app

        uvicorn.run(app, host=settings.host, port=settings.port)
    else:
        from src.qa_validation import main as qa_main
        import asyncio

        print("FastAPI/Uvicorn not installed. Running local QA simulation mode.")
        asyncio.run(qa_main())
