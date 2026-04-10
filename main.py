from __future__ import annotations

import importlib.util

from src.config import settings  # noqa: F401 — triggers .env loading

# Expose `app` at module level so that `uvicorn main:app` works
# (used by render.yaml and standard deployment).
_has_fastapi = importlib.util.find_spec("fastapi") is not None
if _has_fastapi:
    from src.app import app  # noqa: F401
else:
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    _has_uvicorn = importlib.util.find_spec("uvicorn") is not None

    if app is not None and _has_uvicorn:
        import uvicorn

        uvicorn.run(app, host=settings.host, port=settings.port)
    else:
        import asyncio

        from src.qa_validation import main as qa_main

        print("FastAPI/Uvicorn not installed. Running local QA simulation mode.")
        asyncio.run(qa_main())
