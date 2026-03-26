"""
FastAPI application factory.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import router
from src.logging_config import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(
        title="S4 Research Intelligence",
        description=(
            "RAG-powered documentary research assistant for "
            "'S4: The Bob Lazar Story'. Provides source-weighted retrieval, "
            "contradiction detection, and timeline extraction across a "
            "heterogeneous research corpus."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1", tags=["research"])

    @app.get("/health")
    async def health():
        import httpx

        ollama_ok = False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{settings.llm_base_url}/api/tags")
                ollama_ok = resp.status_code == 200
        except Exception:
            pass

        return {
            "status": "ok" if ollama_ok else "degraded",
            "version": "1.0.0",
            "ollama": "connected" if ollama_ok else "unavailable",
            "llm_model": settings.llm_model,
        }

    return app


app = create_app()
