"""
FastAPI application factory.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import router


def create_app() -> FastAPI:
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
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1", tags=["research"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    return app


app = create_app()
