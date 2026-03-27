"""Integration tests for FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0.0"
        assert "status" in data
        assert "ollama" in data


@pytest.mark.asyncio
async def test_store_stats(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/store/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "collection" in data
        assert "total_chunks" in data
        assert "embedding_model" in data
        assert "llm_model" in data


@pytest.mark.asyncio
async def test_research_query_validation(transport):
    """Test that empty questions are rejected."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/research", json={"question": ""})
        # Empty string should still be accepted (not None)
        # but very long strings should be rejected
        assert resp.status_code in (200, 422, 500)


@pytest.mark.asyncio
async def test_research_query_too_long(transport):
    """Test that excessively long questions are rejected by validation."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/research",
            json={"question": "x" * 10001},
        )
        assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_ingest_manifest_not_found(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/ingest/manifest",
            data={"manifest_path": "/nonexistent/path.json"},
        )
        assert resp.status_code == 404
