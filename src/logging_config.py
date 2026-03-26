"""
Structured logging configuration for S4 Research Intelligence.

Uses loguru with JSON serialization for production-grade observability.
Tracks query latency, token usage, and pipeline metrics.
"""

import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from loguru import logger

from config.settings import settings


def setup_logging():
    """Configure loguru for structured JSON logging."""
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # JSON file handler for production
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        str(log_dir / "s4ri_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{message}",
        serialize=True,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )


@contextmanager
def track_latency(operation: str):
    """Context manager to track operation latency."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"{operation} completed", latency_ms=round(elapsed_ms, 2))


def log_query_metrics(func):
    """Decorator to log query pipeline metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Pipeline query complete | "
            f"latency={elapsed_ms:.0f}ms | "
            f"sources={len(result.sources)} | "
            f"contradictions={len(result.contradictions)} | "
            f"confidence={result.confidence:.2f}"
        )
        return result
    return wrapper
