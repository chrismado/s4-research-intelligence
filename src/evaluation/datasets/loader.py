"""Load evaluation datasets from data/evaluation/."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


class DatasetLoader:
    """Centralized loader for all evaluation datasets."""

    def __init__(self, eval_dir: Path | None = None):
        from config.settings import settings

        self.eval_dir = eval_dir or (settings.data_dir / "evaluation")

    def load_golden_queries(self) -> list[dict]:
        """Load the golden test set."""
        return self._load("golden_queries.json")

    def load_adversarial_queries(self) -> list[dict]:
        """Load adversarial test cases."""
        return self._load("adversarial_queries.json")

    def load_unanswerable_queries(self) -> list[dict]:
        """Load unanswerable query test cases."""
        return self._load("unanswerable_queries.json")

    def load_contradiction_sets(self) -> list[dict]:
        """Load contradiction claim pairs."""
        return self._load("contradiction_sets.json")

    def load_injection_prompts(self) -> list[dict]:
        """Load prompt injection test cases."""
        return self._load("injection_prompts.json")

    def load_test_queries(self) -> list[dict]:
        """Load the original RAG test queries."""
        return self._load("test_queries.json")

    def load_agent_test_queries(self) -> list[dict]:
        """Load agent-specific test queries."""
        return self._load("agent_test_queries.json")

    def _load(self, filename: str) -> list[dict]:
        path = self.eval_dir / filename
        if not path.exists():
            logger.warning(f"Dataset not found: {path}")
            return []
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {path}: {e}")
            return []
