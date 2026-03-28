"""Tests for dataset loading."""

import json

from src.evaluation.datasets.loader import DatasetLoader


class TestDatasetLoader:
    def test_load_golden_queries(self, tmp_path):
        data = [{"id": "test-001", "query": "test", "query_type": "factual"}]
        (tmp_path / "golden_queries.json").write_text(json.dumps(data))

        loader = DatasetLoader(eval_dir=tmp_path)
        result = loader.load_golden_queries()
        assert len(result) == 1
        assert result[0]["id"] == "test-001"

    def test_load_missing_file(self, tmp_path):
        loader = DatasetLoader(eval_dir=tmp_path)
        result = loader.load_adversarial_queries()
        assert result == []

    def test_load_invalid_json(self, tmp_path):
        (tmp_path / "golden_queries.json").write_text("not json{{{")
        loader = DatasetLoader(eval_dir=tmp_path)
        result = loader.load_golden_queries()
        assert result == []

    def test_all_loaders_exist(self, tmp_path):
        loader = DatasetLoader(eval_dir=tmp_path)
        # Just verify all methods are callable
        assert callable(loader.load_golden_queries)
        assert callable(loader.load_adversarial_queries)
        assert callable(loader.load_unanswerable_queries)
        assert callable(loader.load_contradiction_sets)
        assert callable(loader.load_injection_prompts)
        assert callable(loader.load_test_queries)
        assert callable(loader.load_agent_test_queries)
