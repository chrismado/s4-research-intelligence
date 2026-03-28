"""Tests for report generation."""

import json

from src.evaluation.report.json_export import JSONExporter
from src.evaluation.report.markdown import MarkdownReporter
from src.evaluation.report.terminal import TerminalDashboard


class TestMarkdownReporter:
    def test_generate_with_all_sections(self, tmp_path):
        reporter = MarkdownReporter()
        results = {
            "golden_set": {
                "pass_rate": 0.85,
                "total": 40,
                "passed": 34,
                "failed": 6,
            },
            "hallucination": {
                "total_claims": 120,
                "total_supported": 100,
                "total_unsupported": 15,
                "total_contradicted": 5,
                "avg_hallucination_rate": 0.125,
                "avg_fabrication_rate": 0.042,
                "avg_grounding_score": 0.833,
                "worst_queries": [],
            },
            "adversarial": {
                "contradiction_detection_rate": 0.86,
                "abstention_rate": 0.75,
                "injection_resistance_rate": 0.95,
                "overall_adversarial_score": 0.86,
            },
        }

        output = tmp_path / "test_report.md"
        md = reporter.generate(results, output_path=output)

        assert output.exists()
        assert "Evaluation Report" in md
        assert "Hallucination" in md
        assert "Adversarial" in md
        assert "0.833" in md or "83.3%" in md

    def test_empty_sections_handled(self, tmp_path):
        reporter = MarkdownReporter()
        output = tmp_path / "empty_report.md"
        md = reporter.generate({}, output_path=output)
        assert "Evaluation Report" in md


class TestJSONExporter:
    def test_export(self, tmp_path):
        exporter = JSONExporter()
        results = {
            "golden_set": {"pass_rate": 0.85, "total": 40, "passed": 34, "failed": 6},
            "regression": {"regressions": []},
        }

        output = tmp_path / "results.json"
        export = exporter.export(results, output_path=output)

        assert output.exists()
        assert export["status"] == "PASS"

        data = json.loads(output.read_text())
        assert data["status"] == "PASS"

    def test_export_with_critical_regression(self, tmp_path):
        exporter = JSONExporter()
        results = {
            "golden_set": {"pass_rate": 0.5},
            "regression": {
                "regressions": [{"severity": "critical", "metric": "pass_rate"}],
            },
        }

        export = exporter.export(results, output_path=tmp_path / "fail.json")
        assert export["status"] == "FAIL"

    def test_exit_code(self):
        exporter = JSONExporter()
        assert exporter.check_exit_code({"regression": {"regressions": []}}) == 0
        assert (
            exporter.check_exit_code({"regression": {"regressions": [{"severity": "critical"}]}})
            == 1
        )


class TestTerminalDashboard:
    def test_status_badge(self):
        dashboard = TerminalDashboard()
        assert "PASS" in dashboard._status_badge(0.9)
        assert "WARN" in dashboard._status_badge(0.75)
        assert "FAIL" in dashboard._status_badge(0.5)

    def test_display_runs_without_error(self):
        """Verify display doesn't crash with empty results."""
        dashboard = TerminalDashboard()
        # Should not raise
        dashboard.display({})
