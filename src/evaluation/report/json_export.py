"""Machine-readable JSON export for CI/CD integration."""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger


class JSONExporter:
    """Export evaluation results as machine-readable JSON.

    Designed for CI/CD integration — exit code 1 if critical regressions detected.
    """

    def export(
        self,
        results: dict,
        output_path: Path | None = None,
    ) -> dict:
        """Export results to JSON file.

        Args:
            results: Combined evaluation results dict.
            output_path: Where to write. Defaults to eval_results.json in project root.

        Returns:
            The exported dict (includes pass/fail status).
        """
        from config.settings import settings

        if output_path is None:
            output_path = settings.project_root / "eval_results.json"

        # Determine overall pass/fail
        regressions = results.get("regression", {}).get("regressions", [])
        critical = [r for r in regressions if r.get("severity") == "critical"]

        export = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "FAIL" if critical else "PASS",
            "regressions": len(regressions),
            "critical_regressions": len(critical),
            "summary": self._build_summary(results),
            "results": results,
        }

        output_path.write_text(json.dumps(export, indent=2, default=str))
        logger.info(f"JSON export written: {output_path}")
        return export

    def _build_summary(self, results: dict) -> dict:
        """Build a concise summary for CI log output."""
        golden = results.get("golden_set", {})
        halluc = results.get("hallucination", {})
        adv = results.get("adversarial", {})

        return {
            "pass_rate": golden.get("pass_rate", 0),
            "grounding_score": halluc.get("avg_grounding_score", 0),
            "adversarial_score": adv.get("overall_adversarial_score", 0),
            "total_tests": golden.get("total", 0),
            "passed": golden.get("passed", 0),
            "failed": golden.get("failed", 0),
        }

    def check_exit_code(self, results: dict) -> int:
        """Return exit code for CI/CD: 0 = pass, 1 = fail."""
        regressions = results.get("regression", {}).get("regressions", [])
        critical = [r for r in regressions if r.get("severity") == "critical"]
        return 1 if critical else 0
