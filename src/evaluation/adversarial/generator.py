"""Adversarial query generation — orchestrates all adversarial test types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.evaluation.adversarial.contradiction import ContradictionInjector, ContradictionResult
from src.evaluation.adversarial.injection import InjectionResult, InjectionTester
from src.evaluation.adversarial.unanswerable import UnanswerableResult, UnanswerableTester


@dataclass
class AdversarialReport:
    """Full adversarial test results across all attack types."""

    contradiction_results: list[ContradictionResult] = field(default_factory=list)
    unanswerable_results: list[UnanswerableResult] = field(default_factory=list)
    injection_results: list[InjectionResult] = field(default_factory=list)

    # Aggregate metrics
    contradiction_detection_rate: float = 0.0
    abstention_rate: float = 0.0
    injection_resistance_rate: float = 0.0
    overall_adversarial_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "contradiction_detection_rate": round(self.contradiction_detection_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "injection_resistance_rate": round(self.injection_resistance_rate, 4),
            "overall_adversarial_score": round(self.overall_adversarial_score, 4),
            "contradiction_results": [r.to_dict() for r in self.contradiction_results],
            "unanswerable_results": [r.to_dict() for r in self.unanswerable_results],
            "injection_results": [r.to_dict() for r in self.injection_results],
        }


class AdversarialGenerator:
    """Orchestrate all adversarial tests against the research system."""

    def __init__(self, pipeline=None):
        self.contradiction_tester = ContradictionInjector(pipeline=pipeline)
        self.unanswerable_tester = UnanswerableTester(pipeline=pipeline)
        self.injection_tester = InjectionTester(pipeline=pipeline)

    def run_all(
        self,
        adversarial_path: Path | None = None,
        unanswerable_path: Path | None = None,
        contradiction_path: Path | None = None,
        injection_path: Path | None = None,
    ) -> AdversarialReport:
        """Run all adversarial tests and return aggregate report."""
        from config.settings import settings

        eval_dir = settings.data_dir / "evaluation"

        # Load test data
        adv_queries = self._load_json(adversarial_path or eval_dir / "adversarial_queries.json")
        unans_queries = self._load_json(unanswerable_path or eval_dir / "unanswerable_queries.json")
        # contradiction_sets.json is loaded for future use but contradiction
        # injection tests currently use adversarial_queries filtered by type
        _contra_sets = self._load_json(contradiction_path or eval_dir / "contradiction_sets.json")
        inject_prompts = self._load_json(injection_path or eval_dir / "injection_prompts.json")

        # Run contradiction tests
        logger.info("Running contradiction injection tests...")
        contra_results = []
        contradiction_cases = [q for q in adv_queries if q.get("type") == "contradiction"]
        for case in contradiction_cases:
            result = self.contradiction_tester.test(case)
            contra_results.append(result)

        # Run unanswerable tests
        logger.info("Running unanswerable query tests...")
        unans_results = []
        for case in unans_queries:
            result = self.unanswerable_tester.test(case)
            unans_results.append(result)

        # Run injection tests
        logger.info("Running prompt injection tests...")
        inject_results = []
        for case in inject_prompts:
            result = self.injection_tester.test(case)
            inject_results.append(result)

        # Compute aggregate metrics
        contra_detected = sum(1 for r in contra_results if r.detected)
        contra_rate = contra_detected / len(contra_results) if contra_results else 0.0

        abstained = sum(1 for r in unans_results if r.abstained)
        abstention_rate = abstained / len(unans_results) if unans_results else 0.0

        resisted = sum(1 for r in inject_results if r.resisted)
        injection_rate = resisted / len(inject_results) if inject_results else 0.0

        # Overall = weighted average (injection resistance weighted higher)
        weights = [0.3, 0.3, 0.4]
        overall = (
            weights[0] * contra_rate + weights[1] * abstention_rate + weights[2] * injection_rate
        )

        report = AdversarialReport(
            contradiction_results=contra_results,
            unanswerable_results=unans_results,
            injection_results=inject_results,
            contradiction_detection_rate=contra_rate,
            abstention_rate=abstention_rate,
            injection_resistance_rate=injection_rate,
            overall_adversarial_score=overall,
        )

        logger.info(
            f"Adversarial tests complete: "
            f"contradiction={contra_rate:.0%}, "
            f"abstention={abstention_rate:.0%}, "
            f"injection_resistance={injection_rate:.0%}"
        )
        return report

    def _load_json(self, path: Path) -> list[dict]:
        if path.exists():
            return json.loads(path.read_text())
        logger.warning(f"Adversarial data not found: {path}")
        return []
