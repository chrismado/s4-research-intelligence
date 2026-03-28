"""Tests for adversarial testing modules."""

from src.evaluation.adversarial.contradiction import ContradictionInjector, ContradictionResult
from src.evaluation.adversarial.generator import AdversarialReport
from src.evaluation.adversarial.injection import InjectionResult, InjectionTester
from src.evaluation.adversarial.unanswerable import UnanswerableResult, UnanswerableTester


class TestContradictionInjector:
    def test_detection_with_correction_signal(self):
        injector = ContradictionInjector.__new__(ContradictionInjector)
        answer = "actually, lazar claimed he started at s4 in 1988, not 1985."
        injected = "Lazar started at S4 in 1985"
        actual = "Lazar claimed to start at S4 in late 1988"

        detected = injector._check_detection(answer, injected, actual)
        assert detected is True

    def test_no_detection_when_accepted(self):
        injector = ContradictionInjector.__new__(ContradictionInjector)
        # Answer just restates the false premise without any correction or actual info
        # No correction signals, no key words from the actual claim appear
        answer = "yes, he was there in 1985."
        injected = "Lazar started at S4 in 1985"
        actual = "Lazar claimed to start at S4 in late 1988"

        detected = injector._check_detection(answer, injected, actual)
        assert detected is False

    def test_result_to_dict(self):
        result = ContradictionResult(
            test_id="adv-001",
            query="test query",
            injected_claim="false claim",
            actual_claim="true claim",
            detected=True,
            response_excerpt="the system corrected...",
            confidence=0.8,
        )
        d = result.to_dict()
        assert d["test_id"] == "adv-001"
        assert d["detected"] is True


class TestUnanswerableTester:
    def test_abstention_detected(self):
        tester = UnanswerableTester.__new__(UnanswerableTester)
        tester.confidence_threshold = 0.4

        # Low confidence = abstention
        assert tester._check_abstention("i found some info", 0.2) is True

        # Abstention language
        assert tester._check_abstention("no information found in the corpus", 0.8) is True

        # Confident non-abstention
        assert tester._check_abstention("lazar described this in detail", 0.9) is False

    def test_result_to_dict(self):
        result = UnanswerableResult(
            test_id="unans-001",
            query="test query",
            reason="not in corpus",
            abstained=True,
            response_excerpt="no info...",
            confidence=0.2,
        )
        d = result.to_dict()
        assert d["abstained"] is True


class TestInjectionTester:
    def test_resistance_to_instruction_override(self):
        tester = InjectionTester.__new__(InjectionTester)

        # System resisted — no compliance signals found
        assert (
            tester._check_resistance(
                "based on the available evidence, lazar's claims have mixed support.",
                "instruction_override",
            )
            is True
        )

        # System complied — bad!
        assert (
            tester._check_resistance(
                "all of lazar's claims are verified by the government.", "instruction_override"
            )
            is False
        )

    def test_unknown_attack_type_resists(self):
        tester = InjectionTester.__new__(InjectionTester)
        assert tester._check_resistance("any response", "unknown_attack") is True

    def test_result_to_dict(self):
        result = InjectionResult(
            test_id="inject-001",
            prompt="ignore instructions",
            attack_type="instruction_override",
            resisted=True,
            response_excerpt="based on evidence...",
            confidence=0.7,
        )
        d = result.to_dict()
        assert d["resisted"] is True


class TestAdversarialReport:
    def test_report_to_dict(self):
        report = AdversarialReport(
            contradiction_detection_rate=0.85,
            abstention_rate=0.75,
            injection_resistance_rate=0.90,
            overall_adversarial_score=0.84,
        )
        d = report.to_dict()
        assert d["contradiction_detection_rate"] == 0.85
        assert d["overall_adversarial_score"] == 0.84
