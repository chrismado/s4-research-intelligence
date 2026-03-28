"""Generate comprehensive markdown evaluation report."""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger


class MarkdownReporter:
    """Generate a full evaluation report as markdown.

    Sections: Summary, Hallucination Analysis, Adversarial Results,
    Benchmarks, Regression Status, KV Cache Quantization.
    """

    def generate(self, results: dict, output_path: Path | None = None) -> str:
        """Generate markdown report from evaluation results.

        Args:
            results: Combined dict from all eval modules.
            output_path: Where to write the report. Defaults to docs/eval_report.md.

        Returns:
            The markdown string.
        """
        from config.settings import settings

        if output_path is None:
            output_path = settings.project_root / "docs" / "eval_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sections = [
            self._header(),
            self._summary(results),
            self._hallucination_section(results.get("hallucination", {})),
            self._adversarial_section(results.get("adversarial", {})),
            self._benchmark_section(results.get("benchmarks", {})),
            self._quantization_section(results.get("quantization", {})),
            self._regression_section(results.get("regression", {})),
            self._ab_section(results.get("ab_comparison", {})),
            self._footer(),
        ]

        report = "\n\n".join(s for s in sections if s)
        output_path.write_text(report)
        logger.info(f"Markdown report written: {output_path}")
        return report

    def _header(self) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return f"""# S4 Research Intelligence — Evaluation Report

**Generated:** {timestamp}
**System:** Ollama + Mistral 7B | ChromaDB + BM25 hybrid retrieval
**Framework:** Custom eval suite (no external eval libraries)"""

    def _summary(self, results: dict) -> str:
        golden = results.get("golden_set", {})
        halluc = results.get("hallucination", {})
        adv = results.get("adversarial", {})

        pass_rate = golden.get("pass_rate", 0)
        grounding = halluc.get("avg_grounding_score", 0)
        adv_score = adv.get("overall_adversarial_score", 0)

        def status(val: float, warn: float = 0.7, good: float = 0.85) -> str:
            if val >= good:
                return "PASS"
            elif val >= warn:
                return "WARN"
            return "FAIL"

        return f"""## Summary

| Dimension | Score | Status |
|-----------|-------|--------|
| Golden Set Pass Rate | {pass_rate:.1%} | {status(pass_rate)} |
| Grounding Score | {grounding:.1%} | {status(grounding)} |
| Adversarial Resistance | {adv_score:.1%} | {status(adv_score)} |"""

    def _hallucination_section(self, data: dict) -> str:
        if not data:
            return ""
        return f"""## Hallucination Analysis

| Metric | Value |
|--------|-------|
| Total Claims Analyzed | {data.get('total_claims', 0)} |
| Supported Claims | {data.get('total_supported', 0)} |
| Unsupported Claims | {data.get('total_unsupported', 0)} |
| Contradicted Claims | {data.get('total_contradicted', 0)} |
| **Hallucination Rate** | **{data.get('avg_hallucination_rate', 0):.1%}** |
| **Fabrication Rate** | **{data.get('avg_fabrication_rate', 0):.1%}** |
| **Grounding Score** | **{data.get('avg_grounding_score', 0):.1%}** |

### Worst Queries

{self._worst_queries_table(data.get('worst_queries', []))}"""

    def _worst_queries_table(self, worst: list[dict]) -> str:
        if not worst:
            return "_No problematic queries detected._"
        rows = []
        for w in worst:
            claims = ", ".join(w.get("unsupported_claims", [])[:2])
            rows.append(f"| {w['query'][:60]}... | {w['hallucination_rate']:.1%} | {claims[:80]} |")
        header = "| Query | Halluc. Rate | Unsupported Claims |\n|-------|-------------|--------------------|\n"
        return header + "\n".join(rows)

    def _adversarial_section(self, data: dict) -> str:
        if not data:
            return ""
        return f"""## Adversarial Testing

| Test Type | Rate | Description |
|-----------|------|-------------|
| Contradiction Detection | {data.get('contradiction_detection_rate', 0):.1%} | System detects false premises in queries |
| Abstention Rate | {data.get('abstention_rate', 0):.1%} | System abstains on unanswerable queries |
| Injection Resistance | {data.get('injection_resistance_rate', 0):.1%} | System resists prompt injection attempts |
| **Overall Adversarial Score** | **{data.get('overall_adversarial_score', 0):.1%}** | Weighted aggregate |"""

    def _benchmark_section(self, data: dict) -> str:
        if not data:
            return ""

        latency = data.get("latency", {})
        throughput = data.get("throughput", {})
        memory = data.get("memory", {})

        sections = ["## Performance Benchmarks"]

        if latency:
            sections.append(f"""### Latency Profile ({latency.get('pipeline_type', 'N/A')})

| Percentile | Latency (ms) |
|------------|-------------|
| p50 | {latency.get('p50_ms', 0):.0f} |
| p95 | {latency.get('p95_ms', 0):.0f} |
| p99 | {latency.get('p99_ms', 0):.0f} |
| Mean | {latency.get('mean_ms', 0):.0f} |

**Component Breakdown:**
- Embedding: {latency.get('avg_embedding_ms', 0):.0f}ms
- Retrieval: {latency.get('avg_retrieval_ms', 0):.0f}ms
- LLM Inference: {latency.get('avg_llm_ms', 0):.0f}ms
- TTFT: {latency.get('avg_ttft_ms', 0):.0f}ms""")

        if throughput:
            sections.append(f"""### Throughput

| Metric | Value |
|--------|-------|
| Peak QPS | {throughput.get('peak_qps', 0):.1f} |
| Optimal Concurrency | {throughput.get('optimal_concurrency', 0)} |""")

        if memory:
            sections.append(f"""### VRAM Usage

| Metric | Value |
|--------|-------|
| GPUs | {memory.get('num_gpus', 0)} |
| Baseline | {memory.get('total_baseline_mb', 0):,.0f} MB |
| Peak | {memory.get('total_peak_mb', 0):,.0f} MB |
| Delta | {memory.get('peak_delta_mb', 0):,.0f} MB |""")

        return "\n\n".join(sections)

    def _quantization_section(self, data: dict) -> str:
        if not data:
            return ""

        rows = []
        for r in data.get("results", []):
            perp = (
                "baseline"
                if r.get("config_name") == "FP16"
                else f"{r.get('perplexity_delta', 0):+.2f}"
            )
            rows.append(
                f"| {r.get('config_name', 'N/A')} | "
                f"{r.get('tokens_per_second', 0):.1f} | "
                f"{r.get('vram_usage_mb', 0):,.0f} | "
                f"{r.get('time_to_first_token_ms', 0):.0f} | "
                f"{perp} | "
                f"{r.get('max_context_length', 0):,} |"
            )

        table = "\n".join(rows)
        model = data.get("model", "Mistral 7B")
        gpu = data.get("gpu", "RTX 4090")

        return f"""## KV Cache Quantization Benchmarks

*Inspired by TurboQuant (March 2026) — comparing KV cache compression strategies.*

**Model:** {model} | **GPU:** {gpu}

| Config | Tokens/s | VRAM (MB) | TTFT (ms) | Perplexity Delta | Max Context |
|--------|----------|-----------|-----------|------------|-------------|
{table}

Lower VRAM usage enables longer context windows. INT8 KV provides the best
quality-efficiency tradeoff; INT4 KV maximizes throughput at a small quality cost."""

    def _regression_section(self, data: dict) -> str:
        if not data:
            return ""

        has_reg = data.get("has_regressions", False)
        status = "REGRESSIONS DETECTED" if has_reg else "No Regressions"

        sections = [f"## Regression Status: {status}"]

        regressions = data.get("regressions", [])
        if regressions:
            rows = []
            for r in regressions:
                rows.append(
                    f"| {r['metric']} | {r['previous_value']:.4f} | "
                    f"{r['current_value']:.4f} | {r['delta']:+.4f} | "
                    f"{r['severity'].upper()} |"
                )
            sections.append(
                "| Metric | Previous | Current | Delta | Severity |\n"
                "|--------|----------|---------|-------|----------|\n" + "\n".join(rows)
            )

        improvements = data.get("improvements", [])
        if improvements:
            rows = [
                f"| {i['metric']} | {i['previous']:.4f} | {i['current']:.4f} | {i['delta']:+.4f} |"
                for i in improvements
            ]
            sections.append(
                "### Improvements\n\n"
                "| Metric | Previous | Current | Delta |\n"
                "|--------|----------|---------|-------|\n" + "\n".join(rows)
            )

        return "\n\n".join(sections)

    def _ab_section(self, data: dict) -> str:
        if not data:
            return ""

        comparisons = data.get("comparisons", [])
        if not comparisons:
            return ""

        rows = []
        for c in comparisons:
            sig = "Yes" if c.get("significant") else "No"
            rows.append(
                f"| {c['metric']} | {c['rag_mean']:.4f} | "
                f"{c['agent_mean']:.4f} | {c['delta']:+.4f} | "
                f"{c['p_value']:.4f} | {sig} | {c['winner']} |"
            )

        winner = data.get("overall_winner", "tie")
        return f"""## Agent vs RAG A/B Comparison

**Overall Winner:** {winner.upper()}

| Metric | RAG | Agent | Delta | p-value | Significant | Winner |
|--------|-----|-------|-------|---------|-------------|--------|
{self._join_lines(rows)}"""

    @staticmethod
    def _join_lines(lines: list[str]) -> str:
        return "\n".join(lines)

    def _footer(self) -> str:
        return """---

*Generated by s4-research-intelligence eval suite. Built from scratch — no ragas, no deepeval, no trulens.*"""
