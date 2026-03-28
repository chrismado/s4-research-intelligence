"""Tests for benchmark modules."""

from src.evaluation.benchmarks.latency import LatencyProfiler, LatencyReport, LatencyResult
from src.evaluation.benchmarks.memory import GPUSnapshot, MemoryProfile
from src.evaluation.benchmarks.quantization import (
    DEFAULT_CONFIGS,
    QuantizationBenchmark,
    QuantizationConfig,
    QuantizationReport,
    QuantizationResult,
)
from src.evaluation.benchmarks.scaling import ScalePoint, ScalingBenchmark
from src.evaluation.benchmarks.throughput import ThroughputReport, ThroughputResult


class TestLatencyProfiler:
    def test_percentile_calculation(self):
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert LatencyProfiler._percentile(values, 50) == 55.0
        assert LatencyProfiler._percentile(values, 0) == 10.0
        assert LatencyProfiler._percentile(values, 100) == 100.0

    def test_percentile_empty(self):
        assert LatencyProfiler._percentile([], 50) == 0.0

    def test_latency_result_to_dict(self):
        r = LatencyResult(
            query="test",
            total_ms=150.5,
            embedding_ms=10.2,
            retrieval_ms=40.3,
            llm_inference_ms=100.0,
        )
        d = r.to_dict()
        assert d["total_ms"] == 150.5
        assert d["embedding_ms"] == 10.2

    def test_latency_report_to_dict(self):
        report = LatencyReport(
            pipeline_type="rag",
            num_queries=10,
            p50_ms=100,
            p95_ms=200,
            p99_ms=300,
        )
        d = report.to_dict()
        assert d["pipeline_type"] == "rag"
        assert d["p50_ms"] == 100


class TestThroughput:
    def test_result_to_dict(self):
        r = ThroughputResult(
            concurrency=4,
            total_queries=20,
            successful=18,
            failed=2,
            total_time_s=5.0,
            queries_per_second=3.6,
            avg_latency_ms=1100,
        )
        d = r.to_dict()
        assert d["concurrency"] == 4
        assert d["queries_per_second"] == 3.6

    def test_report_to_dict(self):
        report = ThroughputReport(peak_qps=5.0, optimal_concurrency=4)
        d = report.to_dict()
        assert d["peak_qps"] == 5.0


class TestMemoryProfiler:
    def test_snapshot_to_dict(self):
        snap = GPUSnapshot(
            gpu_index=0,
            name="NVIDIA RTX 4090",
            total_mb=24576,
            used_mb=8192,
            free_mb=16384,
            utilization_pct=45.0,
        )
        d = snap.to_dict()
        assert d["name"] == "NVIDIA RTX 4090"
        assert d["used_mb"] == 8192.0

    def test_profile_to_dict(self):
        profile = MemoryProfile(
            num_gpus=4,
            total_baseline_mb=32000,
            total_peak_mb=45000,
            peak_delta_mb=13000,
        )
        d = profile.to_dict()
        assert d["num_gpus"] == 4
        assert d["peak_delta_mb"] == 13000.0


class TestQuantization:
    def test_default_configs(self):
        assert len(DEFAULT_CONFIGS) == 3
        names = [c.name for c in DEFAULT_CONFIGS]
        assert "FP16" in names
        assert "INT8 KV" in names
        assert "INT4 KV" in names

    def test_result_to_dict(self):
        r = QuantizationResult(
            config_name="INT8 KV",
            tokens_per_second=52.1,
            vram_usage_mb=9800,
            time_to_first_token_ms=165,
            perplexity_delta=0.02,
            max_context_length=16384,
        )
        d = r.to_dict()
        assert d["config_name"] == "INT8 KV"
        assert d["tokens_per_second"] == 52.1

    def test_comparison_table(self):
        report = QuantizationReport(
            results=[
                QuantizationResult(
                    config_name="FP16",
                    tokens_per_second=45.2,
                    vram_usage_mb=14200,
                    time_to_first_token_ms=180,
                    max_context_length=8192,
                ),
                QuantizationResult(
                    config_name="INT8 KV",
                    tokens_per_second=52.1,
                    vram_usage_mb=9800,
                    time_to_first_token_ms=165,
                    perplexity_delta=0.02,
                    max_context_length=16384,
                ),
            ],
            model="mistral:7b",
            gpu="RTX 4090",
        )
        table = report.comparison_table()
        assert len(table) == 2
        assert table[0]["Config"] == "FP16"
        assert table[0]["Perplexity Δ"] == "baseline"
        assert table[1]["Perplexity Δ"] == "+0.02"

    def test_estimate_max_context(self):
        bench = QuantizationBenchmark.__new__(QuantizationBenchmark)
        fp16 = QuantizationConfig(name="FP16", cache_type_k="f16", cache_type_v="f16")
        int8 = QuantizationConfig(name="INT8", cache_type_k="q8_0", cache_type_v="q8_0")
        int4 = QuantizationConfig(name="INT4", cache_type_k="q4_0", cache_type_v="q4_0")

        assert bench._estimate_max_context(fp16) == 8192
        assert bench._estimate_max_context(int8) == 16384
        assert bench._estimate_max_context(int4) == 32768


class TestScaling:
    def test_scale_point_to_dict(self):
        p = ScalePoint(
            corpus_pct=0.5,
            num_documents=500,
            avg_source_recall=0.75,
            avg_confidence=0.8,
            avg_latency_ms=200,
            total_queries=40,
        )
        d = p.to_dict()
        assert d["corpus_pct"] == 0.5
        assert d["num_documents"] == 500

    def test_analyze_trend_linear(self):
        bench = ScalingBenchmark.__new__(ScalingBenchmark)
        points = [
            ScalePoint(0.25, 100, 0.40, 0.6, 100, 40),
            ScalePoint(0.50, 200, 0.55, 0.7, 110, 40),
            ScalePoint(0.75, 300, 0.70, 0.75, 120, 40),
            ScalePoint(1.00, 400, 0.85, 0.8, 130, 40),
        ]
        assert bench._analyze_trend(points) == "linear"

    def test_analyze_trend_degrading(self):
        bench = ScalingBenchmark.__new__(ScalingBenchmark)
        points = [
            ScalePoint(0.25, 100, 0.8, 0.6, 100, 40),
            ScalePoint(0.50, 200, 0.75, 0.7, 110, 40),
            ScalePoint(0.75, 300, 0.70, 0.75, 120, 40),
            ScalePoint(1.00, 400, 0.65, 0.8, 130, 40),
        ]
        assert bench._analyze_trend(points) == "degrading"
