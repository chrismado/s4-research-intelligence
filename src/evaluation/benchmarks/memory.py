"""VRAM profiling via NVML (pynvml) — real GPU memory numbers from RTX 4090s."""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger


@dataclass
class GPUSnapshot:
    """VRAM snapshot for a single GPU."""

    gpu_index: int
    name: str
    total_mb: float
    used_mb: float
    free_mb: float
    utilization_pct: float

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "name": self.name,
            "total_mb": round(self.total_mb, 1),
            "used_mb": round(self.used_mb, 1),
            "free_mb": round(self.free_mb, 1),
            "utilization_pct": round(self.utilization_pct, 1),
        }


@dataclass
class MemoryProfile:
    """VRAM usage profile across baseline, peak, and per-query states."""

    num_gpus: int = 0
    baseline: list[GPUSnapshot] = field(default_factory=list)
    peak: list[GPUSnapshot] = field(default_factory=list)
    during_query: list[list[GPUSnapshot]] = field(default_factory=list)
    total_baseline_mb: float = 0.0
    total_peak_mb: float = 0.0
    peak_delta_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "num_gpus": self.num_gpus,
            "total_baseline_mb": round(self.total_baseline_mb, 1),
            "total_peak_mb": round(self.total_peak_mb, 1),
            "peak_delta_mb": round(self.peak_delta_mb, 1),
            "baseline": [g.to_dict() for g in self.baseline],
            "peak": [g.to_dict() for g in self.peak],
        }


class MemoryProfiler:
    """Profile GPU VRAM usage during inference using pynvml.

    Captures baseline usage (model loaded, no query) and peak usage
    during multi-agent query execution. Tracks per-GPU stats across
    all available GPUs.
    """

    def __init__(self):
        self._nvml_available = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_available = True
            self._pynvml = pynvml
        except Exception as e:
            logger.warning(f"pynvml not available (GPU profiling disabled): {e}")

    def is_available(self) -> bool:
        return self._nvml_available

    def get_gpu_count(self) -> int:
        if not self._nvml_available:
            return 0
        return self._pynvml.nvmlDeviceGetCount()

    def snapshot(self) -> list[GPUSnapshot]:
        """Take a VRAM snapshot across all GPUs."""
        if not self._nvml_available:
            return []

        snapshots = []
        for i in range(self.get_gpu_count()):
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)
            info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = self._pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = 0.0

            snapshots.append(
                GPUSnapshot(
                    gpu_index=i,
                    name=name,
                    total_mb=info.total / (1024 * 1024),
                    used_mb=info.used / (1024 * 1024),
                    free_mb=info.free / (1024 * 1024),
                    utilization_pct=gpu_util,
                )
            )

        return snapshots

    def profile_query(
        self,
        query_fn,
        sample_interval_s: float = 0.1,
    ) -> MemoryProfile:
        """Profile VRAM during a query execution.

        Args:
            query_fn: Callable that runs the query. Will be called once.
            sample_interval_s: How often to sample VRAM during execution.
        """
        if not self._nvml_available:
            return MemoryProfile()

        # Baseline
        baseline = self.snapshot()
        baseline_total = sum(g.used_mb for g in baseline)

        # Run query with VRAM sampling
        peak_snapshots = list(baseline)
        peak_total = baseline_total
        samples: list[list[GPUSnapshot]] = []

        import threading

        stop_event = threading.Event()

        def sampler():
            nonlocal peak_snapshots, peak_total
            while not stop_event.is_set():
                snap = self.snapshot()
                total = sum(g.used_mb for g in snap)
                samples.append(snap)
                if total > peak_total:
                    peak_total = total
                    peak_snapshots = snap
                stop_event.wait(sample_interval_s)

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()

        try:
            query_fn()
        finally:
            stop_event.set()
            thread.join(timeout=2.0)

        return MemoryProfile(
            num_gpus=self.get_gpu_count(),
            baseline=baseline,
            peak=peak_snapshots,
            during_query=samples,
            total_baseline_mb=baseline_total,
            total_peak_mb=peak_total,
            peak_delta_mb=peak_total - baseline_total,
        )

    def __del__(self):
        import contextlib

        if self._nvml_available:
            with contextlib.suppress(Exception):
                self._pynvml.nvmlShutdown()
