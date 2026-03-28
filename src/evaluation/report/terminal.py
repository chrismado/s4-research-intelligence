"""Rich terminal dashboard for evaluation runs."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


class TerminalDashboard:
    """Display evaluation results as a rich terminal dashboard.

    Uses rich tables, panels, and colored indicators for professional
    presentation during eval runs and in screen recordings.
    """

    def display(self, results: dict) -> None:
        """Display the full evaluation dashboard."""
        console.print()
        console.rule("[bold blue]S4 Research Intelligence — Evaluation Dashboard[/]")
        console.print()

        self._display_summary(results)
        self._display_golden_set(results.get("golden_set", {}))
        self._display_hallucination(results.get("hallucination", {}))
        self._display_adversarial(results.get("adversarial", {}))
        self._display_benchmarks(results.get("benchmarks", {}))
        self._display_quantization(results.get("quantization", {}))
        self._display_regression(results.get("regression", {}))
        self._display_ab(results.get("ab_comparison", {}))

        console.print()
        console.rule("[bold blue]End of Evaluation[/]")

    def create_progress(self) -> Progress:
        """Create a progress bar for eval runs."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )

    def _status_badge(self, value: float, warn: float = 0.7, good: float = 0.85) -> str:
        """Return a colored status badge."""
        if value >= good:
            return "[green]PASS[/]"
        elif value >= warn:
            return "[yellow]WARN[/]"
        return "[red]FAIL[/]"

    def _display_summary(self, results: dict) -> None:
        golden = results.get("golden_set", {})
        halluc = results.get("hallucination", {})
        adv = results.get("adversarial", {})

        table = Table(title="Summary", show_header=True, header_style="bold cyan")
        table.add_column("Dimension", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")

        pass_rate = golden.get("pass_rate", 0)
        grounding = halluc.get("avg_grounding_score", 0)
        adv_score = adv.get("overall_adversarial_score", 0)

        table.add_row("Golden Set Pass Rate", f"{pass_rate:.1%}", self._status_badge(pass_rate))
        table.add_row("Grounding Score", f"{grounding:.1%}", self._status_badge(grounding))
        table.add_row("Adversarial Resistance", f"{adv_score:.1%}", self._status_badge(adv_score))

        console.print(Panel(table, border_style="blue"))

    def _display_golden_set(self, data: dict) -> None:
        if not data:
            return

        table = Table(title="Golden Set Results", show_header=True, header_style="bold cyan")
        table.add_column("Query Type", style="bold")
        table.add_column("Total", justify="right")
        table.add_column("Passed", justify="right")
        table.add_column("Pass Rate", justify="right")

        for qt, stats in data.get("by_type", {}).items():
            rate = stats.get("pass_rate", 0)
            color = "green" if rate >= 0.7 else "yellow" if rate >= 0.5 else "red"
            table.add_row(
                qt,
                str(stats.get("total", 0)),
                str(stats.get("passed", 0)),
                f"[{color}]{rate:.1%}[/]",
            )

        console.print(Panel(table, border_style="cyan"))

    def _display_hallucination(self, data: dict) -> None:
        if not data:
            return

        table = Table(title="Hallucination Analysis", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        metrics = [
            ("Total Claims", str(data.get("total_claims", 0))),
            ("Supported", str(data.get("total_supported", 0))),
            ("Unsupported", str(data.get("total_unsupported", 0))),
            ("Contradicted", str(data.get("total_contradicted", 0))),
        ]

        for name, val in metrics:
            table.add_row(name, val)

        # Scores with color
        grounding = data.get("avg_grounding_score", 0)
        halluc_rate = data.get("avg_hallucination_rate", 0)
        fab_rate = data.get("avg_fabrication_rate", 0)

        table.add_row(
            "Grounding Score",
            f"[{'green' if grounding > 0.8 else 'yellow' if grounding > 0.6 else 'red'}]{grounding:.1%}[/]",
        )
        table.add_row(
            "Hallucination Rate",
            f"[{'green' if halluc_rate < 0.15 else 'yellow' if halluc_rate < 0.3 else 'red'}]{halluc_rate:.1%}[/]",
        )
        table.add_row(
            "Fabrication Rate",
            f"[{'green' if fab_rate < 0.05 else 'yellow' if fab_rate < 0.1 else 'red'}]{fab_rate:.1%}[/]",
        )

        console.print(Panel(table, border_style="cyan"))

    def _display_adversarial(self, data: dict) -> None:
        if not data:
            return

        table = Table(title="Adversarial Testing", show_header=True, header_style="bold cyan")
        table.add_column("Test Type", style="bold")
        table.add_column("Rate", justify="right")
        table.add_column("Status", justify="center")

        metrics = [
            ("Contradiction Detection", data.get("contradiction_detection_rate", 0)),
            ("Abstention Rate", data.get("abstention_rate", 0)),
            ("Injection Resistance", data.get("injection_resistance_rate", 0)),
        ]

        for name, rate in metrics:
            table.add_row(name, f"{rate:.1%}", self._status_badge(rate))

        overall = data.get("overall_adversarial_score", 0)
        table.add_row(
            "[bold]Overall[/]",
            f"[bold]{overall:.1%}[/]",
            self._status_badge(overall),
        )

        console.print(Panel(table, border_style="cyan"))

    def _display_benchmarks(self, data: dict) -> None:
        if not data:
            return

        latency = data.get("latency", {})
        if latency:
            table = Table(
                title=f"Latency Profile ({latency.get('pipeline_type', 'N/A')})",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Percentile", style="bold")
            table.add_column("Latency (ms)", justify="right")

            table.add_row("p50", f"{latency.get('p50_ms', 0):,.0f}")
            table.add_row("p95", f"{latency.get('p95_ms', 0):,.0f}")
            table.add_row("p99", f"{latency.get('p99_ms', 0):,.0f}")
            table.add_row("Mean", f"{latency.get('mean_ms', 0):,.0f}")

            console.print(Panel(table, border_style="cyan"))

        memory = data.get("memory", {})
        if memory and memory.get("num_gpus", 0) > 0:
            table = Table(title="VRAM Usage", show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")

            table.add_row("GPUs", str(memory.get("num_gpus", 0)))
            table.add_row("Baseline", f"{memory.get('total_baseline_mb', 0):,.0f} MB")
            table.add_row("Peak", f"{memory.get('total_peak_mb', 0):,.0f} MB")
            table.add_row("Delta", f"{memory.get('peak_delta_mb', 0):,.0f} MB")

            console.print(Panel(table, border_style="cyan"))

    def _display_quantization(self, data: dict) -> None:
        if not data or not data.get("results"):
            return

        table = Table(
            title="KV Cache Quantization (TurboQuant-inspired)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Config", style="bold")
        table.add_column("Tokens/s", justify="right")
        table.add_column("VRAM (MB)", justify="right")
        table.add_column("TTFT (ms)", justify="right")
        table.add_column("Perplexity Delta", justify="right")
        table.add_column("Max Context", justify="right")

        for r in data.get("results", []):
            perp = (
                "baseline"
                if r.get("config_name") == "FP16"
                else f"{r.get('perplexity_delta', 0):+.2f}"
            )
            table.add_row(
                r.get("config_name", "N/A"),
                f"{r.get('tokens_per_second', 0):.1f}",
                f"{r.get('vram_usage_mb', 0):,.0f}",
                f"{r.get('time_to_first_token_ms', 0):.0f}",
                perp,
                f"{r.get('max_context_length', 0):,}",
            )

        console.print(Panel(table, border_style="cyan"))

    def _display_regression(self, data: dict) -> None:
        if not data:
            return

        has_reg = data.get("has_regressions", False)
        title_style = "bold red" if has_reg else "bold green"
        title = "Regression Status: REGRESSIONS DETECTED" if has_reg else "Regression Status: Clean"

        if not has_reg:
            console.print(
                Panel("[green]No regressions detected[/]", title=title, border_style="green")
            )
            return

        table = Table(title=title, show_header=True, header_style=title_style)
        table.add_column("Metric", style="bold")
        table.add_column("Previous", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Severity", justify="center")

        for r in data.get("regressions", []):
            sev_color = "red" if r.get("severity") == "critical" else "yellow"
            table.add_row(
                r["metric"],
                f"{r['previous_value']:.4f}",
                f"{r['current_value']:.4f}",
                f"[{sev_color}]{r['delta']:+.4f}[/]",
                f"[{sev_color}]{r['severity'].upper()}[/]",
            )

        console.print(Panel(table, border_style="red"))

    def _display_ab(self, data: dict) -> None:
        if not data or not data.get("comparisons"):
            return

        winner = data.get("overall_winner", "tie").upper()
        winner_color = "green" if winner == "AGENT" else "blue" if winner == "RAG" else "yellow"

        table = Table(
            title=f"Agent vs RAG A/B Comparison — Winner: [{winner_color}]{winner}[/]",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        table.add_column("RAG", justify="right")
        table.add_column("Agent", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Sig.", justify="center")
        table.add_column("Winner", justify="center")

        for c in data.get("comparisons", []):
            sig = "[green]Yes[/]" if c.get("significant") else "[dim]No[/]"
            delta_color = (
                "green" if c.get("delta", 0) > 0 else "red" if c.get("delta", 0) < 0 else "white"
            )
            w = c.get("winner", "tie")
            w_color = "green" if w == "agent" else "blue" if w == "rag" else "dim"

            table.add_row(
                c["metric"],
                f"{c['rag_mean']:.4f}",
                f"{c['agent_mean']:.4f}",
                f"[{delta_color}]{c['delta']:+.4f}[/]",
                f"{c['p_value']:.4f}",
                sig,
                f"[{w_color}]{w}[/]",
            )

        console.print(Panel(table, border_style="cyan"))
