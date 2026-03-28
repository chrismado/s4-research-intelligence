"""
CLI interface for S4 Research Intelligence.

Usage:
    s4ri ingest --manifest data/raw/manifest.json
    s4ri query "What did Bob Lazar claim about Element 115?"
    s4ri serve
    s4ri stats
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer(name="s4ri", help="S4 Research Intelligence — Documentary Research Assistant")


@app.command()
def ingest(
    manifest: Path = typer.Option(None, help="Path to JSON manifest file"),  # noqa: B008
    file: Path = typer.Option(None, help="Path to a single document"),  # noqa: B008
    source_type: str = typer.Option("production_note", help="Source type for single file"),
):
    """Ingest documents into the vector store."""
    from src.ingestion.chunker import chunk_document, chunk_documents
    from src.ingestion.loader import load_document, load_from_manifest
    from src.ingestion.vectorstore import VectorStore

    store = VectorStore()

    if manifest:
        docs = load_from_manifest(manifest)
        chunks = chunk_documents(docs)
        added = store.add_chunks(chunks)
        console.print(f"[green]OK[/green] Ingested {len(docs)} documents -> {added} chunks")
    elif file:
        doc = load_document(file, metadata_override={"source_type": source_type})
        chunks = chunk_document(doc)
        added = store.add_chunks(chunks)
        console.print(f"[green]OK[/green] Ingested '{file.name}' -> {added} chunks")
    else:
        console.print("[red]Provide --manifest or --file[/red]")
        raise typer.Exit(1)

    console.print(f"Total chunks in store: {store.count}")


@app.command()
def query(
    question: str = typer.Argument(help="Research question"),
    source_type: str = typer.Option(None, help="Filter by source type"),
    top_k: int = typer.Option(5, help="Number of sources to retrieve"),
):
    """Query the research corpus."""
    from src.models.documents import SourceType
    from src.models.queries import ResearchQuery
    from src.retrieval.pipeline import ResearchPipeline

    filters = {}
    if source_type:
        filters["source_types"] = [SourceType(source_type)]

    rq = ResearchQuery(question=question, top_k=top_k, **filters)
    pipeline = ResearchPipeline()
    response = pipeline.query(rq)

    # Display answer
    console.print(Panel(response.answer, title="[bold]Answer[/bold]", border_style="blue"))

    # Sources table
    if response.sources:
        table = Table(title="Sources", show_lines=True)
        table.add_column("File", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Relevance", justify="right")
        table.add_column("Reliability", justify="right")
        table.add_column("Combined", justify="right", style="green")

        for src in response.sources:
            table.add_row(
                src.source_file,
                src.source_type.value,
                f"{src.relevance_score:.3f}",
                f"{src.reliability_score:.2f}",
                f"{src.combined_score:.3f}",
            )
        console.print(table)

    # Contradictions
    if response.contradictions:
        console.print("\n[bold red]CONTRADICTIONS DETECTED[/bold red]")
        for c in response.contradictions:
            console.print(f"  * {c.source_a}: {c.claim_a}")
            console.print(f"    vs {c.source_b}: {c.claim_b}")
            console.print(f"    -> {c.explanation}\n")

    # Timeline
    if response.timeline:
        console.print("\n[bold]Timeline Events[/bold]")
        for e in response.timeline:
            console.print(f"  [{e.date or '?'}] {e.description} (from {e.source})")

    # Confidence
    conf_color = (
        "green" if response.confidence > 0.7 else "yellow" if response.confidence > 0.4 else "red"
    )
    console.print(f"\n[{conf_color}]Confidence: {response.confidence:.0%}[/{conf_color}]")

    if response.reasoning:
        console.print(f"[dim]Reasoning: {response.reasoning}[/dim]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
):
    """Start the FastAPI server."""
    import uvicorn

    console.print(f"[green]Starting S4 Research Intelligence API on {host}:{port}[/green]")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=True)


@app.command()
def stats():
    """Show vector store statistics."""
    from src.ingestion.vectorstore import VectorStore

    store = VectorStore()
    console.print(f"Collection: {store._collection.name}")
    console.print(f"Total chunks: {store.count}")
    console.print(f"Embedding model: {store._embeddings.model_name}")


@app.command()
def evaluate(
    test_set: Path = typer.Option(  # noqa: B008
        "data/evaluation/test_queries.json",
        help="Path to evaluation test set",
    ),
    output: Path = typer.Option(None, help="Save results to JSON file"),  # noqa: B008
    compare_agent: bool = typer.Option(False, "--compare-agent", help="A/B compare agent vs RAG"),
):
    """Run RAG evaluation pipeline against test queries."""
    import json

    from src.evaluation.evaluator import RAGEvaluator

    evaluator = RAGEvaluator()

    if compare_agent:
        console.print("[bold]Running Agent vs RAG A/B Comparison[/bold]\n")
        results = evaluator.compare_agent_vs_rag(test_set)

        table = Table(title="Agent vs RAG Comparison", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("RAG", justify="right")
        table.add_column("Agent", justify="right")
        table.add_column("Delta", justify="right", style="bold")

        rag = results["rag"]
        ag = results["agent"]
        delta = results["delta"]

        table.add_row(
            "Avg Confidence",
            f"{rag['avg_confidence']:.2%}",
            f"{ag['avg_confidence']:.2%}",
            f"{delta['confidence']:+.2%}",
        )
        table.add_row(
            "Avg Source Recall",
            f"{rag['avg_source_recall']:.2%}",
            f"{ag['avg_source_recall']:.2%}",
            f"{delta['source_recall']:+.2%}",
        )
        table.add_row(
            "Avg Date Coverage",
            f"{rag['avg_date_coverage']:.2%}",
            f"{ag['avg_date_coverage']:.2%}",
            f"{delta['date_coverage']:+.2%}",
        )
        table.add_row(
            "Avg Latency (ms)",
            f"{rag['avg_latency_ms']:.0f}",
            f"{ag['avg_latency_ms']:.0f}",
            f"{delta['latency_overhead_ms']:+.0f}",
        )
        console.print(table)
    else:
        console.print("[bold]Running RAG Evaluation Pipeline[/bold]\n")
        results = evaluator.evaluate_batch(test_set)

        table = Table(title="Evaluation Results", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Queries", str(results["total_queries"]))
        table.add_row(
            "Avg Confidence",
            f"{results['avg_confidence']:.2%}",
        )
        table.add_row(
            "Avg Source Recall",
            f"{results['avg_source_recall']:.2%}",
        )
        table.add_row(
            "Avg Date Coverage",
            f"{results['avg_date_coverage']:.2%}",
        )
        table.add_row(
            "Citation Rate",
            f"{results['citation_rate']:.0%}",
        )
        table.add_row(
            "Contradiction Accuracy",
            f"{results['contradiction_accuracy']:.0%}",
        )
        console.print(table)

        console.print("\n[bold]Individual Results[/bold]")
        for r in results["individual_results"]:
            conf_color = (
                "green" if r["confidence"] > 0.7 else "yellow" if r["confidence"] > 0.4 else "red"
            )
            console.print(f"\n  Q: {r['question']}")
            console.print(
                f"  Confidence: [{conf_color}]"
                f"{r['confidence']:.0%}"
                f"[/{conf_color}] | Sources: {r['num_sources']}"
                f" | Contradictions: "
                f"{r['num_contradictions']}"
            )

    if output:
        output.write_text(json.dumps(results, indent=2, default=str))
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command(name="eval")
def eval_cmd(
    all_tests: bool = typer.Option(False, "--all", help="Run all evaluation tests"),
    hallucination: bool = typer.Option(
        False, "--hallucination", help="Run hallucination detection"
    ),
    adversarial: bool = typer.Option(False, "--adversarial", help="Run adversarial tests"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Run benchmarks"),
    regression: bool = typer.Option(False, "--regression", help="Run regression checks"),
    quantization: bool = typer.Option(False, "--quantization", help="Run quantization benchmarks"),
    compare: bool = typer.Option(False, "--compare", help="A/B compare agent vs RAG"),
    report: Path = typer.Option(  # noqa: B008
        None, "--report", help="Save markdown report"
    ),
    dashboard: bool = typer.Option(False, "--dashboard", help="Show terminal dashboard"),
):
    """Run the evaluation suite."""
    from src.evaluation.suite import EvalSuite

    suite = EvalSuite()
    results = {}

    if all_tests:
        results = suite.run_all()
    else:
        if hallucination:
            results["hallucination"] = suite.run_hallucination().to_dict()
        if adversarial:
            results["adversarial"] = suite.run_adversarial().to_dict()
        if benchmark:
            results["benchmarks"] = suite.run_benchmarks()
        if regression:
            results["regression"] = suite.run_regression().to_dict()
        if quantization:
            results["quantization"] = suite.run_quantization().to_dict()
        if compare:
            results["ab_comparison"] = suite.run_compare().to_dict()
        if not results:
            results["golden_set"] = suite.run_golden_set().to_dict()

    if dashboard:
        from src.evaluation.report.terminal import TerminalDashboard

        dash = TerminalDashboard()
        dash.display(results)
    else:
        console.print(f"[green]Evaluation complete.[/green] " f"{len(results)} module(s) ran.")

    if report:
        from src.evaluation.report.markdown import MarkdownReporter

        reporter = MarkdownReporter()
        md = reporter.generate(results)
        report.write_text(md)
        console.print(f"[green]Report saved to {report}[/green]")


@app.command()
def agent(
    question: str = typer.Argument(
        help="Research question for the multi-agent pipeline",
    ),
    show_trace: bool = typer.Option(False, "--show-trace", help="Show the full decision trace"),
):
    """Run a multi-agent research query with planning, dispatch, and synthesis."""
    import json

    from src.agents.orchestrator import run_agent_query

    console.print(
        Panel(
            f"[bold]Query:[/bold] {question}",
            title="[bold blue]S4 Multi-Agent Research[/bold blue]",
            border_style="blue",
        )
    )

    result = run_agent_query(question)

    # Research plan
    console.print("\n[bold]Research Plan[/bold]")
    for step in result.get("research_plan", []):
        console.print(
            f"  [{step.get('priority', '?')}] "
            f"[cyan]{step['agent']}[/cyan]: "
            f"{step.get('reason', '')}"
        )

    # Answer
    console.print(
        Panel(
            result.get("synthesis", "No synthesis generated."),
            title="[bold]Answer[/bold]",
            border_style="green",
        )
    )

    # Sources
    if result.get("corpus_results"):
        table = Table(title="Corpus Results", show_lines=True)
        table.add_column("File", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Relevance", justify="right")
        table.add_column("Reliability", justify="right")
        table.add_column("Combined", justify="right", style="green")

        for src in result["corpus_results"][:5]:
            table.add_row(
                src.get("source_file", "?"),
                src.get("source_type", "?"),
                f"{src.get('relevance_score', 0):.3f}",
                f"{src.get('reliability_score', 0):.2f}",
                f"{src.get('combined_score', 0):.3f}",
            )
        console.print(table)

    # Cross-reference
    cross_ref = result.get("cross_ref_results", {})
    if cross_ref and cross_ref.get("summary"):
        console.print(
            Panel(
                cross_ref["summary"],
                title="[bold]Cross-Reference Report[/bold]",
                border_style="yellow",
            )
        )
        if cross_ref.get("contradicting"):
            console.print("[bold red]  Contradicting sources found![/bold red]")
            for c in cross_ref["contradicting"]:
                console.print(f"    - {c.get('source', '?')}: " f"{c.get('evidence', '')[:100]}")

    # Timeline
    if result.get("timeline_results"):
        console.print("\n[bold]Timeline[/bold]")
        for e in result["timeline_results"]:
            conf_color = "green" if e.get("confidence", 0) > 0.7 else "yellow"
            console.print(
                f"  [{e.get('date', '?')}] "
                f"{e.get('description', '')} "
                f"[dim](from {e.get('source', '?')})[/dim] "
                f"[{conf_color}]"
                f"{e.get('confidence', 0):.0%}"
                f"[/{conf_color}]"
            )

    # Fact-check verdicts
    if result.get("fact_check_results"):
        console.print("\n[bold]Fact-Check Verdicts[/bold]")
        for fc in result["fact_check_results"]:
            verdict = fc.get("verdict", "?")
            color = {
                "VERIFIED": "green",
                "DISPUTED": "yellow",
                "UNVERIFIABLE": "dim",
                "CONTRADICTED": "red",
            }.get(verdict, "white")
            console.print(
                f"  [{color}]{verdict}[/{color}] "
                f"{fc.get('claim', '?')[:80]} "
                f"(confidence: {fc.get('confidence', 0):.0%})"
            )
            if fc.get("reasoning"):
                console.print(f"    [dim]{fc['reasoning'][:120]}[/dim]")

    # Confidence
    conf = result.get("confidence", 0)
    conf_color = "green" if conf > 0.7 else "yellow" if conf > 0.4 else "red"
    console.print(f"\n[{conf_color}]Confidence: {conf:.0%}[/{conf_color}]")
    console.print(f"[dim]Trace ID: {result.get('trace_id', '?')}[/dim]")
    console.print(f"[dim]Retries: {result.get('retries', 0)}[/dim]")

    # Full trace
    if show_trace:
        console.print("\n[bold]Decision Trace[/bold]")
        for entry in result.get("trace", []):
            node = entry.get("node", "?")
            action = entry.get("action", "?")
            duration = entry.get("duration_ms", 0)
            error = entry.get("error")

            icon = "[red]X[/red]" if error else "[green]+[/green]"
            console.print(
                f"  {icon} [cyan]{node}[/cyan]." f"[yellow]{action}[/yellow] " f"({duration:.0f}ms)"
            )
            if entry.get("outputs"):
                out = json.dumps(entry["outputs"], default=str)[:200]
                console.print(f"    [dim]{out}[/dim]")
            if error:
                console.print(f"    [red]{error}[/red]")


@app.command()
def ui():
    """Launch the Streamlit frontend."""
    import subprocess
    import sys

    frontend_path = Path(__file__).parent / "frontend" / "app.py"
    console.print("[green]Launching S4 Research Intelligence UI...[/green]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(frontend_path)])


if __name__ == "__main__":
    app()
