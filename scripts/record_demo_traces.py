"""
Record real agent traces for the static demo site.

Run this locally with the agent system running:
    python scripts/record_demo_traces.py

Requires: FastAPI backend running on http://localhost:8000
          Ollama + Mistral 7B running on http://localhost:11434

Captures full agent traces (plan, execution, timing, results, sources)
and saves them as JSON files for the static GitHub Pages demo.
"""

import json
import time
from pathlib import Path

import httpx

DEMO_QUERIES = [
    {
        "id": "element-115",
        "query": "What did Bob Lazar claim about Element 115?",
        "description": "Factual lookup — single agent dispatch",
    },
    {
        "id": "los-alamos",
        "query": "Is it true that Lazar worked at Los Alamos National Laboratory?",
        "description": "Fact verification — corpus search + fact-check",
    },
    {
        "id": "timeline",
        "query": (
            "What is the chronological sequence of Lazar's"
            " public disclosures between 1988 and 1990?"
        ),
        "description": "Timeline construction — corpus search + timeline agent",
    },
    {
        "id": "government-corroboration",
        "query": (
            "Do government documents corroborate Lazar's" " claim about the S4 facility location?"
        ),
        "description": "Cross-reference — all four agents dispatched",
    },
    {
        "id": "propulsion",
        "query": "How did Lazar describe the propulsion system he allegedly worked on?",
        "description": "Deep factual — corpus search with source weighting",
    },
]


def record_trace(query_config: dict, backend_url: str = "http://localhost:8000") -> dict:
    """Run a query through the agent system and capture the complete trace."""
    start = time.time()

    response = httpx.post(
        f"{backend_url}/api/v1/research/agent",
        json={"question": query_config["query"]},
        timeout=120.0,
    )
    response.raise_for_status()
    result = response.json()

    trace_id = result.get("trace_id")
    trace = {}
    if trace_id:
        trace_response = httpx.get(
            f"{backend_url}/api/v1/research/agent/trace/{trace_id}",
            timeout=30.0,
        )
        if trace_response.status_code == 200:
            trace = trace_response.json()

    return {
        "id": query_config["id"],
        "query": query_config["query"],
        "description": query_config["description"],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_duration_ms": int((time.time() - start) * 1000),
        "result": result,
        "trace": trace,
    }


def main():
    output_dir = Path("docs/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_url = "http://localhost:8000"
    print(f"Connecting to backend at {backend_url}...")

    # Health check
    try:
        health = httpx.get(f"{backend_url}/health", timeout=5.0)
        health.raise_for_status()
        print(f"Backend healthy: {health.json()}")
    except httpx.HTTPError as e:
        print(f"ERROR: Backend not reachable at {backend_url}: {e}")
        print("Start the backend with: s4ri serve")
        return

    traces = []
    for i, query_config in enumerate(DEMO_QUERIES, 1):
        print(f"\n[{i}/{len(DEMO_QUERIES)}] Recording: {query_config['query']}")
        try:
            trace = record_trace(query_config, backend_url)
            traces.append(trace)
            duration = trace["total_duration_ms"] / 1000
            confidence = trace["result"].get("confidence", "N/A")
            print(f"  Done in {duration:.1f}s — confidence: {confidence}")

            # Save individual trace
            out_path = output_dir / f"{query_config['id']}.json"
            out_path.write_text(json.dumps(trace, indent=2))
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Save manifest
    manifest = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "Mistral 7B (Ollama)",
        "system": "S4 Research Intelligence — Multi-Agent Pipeline",
        "queries": [
            {"id": t["id"], "query": t["query"], "description": t["description"]} for t in traces
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nRecorded {len(traces)} traces to {output_dir}")


if __name__ == "__main__":
    main()
