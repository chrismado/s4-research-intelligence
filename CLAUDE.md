# CLAUDE.md — S4 Research Intelligence

## What this project is

Multi-agent research system for the documentary "S4: The Bob Lazar Story." Ingests a corpus of interview transcripts, FOIA documents, scientific papers, and archival references, then provides orchestrated research queries using specialized sub-agents (corpus search, cross-reference, timeline, fact-check).

## Essential commands

```bash
# Install
pip install -e ".[dev,eval]"

# Ollama must be running with Mistral loaded
ollama pull mistral:7b-instruct-v0.3-q5_K_M

# Run tests (128+ tests, should all pass)
python -m pytest tests/ -v

# Lint (must pass clean — ruff with line-length 100)
python -m ruff check src/ tests/

# Ingest corpus (run from project root)
python -c "
from src.ingestion.loader import load_from_manifest
from src.ingestion.chunker import chunk_documents
from src.ingestion.vectorstore import VectorStore
from pathlib import Path
docs = load_from_manifest(Path('data/raw/manifest.json'))
chunks = chunk_documents(docs)
store = VectorStore()
store.add_chunks(chunks)
"

# Smoke test the agent pipeline
python -c "from src.cli import app; app()" agent "What did Bob Lazar claim about Element 115?"

# Start API server
python -c "from src.cli import app; app()" serve
```

## Architecture overview

```
src/agents/orchestrator.py  — LangGraph StateGraph, main entry: run_agent_query()
src/agents/corpus_search.py — Wraps hybrid search (BM25 + semantic) + LLM analysis
src/agents/cross_reference.py — Cross-source corroboration across 6 source types
src/agents/timeline_agent.py — Date extraction + chronological ordering
src/agents/fact_checker.py  — Verdict assignment (VERIFIED/DISPUTED/UNVERIFIABLE/CONTRADICTED)
src/agents/state.py         — ResearchState TypedDict (shared state flowing through graph)
src/agents/tools.py         — Shared tools: llm_call(), corpus_search_tool(), make_trace_entry()
```

The orchestrator flow is: `analyze_query -> dispatch -> [agents] -> synthesize -> evaluate -> [retry or END]`

Routing is conditional — the orchestrator classifies query type and only dispatches relevant agents. `corpus_search` always runs; others are optional.

## Key conventions

- **All LLM calls go through `src/agents/tools.py:llm_call()`** — takes a system prompt and user prompt, calls Ollama, parses JSON response. Never call Ollama directly from agents.
- **System prompts live in `src/prompts/agent_prompts.py`** — 10 constants. Never hardcode prompts in agent code.
- **Trace entries use `make_trace_entry()`** from `tools.py` — provides real UTC timestamps. Never construct trace dicts inline.
- **Source-weighted scoring**: `combined_score = relevance * 0.65 + reliability * 0.35`. Weights are in `config/settings.py`.
- **Ruff config**: `line-length = 100`, select `["E", "F", "I", "N", "W", "UP", "B", "SIM"]`. Must pass clean before any PR.
- **Tests use `unittest.mock.patch`** to mock LLM calls and tool functions. Never require Ollama for tests.
- **Don't break the existing RAG pipeline** — `src/retrieval/pipeline.py` is the original query path. Agent layer wraps it, doesn't replace it.

## CLI entry point

The CLI is at `src/cli.py` using Typer. Entry point in pyproject.toml: `s4ri = "src.cli:app"`. But `python -m src.cli` may resolve incorrectly — use `python -c "from src.cli import app; app()"` to be safe.

Commands: `ingest`, `query`, `agent`, `evaluate`, `eval`, `stats`, `serve`, `ui`

## API

FastAPI app at `src/api/app.py`, routes at `src/api/routes.py`. Key agent endpoints:
- `POST /api/v1/research/agent` — runs `run_agent_query()` via `asyncio.to_thread`
- `GET /api/v1/research/agent/trace/{trace_id}` — retrieves stored trace
- `POST /api/v1/research/agent/stream` — SSE streaming of agent execution

## Config

All config in `config/settings.py` via pydantic-settings. Env vars prefixed `S4RI_`. Key settings:
- `llm_model`: `mistral:7b-instruct-v0.3-q5_K_M`
- `embedding_model`: `sentence-transformers/all-MiniLM-L6-v2`
- `chroma_collection`: `s4_research`
- `chunk_size`: 1000, `chunk_overlap`: 200

## Observability

`src/observability/tracer.py` — `ResearchTracer` class with Langfuse integration + JSON fallback. The orchestrator creates a tracer per query, wraps the pipeline in a timed span, and logs all graph trace entries as decisions. Langfuse is not configured yet (no public key set) — traces use structured JSON logging.

## Evaluation

`src/evaluation/` contains a full eval suite: hallucination detection, adversarial testing, performance benchmarks, regression tracking, A/B comparison. Run via `s4ri eval --all`. Test data in `data/evaluation/`.

## Frontend

`src/frontend/app.py` — Streamlit app with agent mode (SSE streaming) and RAG mode toggle. Launch via `s4ri ui`.

## Data

- `data/raw/` — 6 source documents + `manifest.json`. Currently ingested: 29 chunks in ChromaDB.
- `data/vectors/` — ChromaDB persistent storage (auto-created on first ingest)
- `data/evaluation/` — Golden queries (41), adversarial sets, injection prompts

## What NOT to do

- Don't add CrewAI, AutoGen, or other orchestration frameworks — LangGraph only
- Don't use external LLM APIs — everything runs local via Ollama
- Don't modify source reliability weights without discussion — they affect all retrieval scoring
- Don't skip `make_trace_entry()` — inline trace dicts with empty timestamps was a bug we fixed
- Don't hardcode system prompts in agent files — use constants from `agent_prompts.py`
- Don't mock the vector store in integration-style tests if you need real retrieval behavior

## Known issues / next steps

- Langfuse not configured (needs `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` env vars)
- Sub-agents run sequentially via LangGraph edges — parallel execution not yet implemented
- Agent endpoint has no conversation memory (multi-turn not supported for agent queries)
- `python -m src.cli` resolves to wrong module — use `python -c "from src.cli import app; app()"` or install with `pip install -e .` and use `s4ri` directly
