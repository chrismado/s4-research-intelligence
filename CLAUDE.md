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

## Evaluation suite — design decisions and review fixes

The eval suite was built in one pass, reviewed, and had 18 issues fixed. This section documents the non-obvious decisions so you don't re-investigate them.

### Architecture

- **No external eval libs** — ragas, deepeval, and trulens were explicitly excluded. All scoring, claim extraction, and statistical testing are implemented from scratch.
- **Paired t-test without scipy** — `comparator.py:_paired_t_test()` and `_t_distribution_p()` implement the full paired t-test. Uses normal approximation for df > 30, continued fraction approximation for df < 30. This is accurate enough for significance testing but not research-grade.
- **Heuristic-based detection** — Hallucination negation detection, adversarial contradiction detection, abstention detection, and injection resistance all use string-matching heuristics (not LLM-as-judge). This is a known limitation — the heuristics work well for the S4 corpus but may need tuning for other domains.
- **Settings wiring** — `config/settings.py` has 6 eval thresholds (`eval_support_threshold`, `eval_contradiction_threshold`, `eval_abstention_confidence`, `eval_pass_threshold`, `eval_regression_warning`, `eval_regression_critical`). These flow into `HallucinationDetector`, `GoldenSetRunner`, and `RegressionTracker` via `suite.py`. If you add new eval modules, wire their thresholds through settings the same way.

### Review fixes applied (commit `56a1964`)

These were bugs found during review and fixed before the commit:

1. **Latency winner logic** (`comparator.py:159`) — Was `winner="rag" if agent_mean > rag_mean else "tie"`, which could never report "agent" as winner. Fixed to proper three-way conditional.
2. **CLI result type mismatch** (`cli.py`) — Individual `--flag` paths returned raw objects, but `--all` returned `.to_dict()` dicts. Dashboard and reporters would get inconsistent types. Fixed: all paths now call `.to_dict()`.
3. **CI workflow invalid flag** (`eval.yml`) — Had `--report json` but `--report` expects a file path. Fixed to `--report eval_report.md`.
4. **Settings not wired** — `eval_support_threshold`, `eval_pass_threshold`, `eval_regression_warning/critical` were declared in settings.py but modules hardcoded their own defaults. Fixed: `suite.py` passes settings values to all constructors.
5. **`run_all()` incomplete** (`suite.py`) — Omitted quantization and A/B comparison. Progress bar said 6 steps but should have been 8. Fixed.
6. **Empty `key_words` false positive** (`contradiction.py:104`) — When `actual_claim` had no words longer than 4 chars, `key_words` was empty, and `0 >= 0.0` evaluated to True (false positive). Fixed with `key_words and` guard.
7. **`_score_sources()` masking failures** (`golden_set.py:254`) — Returned 0.5 (neutral) for both "no expected sources" and "sources expected but retrieval failed." Fixed: returns 1.0 when no verification needed, 0.0 when retrieval fails.
8. **`test_none_response` testing wrong thing** (`test_claim_extractor.py:42`) — Was passing `""` instead of `None`. Fixed and added `str | None` type hint to `extract()`.
9. **Unused contradiction data load** (`generator.py:65`) — `_load_json()` was called but return value discarded. Assigned to `_contra_sets` with comment.
10. **Perplexity delta display** (`quantization.py`, `terminal.py`, `markdown.py`) — Used `f"+{delta:.2f}"` which showed `+-0.05` for negatives. Fixed to `f"{delta:+.2f}"`.
11. **`chr(10).join()`** (`markdown.py:266`) — Worked but was inconsistent. Replaced with `_join_lines()` helper.
12. **API module validation** (`routes.py`) — `/eval/run` silently skipped invalid module names. Added validation with 400 error.
13. **Improved t-distribution p-value** (`comparator.py:219`) — Original `0.5 * x^a` formula was too crude for small df. Replaced with continued fraction approximation.
14. **`scaling.py` missing error handling** — `golden_queries_path.read_text()` had no try-except. Added with graceful fallback.

### What the tests DON'T cover

- No integration tests that hit a real vector store or Ollama — all tests use mocks or test dataclass behavior
- `AdversarialGenerator.run_all()` is not tested end-to-end
- `ABComparator.compare()` is not tested end-to-end (only `_paired_t_test()` is tested)
- ROUGE-L scoring path in `golden_set.py` is not tested (only the keyword fallback)
- Thread safety in `memory.py` peak tracking is not tested
- The `_t_distribution_p()` continued fraction approximation is not tested for accuracy against known p-values

### Module map

```
src/evaluation/
├── suite.py                 # Orchestrator — run_all(), run_golden_set(), run_hallucination(), etc.
├── hallucination/
│   ├── claim_extractor.py   # NLTK sent_tokenize → filter → Claim objects
│   ├── detector.py          # Claims → vector store search → supported/unsupported/contradicted
│   └── scorer.py            # Batch aggregation, history tracking, trend computation
├── adversarial/
│   ├── generator.py         # Orchestrates all 3 attack types, computes weighted aggregate
│   ├── contradiction.py     # Plants false premises, checks if system corrects them
│   ├── unanswerable.py      # Questions with no corpus answer, checks for abstention
│   └── injection.py         # 8 attack types with per-type compliance signal detection
├── benchmarks/
│   ├── latency.py           # Per-component timing with percentiles
│   ├── throughput.py         # ThreadPoolExecutor at multiple concurrency levels
│   ├── memory.py            # pynvml VRAM profiling with background sampling thread
│   ├── quantization.py      # FP16/INT8/INT4 via llama.cpp or Ollama fallback
│   └── scaling.py           # Accuracy at 25/50/75/100% corpus
├── regression/
│   ├── golden_set.py        # Run golden queries, score with ROUGE-L + keyword matching
│   ├── tracker.py           # Save/load history, compare with warning/critical thresholds
│   └── comparator.py        # Paired t-test A/B comparison (RAG vs Agent)
├── report/
│   ├── markdown.py          # Full report with all sections
│   ├── json_export.py       # CI/CD export with exit code logic
│   └── terminal.py          # Rich dashboard with colored tables
└── datasets/
    └── loader.py            # Loads all 5 JSON dataset files from data/evaluation/
```

## Known issues / next steps

- Langfuse not configured (needs `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` env vars)
- Sub-agents run sequentially via LangGraph edges — parallel execution not yet implemented
- Agent endpoint has no conversation memory (multi-turn not supported for agent queries)
- `python -m src.cli` resolves to wrong module — use `python -c "from src.cli import app; app()"` or install with `pip install -e .` and use `s4ri` directly
- `contradiction_sets.json` is loaded by the adversarial generator but not yet used for separate contradiction pair testing (currently uses `adversarial_queries.json` filtered by type)
- API eval results (`_eval_results`, `_eval_status` dicts in routes.py) grow unbounded — no cleanup/expiry implemented
- Hallucination negation detection is XOR-based (one side has negation marker, other doesn't) — will miss semantic contradictions without explicit negation words
