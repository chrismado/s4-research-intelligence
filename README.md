# S4 Research Intelligence

**Autonomous multi-agent research system** built for [S4: The Bob Lazar Story](https://www.imdb.com/title/tt0000000/) — a feature-length documentary investigating Bob Lazar's claims about reverse-engineering extraterrestrial technology at a facility known as S4 near Area 51.

This system ingests a heterogeneous research corpus (interview transcripts, FOIA documents, scientific papers, archival references, production notes) and provides **multi-agent orchestrated research** with source-weighted retrieval, cross-referencing, timeline extraction, fact-checking, and full decision tracing.

## Why This Exists

Producing a documentary about one of the most contested stories in modern history means navigating thousands of pages of contradictory source material — government documents that deny what eyewitnesses claim, scientific papers that partially validate predictions made decades earlier, and interview transcripts spanning 35 years where details drift.

I built the visual AI pipelines for S4 (environment generation, de-aging, neural compositing). This is the knowledge AI — a production tool for navigating the research corpus with the same rigor I apply to the visual pipeline.

## Architecture

```
                         +-------------------+
                         |   Orchestrator    |
                         |   (LangGraph      |
                         |    StateGraph)     |
                         +--------+----------+
                                  |
                    Plan -> Dispatch -> Execute -> Synthesize -> Evaluate
                                  |                                 |
                                  |                          (retry if low
                                  |                           confidence)
                    +-------------+-------------+
                    |             |             |             |
              +-----+----+ +----+-----+ +-----+----+ +-----+-----+
              |  Corpus  | |  Cross-  | | Timeline | |   Fact-   |
              |  Search  | |Reference | |  Agent   | |  Checker  |
              +-----+----+ +----+-----+ +-----+----+ +-----+-----+
                    |             |             |             |
              +-----+----+ +----+-----+ +-----+----+ +-----+-----+
              | Hybrid   | | 6 Source | |  Date    | | Verdict:  |
              | BM25 +   | |  Types   | | Validate | | VERIFIED  |
              | Semantic  | | Compare  | | + Sort   | | DISPUTED  |
              +----------+ +----------+ +----------+ | CONTRA-   |
                    |                                  | DICTED    |
              +-----+----+                            | UNVERIF-  |
              | ChromaDB |                            | IABLE     |
              | Vector   |                            +-----------+
              | Store    |
              +----------+
```

The orchestrator classifies each query (factual, timeline, verification, exploration), plans which agents to invoke, dispatches them, synthesizes results with citations, and self-evaluates. If confidence is below 0.4, it retries (up to 2x).

## Features

### Multi-Agent Orchestration (LangGraph)

The system dispatches specialized sub-agents based on query type:

- **Corpus Search Agent** — Hybrid search (BM25 + semantic) with source-weighted reranking. Combined score = relevance (65%) + reliability (35%).
- **Cross-Reference Agent** — Searches 6 source types for corroborating/contradicting evidence. Produces a structured report with agreement levels.
- **Timeline Agent** — Extracts chronological events with date validation, flags temporal conflicts across sources.
- **Fact-Check Agent** — Assigns verdicts (VERIFIED / DISPUTED / UNVERIFIABLE / CONTRADICTED) with evidence citations and confidence scores.

Not every query invokes every agent. Simple factual lookups use only corpus search. Complex verification queries dispatch all four.

### Source-Weighted Retrieval

Retrieved results are scored by a weighted combination of semantic similarity (65%) and source reliability (35%). Government FOIA documents and scientific papers rank higher than news articles at the same semantic similarity.

### Observability & Tracing

Every agent query produces a full decision trace:
- Which agents were planned and why
- What each agent found (or didn't find)
- Synthesis reasoning and confidence calibration
- Self-evaluation scores (completeness, citation quality, balance)
- Real UTC timestamps and duration for every step

Traces are stored in-memory and available via the `/research/agent/trace/{trace_id}` endpoint. Langfuse integration is supported when configured; structured JSON logging is the fallback.

### Contradiction Detection

When sources disagree on facts, dates, or claims, the system flags contradictions with citations from both sides and an analysis of which source carries stronger provenance.

### Evaluation Pipeline

Built-in quality measurement with test queries and ground truth. Includes hallucination detection, adversarial testing, performance benchmarks, regression tracking, and A/B comparison (agent vs RAG).

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph StateGraph |
| Vector Store | ChromaDB (cosine similarity, persistent) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Mistral 7B Instruct (via Ollama) — fully local, no API keys |
| Observability | Langfuse + structured JSON fallback |
| API | FastAPI with async support |
| CLI | Typer + Rich |
| Frontend | Streamlit with SSE streaming |
| Evaluation | Custom suite — pynvml, rouge-score, NLTK (no ragas/deepeval) |
| Language | Python 3.10+ |

## Quick Start

```bash
# Clone and install
git clone https://github.com/chrismado/s4-research-intelligence.git
cd s4-research-intelligence
pip install -e ".[dev]"

# Pull the LLM
ollama pull mistral:7b-instruct-v0.3-q5_K_M

# Copy environment config
cp .env.example .env

# Ingest the research corpus
python -c "
from src.ingestion.loader import load_from_manifest
from src.ingestion.chunker import chunk_documents
from src.ingestion.vectorstore import VectorStore
from pathlib import Path

docs = load_from_manifest(Path('data/raw/manifest.json'))
chunks = chunk_documents(docs)
store = VectorStore()
store.add_chunks(chunks)
print(f'Ingested {len(docs)} documents -> {store.count} chunks')
"

# Run a multi-agent research query
python -c "from src.cli import app; app()" agent "What did Bob Lazar claim about Element 115?" --show-trace

# Start the API server
python -c "from src.cli import app; app()" serve
# -> http://localhost:8000/docs for interactive API docs

# Launch the Streamlit frontend
python -c "from src.cli import app; app()" ui
```

## Usage

### CLI

```bash
# Multi-agent research (with decision trace)
s4ri agent "Do government documents corroborate the S4 facility?" --show-trace

# Direct RAG query
s4ri query "What did Lazar say about propulsion?" --source-type interview_transcript

# A/B compare agent vs RAG
s4ri evaluate --compare-agent

# Full evaluation suite
s4ri eval --all --dashboard

# Vector store stats
s4ri stats
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/research/agent` | Multi-agent orchestrated research query |
| GET | `/api/v1/research/agent/trace/{trace_id}` | Full decision trace for a query |
| POST | `/api/v1/research/agent/stream` | SSE streaming of agent execution |
| POST | `/api/v1/research` | Direct RAG query |
| POST | `/api/v1/research/conversation` | Multi-turn RAG with memory |
| POST | `/api/v1/ingest/file` | Upload and ingest a document |
| POST | `/api/v1/ingest/manifest` | Batch ingest from manifest |
| GET | `/api/v1/store/stats` | Vector store statistics |
| POST | `/api/v1/eval/run` | Launch async evaluation run |
| GET | `/api/v1/eval/status/{id}` | Check eval run status |
| GET | `/api/v1/eval/report/{id}` | Get completed eval results |
| GET | `/api/v1/eval/history` | List all eval runs |

### Example: Agent Query

```bash
curl -X POST http://localhost:8000/api/v1/research/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "Is it true that Lazar worked at Los Alamos?"}'
```

Response includes synthesis with citations, fact-check verdicts, corpus results with relevance/reliability scores, and a `trace_id` for full decision trace retrieval.

### Example: Decision Trace

```
+ orchestrator.plan (5869ms)
  {"query_type": "factual", "agents_planned": ["corpus_search", "fact_check"]}
+ orchestrator.dispatch (0ms)
  {"dispatched": ["corpus_search", "fact_check"]}
+ corpus_search.search_and_analyze (3166ms)
  {"result_count": 8}
+ fact_check.verdict (6674ms)
  {"verdict": "DISPUTED", "reason": "conflicting_evidence"}
+ orchestrator.synthesize (4371ms)
  {"confidence": 0.96, "sources_count": 3}
+ orchestrator.evaluate (4215ms)
  {"overall_score": 0.96, "should_retry": false}
```

## Source Reliability Weights

| Source Type | Weight | Rationale |
|---|---|---|
| Government Document | 0.95 | Official records, FOIA releases |
| Scientific Paper | 0.90 | Peer-reviewed research |
| Archival Reference | 0.85 | Museum/institutional records |
| Eyewitness Account | 0.75 | Direct observation, subject to memory |
| Interview Transcript | 0.70 | First-person claims, may contain bias |
| Book Excerpt | 0.65 | Published analysis, editorial filtering |
| News Article | 0.60 | Journalistic reporting, variable accuracy |
| Production Note | 0.50 | Internal working documents |

## Evaluation Suite

Production-grade evaluation and benchmarking framework — built from scratch with no external eval libraries.

### Golden Test Set

41 ground-truth Q&A pairs spanning four query types:

| Type | Count | Example |
|------|-------|---------|
| Factual | 11 | "What element did Lazar claim was used as nuclear fuel?" |
| Timeline | 10 | "When did Lazar first take friends to watch test flights?" |
| Verification | 10 | "Did Lazar's W-2 from Naval Intelligence list a valid EIN?" |
| Cross-reference | 10 | "Which claims about S4's layout are corroborated?" |

### Hallucination Detection

Three-phase pipeline (no LLM-as-judge): claim extraction via NLTK, source matching against ChromaDB, and grounding/fabrication scoring.

### Adversarial Testing

| Attack Type | Cases | What It Tests |
|-------------|-------|---------------|
| Contradiction Injection | 7 | False premise — does the system correct it? |
| Unanswerable Queries | 8 | No answer in corpus — does it abstain? |
| Prompt Injection | 8 | Instruction override, role hijack, data exfil |

### Performance Benchmarks

Latency profiling (p50/p95/p99), throughput (QPS at concurrency 1/2/4/8), VRAM monitoring via pynvml, KV cache quantization (FP16/INT8/INT4), and corpus scaling analysis.

### A/B Comparison

Paired t-test comparing Agent vs RAG pipelines across confidence, source recall, date coverage, and latency.

```bash
s4ri eval --all --dashboard       # Full suite with terminal dashboard
s4ri eval --compare               # A/B: Agent vs RAG
s4ri eval --all --report report.md  # Save markdown report
```

## Project Structure

```
s4-research-intelligence/
├── config/settings.py              # Central configuration
├── src/
│   ├── agents/                     # Multi-agent orchestration layer
│   │   ├── orchestrator.py         # LangGraph StateGraph (plan/dispatch/synth/eval)
│   │   ├── corpus_search.py        # Hybrid search + LLM analysis agent
│   │   ├── cross_reference.py      # Cross-source corroboration agent
│   │   ├── timeline_agent.py       # Chronological event extraction agent
│   │   ├── fact_checker.py         # Verdict assignment agent
│   │   ├── state.py                # ResearchState TypedDict (shared state)
│   │   └── tools.py                # Agent tool wrappers (LLM, search, trace)
│   ├── observability/              # Tracing and metrics
│   │   ├── tracer.py               # Langfuse + JSON fallback tracer
│   │   └── metrics.py              # Agent and pipeline metrics
│   ├── ingestion/                  # Document loading, chunking, embedding
│   ├── retrieval/                  # RAG pipeline with source-weighted reranking
│   ├── api/                        # FastAPI REST interface
│   ├── models/                     # Pydantic schemas
│   ├── prompts/                    # System prompts for all agents
│   │   └── agent_prompts.py        # 8 specialized prompt constants
│   ├── evaluation/                 # Full eval suite
│   │   ├── evaluator.py            # RAG + Agent evaluator with A/B compare
│   │   ├── suite.py                # Eval suite orchestrator
│   │   ├── hallucination/          # Claim extraction -> source matching
│   │   ├── adversarial/            # Contradiction, unanswerable, injection
│   │   ├── benchmarks/             # Latency, throughput, VRAM, quantization
│   │   ├── regression/             # Golden set, tracker, A/B comparator
│   │   └── report/                 # Markdown, JSON, terminal dashboard
│   ├── frontend/app.py             # Streamlit UI with SSE agent streaming
│   └── cli.py                      # Typer CLI (ingest, query, agent, eval, serve, ui)
├── tests/
│   ├── test_agents.py              # 32 agent tests
│   └── test_evaluation/            # 62 eval tests
├── data/
│   ├── raw/                        # Source documents + manifest
│   └── evaluation/                 # Golden queries, adversarial sets
├── .pre-commit-config.yaml         # Ruff + pytest hooks
└── pyproject.toml
```

## Background

This project is part of the production infrastructure for **S4: The Bob Lazar Story**, a feature-length documentary. The visual AI pipelines for the film include:

- **Environment generation** — U-Net with ControlNet-style conditioning trained on Blender CG environments
- **De-aging** — StyleGAN2-ADA with optical flow-based temporal consistency
- **Neural matting** — ML-based green screen removal replacing traditional keying
- **Gaussian Splatting** — Nerfstudio training + gsplat CUDA rendering with depth compositing

This research assistant serves the knowledge side of the same production — navigating the documentary's source material with the same engineering rigor applied to the visual pipeline.

## License

MIT
