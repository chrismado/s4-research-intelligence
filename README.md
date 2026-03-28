# S4 Research Intelligence

**RAG-powered documentary research assistant** built for [S4: The Bob Lazar Story](https://www.imdb.com/title/tt0000000/) — a feature-length documentary investigating Bob Lazar's claims about reverse-engineering extraterrestrial technology at a facility known as S4 near Area 51.

This system ingests a heterogeneous research corpus (interview transcripts, FOIA documents, scientific papers, archival references, production notes) and provides conversational querying with **source-weighted retrieval**, **contradiction detection**, and **timeline extraction**.

## Why This Exists

Producing a documentary about one of the most contested stories in modern history means navigating thousands of pages of contradictory source material — government documents that deny what eyewitnesses claim, scientific papers that partially validate predictions made decades earlier, and interview transcripts spanning 35 years where details drift.

I built the visual AI pipelines for S4 (environment generation, de-aging, neural compositing). This is the knowledge AI — a production tool for navigating the research corpus with the same rigor I apply to the visual pipeline.

## Features

**Source-Weighted Retrieval** — Retrieved results are scored by a weighted combination of semantic similarity (65%) and source reliability (35%). Government FOIA documents and scientific papers rank higher than news articles at the same semantic similarity. This produces research-grade results where provenance matters.

**Contradiction Detection** — When sources disagree on facts, dates, or claims, the system flags contradictions with citations from both sides and an analysis of which source carries stronger provenance.

**Timeline Extraction** — Extracts chronological events from retrieved sources with confidence scoring. Only includes dates explicitly stated in sources — never inferred.

**Rich Metadata & Filtering** — Every document carries structured metadata: source type, author, date, subjects, classification, reliability score. Queries can filter by any combination.

**Evaluation Pipeline** — Built-in quality measurement with test queries and ground truth. Tracks citation rate, source recall, contradiction detection accuracy, and answer confidence.

## Architecture

```
Query → Vector Search (ChromaDB) → Source-Weighted Reranking → Context Assembly
  → LLM Generation (Mistral 7B via Ollama) → Structured JSON Response
      → Answer with citations
      → Contradiction alerts
      → Timeline events
      → Confidence score with reasoning trace
```

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Store | ChromaDB (cosine similarity, persistent) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Mistral 7B Instruct (via Ollama) — fully local, no API keys |
| API | FastAPI with async support |
| CLI | Typer + Rich |
| Containerization | Docker Compose (app + Ollama with GPU passthrough) |
| Evaluation | Custom suite — pynvml, rouge-score, NLTK (no ragas/deepeval) |
| Language | Python 3.10+ |

## Quick Start

```bash
# Clone and install
git clone https://github.com/chrismatteau/s4-research-intelligence.git
cd s4-research-intelligence
pip install -e ".[dev]"

# Pull the LLM
ollama pull mistral:7b-instruct-v0.3-q5_K_M

# Copy environment config
cp .env.example .env

# Ingest your research corpus
python -m src.cli ingest --manifest data/raw/manifest.example.json

# Query
python -m src.cli query "What did Bob Lazar claim about Element 115?"

# Start the API server
python -m src.cli serve
# → http://localhost:8000/docs for interactive API docs
```

### Docker

```bash
cd docker
docker compose up -d
# API available at http://localhost:8000
# Ollama available at http://localhost:11434
```

## Usage Examples

### CLI

```bash
# Basic research query
s4ri query "What government documents reference the Nevada Test Site?"

# Filtered by source type
s4ri query "What did Lazar say about propulsion?" --source-type interview_transcript

# More retrieval depth
s4ri query "Timeline of Lazar's public appearances" --top-k 10
```

### API

```bash
# Research query with filters
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the relationship between Element 115 and gravity propulsion?",
    "source_types": ["interview_transcript", "scientific_paper"],
    "include_contradictions": true
  }'

# Upload a new document
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@new_interview.txt" \
  -F "source_type=interview_transcript" \
  -F "title=New Lazar Interview 2025"
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

Production-grade evaluation and benchmarking framework — built from scratch with no external eval libraries (no ragas, no deepeval, no trulens).

### Golden Test Set

41 ground-truth Q&A pairs spanning four query types, all drawn from the S4/Bob Lazar research corpus:

| Type | Count | Example |
|------|-------|---------|
| Factual | 11 | "What element did Lazar claim was used as nuclear fuel in the alien craft?" |
| Timeline | 10 | "When did Lazar first take friends to watch test flights near S4?" |
| Verification | 10 | "Did Lazar's W-2 from the Department of Naval Intelligence list a valid EIN?" |
| Cross-reference | 10 | "Which of Lazar's claims about the layout of S4 are corroborated by other sources?" |

Each case includes expected answer keywords, difficulty level, and query type metadata for per-dimension scoring.

### Hallucination Detection

Three-phase pipeline that operates without an LLM-as-judge:

1. **Claim Extraction** — NLTK sentence tokenization with regex-based filtering (strips hedging, meta-commentary, compound splitting)
2. **Source Matching** — Each claim is searched against the ChromaDB vector store. Cosine similarity above the support threshold (default 0.75) marks a claim as grounded; negation detection flags contradictions
3. **Scoring** — Computes grounding score, hallucination rate, and fabrication rate per response. Batch aggregation with trend tracking over time

### Adversarial Testing

Three attack categories with heuristic-based detection:

| Attack Type | Cases | What It Tests |
|-------------|-------|---------------|
| Contradiction Injection | 7 | Embeds a false premise in the query — does the system correct it? |
| Unanswerable Queries | 8 | Questions with no answer in the corpus — does the system abstain? |
| Prompt Injection | 8 | Instruction override, role hijack, data exfiltration — does the system resist? |

Overall adversarial score is a weighted aggregate: 30% contradiction + 30% abstention + 40% injection resistance.

### Performance Benchmarks

| Benchmark | Metrics | Implementation |
|-----------|---------|----------------|
| Latency | p50/p95/p99, per-component (embedding, retrieval, LLM, TTFT) | `time.perf_counter()` with warmup queries |
| Throughput | QPS at concurrency levels [1, 2, 4, 8], error rate | `ThreadPoolExecutor` |
| VRAM | Per-GPU baseline/peak/delta via pynvml | Background sampling thread during inference |
| KV Cache Quantization | FP16/INT8/INT4 tokens/s, VRAM, TTFT, perplexity delta, max context | llama.cpp `--cache-type-k`/`--cache-type-v` with Ollama fallback |
| Corpus Scaling | Accuracy vs data volume at 25/50/75/100% | Proportional top_k reduction |

KV cache quantization benchmarks are inspired by TurboQuant (March 2026).

### Regression Tracking

- Saves each eval run's metrics to JSON history files
- Compares current run against the most recent previous run
- Configurable thresholds: warning (5% drop) and critical (10% drop)
- Tracks 9 metrics: pass rate, relevance, completeness, source accuracy, confidence calibration, grounding score, contradiction detection, abstention, injection resistance

### A/B Comparison

Paired t-test (implemented from scratch, no scipy) comparing RAG vs Agent pipelines across 5 metrics. Reports statistical significance (p < 0.05) and determines per-metric and overall winner.

### Reports

Three output formats:

- **Markdown** — Full report with tables for every section, written to `docs/eval_report.md`
- **JSON** — Machine-readable export with pass/fail status for CI/CD (`exit code 1` on critical regressions)
- **Terminal Dashboard** — Rich tables with colored status badges, progress bars during eval runs

### CLI

```bash
# Run everything
s4ri eval --all --dashboard

# Individual modules
s4ri eval --hallucination
s4ri eval --adversarial
s4ri eval --benchmark
s4ri eval --regression
s4ri eval --quantization
s4ri eval --compare              # A/B: Agent vs RAG

# Save report
s4ri eval --all --report eval_report.md
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/eval/run` | Launch async eval run (accepts module list) |
| GET | `/eval/status/{id}` | Check run status |
| GET | `/eval/report/{id}` | Get completed results |
| GET | `/eval/history` | List all runs |

### CI/CD

GitHub Actions workflow (`.github/workflows/eval.yml`) runs on every push and PR to main:

1. Lint eval code with ruff
2. Run 62 eval unit tests
3. Run regression checks (if vector store available)

## Project Structure

```
s4-research-intelligence/
├── config/settings.py              # Central configuration
├── src/
│   ├── ingestion/                  # Document loading, chunking, embedding
│   ├── retrieval/                  # RAG pipeline with source-weighted reranking
│   ├── api/                        # FastAPI REST interface
│   ├── models/                     # Pydantic schemas
│   ├── prompts/                    # Engineered prompt templates
│   ├── evaluation/
│   │   ├── evaluator.py            # Original RAG evaluator (preserved)
│   │   ├── suite.py                # Eval suite orchestrator
│   │   ├── hallucination/          # Claim extraction → source matching → scoring
│   │   ├── adversarial/            # Contradiction, unanswerable, injection tests
│   │   ├── benchmarks/             # Latency, throughput, VRAM, quantization, scaling
│   │   ├── regression/             # Golden set runner, tracker, A/B comparator
│   │   ├── report/                 # Markdown, JSON, terminal dashboard
│   │   └── datasets/               # Dataset loader
│   └── cli.py                      # Command-line interface
├── tests/
│   └── test_evaluation/            # 62 unit tests across 7 test files
├── data/
│   ├── raw/                        # Source documents (not committed)
│   └── evaluation/
│       ├── golden_queries.json     # 41 ground-truth Q&A pairs
│       ├── adversarial_queries.json
│       ├── unanswerable_queries.json
│       ├── contradiction_sets.json
│       └── injection_prompts.json
├── .github/workflows/eval.yml      # CI pipeline
├── docker/                         # Containerization
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
