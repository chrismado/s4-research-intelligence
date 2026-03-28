# S4 Research Intelligence

**RAG-powered documentary research assistant** built for S4: The Bob Lazar Story — a feature-length documentary investigating Bob Lazar's claims about reverse-engineering extraterrestrial technology at a facility known as S4 near Area 51.

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
| Evaluation | Custom RAGAS-style metrics |
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

## Evaluation

```bash
# Run evaluation against test queries with ground truth
python -m src.evaluation.evaluator --test-set data/evaluation/test_queries.json
```

Metrics tracked: source recall, date coverage, citation rate, contradiction detection accuracy, answer confidence calibration.

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
│   ├── evaluation/                 # Quality measurement pipeline
│   └── cli.py                      # Command-line interface
├── tests/                          # Unit and integration tests
├── docker/                         # Containerization
├── data/
│   ├── raw/                        # Source documents (not committed)
│   └── evaluation/                 # Test queries with ground truth
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
