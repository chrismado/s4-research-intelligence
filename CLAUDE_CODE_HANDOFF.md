# Claude Code Handoff — S4 Research Intelligence

## Who You're Building This For

Chris Matteau — Montreal-based ML engineer, self-taught, 7 years production experience. Built ML pipelines for a feature documentary called "S4: The Bob Lazar Story" (environment generation via U-Net + ControlNet conditioning, de-aging via StyleGAN2-ADA, neural matting, Gaussian Splatting). He's applying for a **Generative AI Specialist** role at **CGI Montreal** that requires RAG, LLMs, vector databases, prompt engineering, and Python. This project bridges his existing CV/generative ML work with the NLP/LLM skills the role demands.

## The Story This Project Tells

"I built the visual AI pipelines for S4 — environment generation, de-aging, neural compositing. But I also had thousands of pages of research material for the documentary: interview transcripts, FOIA documents, archival references, scientific papers. So I built a RAG system to query my own research corpus with source-weighted retrieval, contradiction detection, and timeline extraction. Both the visual AI and the knowledge AI serve the same film."

This is NOT a tutorial clone. It's a production tool grounded in real documentary research.

## What's Already Scaffolded

The full project structure is in place with working code for:

```
s4-research-intelligence/
├── config/settings.py          # Pydantic settings with env var overrides
├── src/
│   ├── ingestion/
│   │   ├── loader.py           # Multi-format doc loader with metadata extraction
│   │   ├── chunker.py          # Recursive chunking with metadata propagation
│   │   └── vectorstore.py      # ChromaDB wrapper with HuggingFace embeddings
│   ├── retrieval/
│   │   └── pipeline.py         # Full RAG pipeline with source-weighted reranking
│   ├── api/
│   │   ├── app.py              # FastAPI factory
│   │   └── routes.py           # REST endpoints for research, ingestion, stats
│   ├── models/
│   │   ├── documents.py        # Document/chunk schemas with rich metadata
│   │   └── queries.py          # Query/response models with citations, contradictions, timeline
│   ├── prompts/
│   │   └── templates.py        # Engineered prompts for research, contradiction detection, timeline
│   ├── evaluation/
│   │   └── evaluator.py        # RAG quality evaluation with test queries
│   └── cli.py                  # Typer CLI: ingest, query, serve, stats
├── tests/
│   ├── test_models.py          # Schema validation tests
│   └── test_ingestion.py       # Loader and chunker tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml      # App + Ollama with GPU passthrough
├── data/
│   ├── raw/manifest.example.json    # Example manifest showing metadata structure
│   └── evaluation/test_queries.json # 5 evaluation queries with ground truth
├── pyproject.toml              # Dependencies, scripts, tooling config
├── .env.example
└── .gitignore
```

## What Needs To Be Done

### Phase 1: Get It Running (Priority)

1. **Install dependencies and verify imports**
   - `pip install -e ".[dev]"` — fix any dependency conflicts
   - Ensure all imports resolve correctly (the project uses relative imports from `src/`)
   - You may need to add a `conftest.py` with sys.path setup for pytest

2. **Test with sample data**
   - Chris needs to populate `data/raw/` with his actual S4 research files (transcripts, PDFs, notes)
   - For now, create 2-3 synthetic sample documents so the pipeline can be tested end-to-end:
     - A fake interview transcript (~500 words)
     - A fake government document excerpt (~300 words)
     - A fake production note (~200 words)
   - All should reference Bob Lazar, S4, Area 51, Element 115 so queries have something to retrieve

3. **Verify the full pipeline works end-to-end**
   - `python -m src.cli ingest --manifest data/raw/manifest.example.json`
   - `python -m src.cli query "What did Bob Lazar claim about Element 115?"`
   - `python -m src.cli serve` → hit `http://localhost:8000/docs`
   - Fix any runtime errors in the pipeline

4. **Run tests**
   - `pytest tests/ -v`
   - Fix any failures, add missing test coverage

### Phase 2: Make It Impressive

5. **Add a Streamlit or Gradio frontend** (optional but high-impact for demos)
   - Simple chat interface that sends queries to the FastAPI backend
   - Show sources panel with relevance/reliability scores
   - Timeline visualization for extracted events
   - Contradiction alerts with side-by-side source comparison
   - This makes it demo-able in an interview — "let me show you"

6. **Add conversation memory / multi-turn**
   - Store conversation history so follow-up questions have context
   - "What about his educational claims?" should know we're still talking about Lazar
   - Use LangChain's ConversationBufferMemory or similar

7. **Add the RAGAS evaluation integration**
   - Wire up `src/evaluation/evaluator.py` to use RAGAS metrics properly
   - Generate evaluation reports that can be shown in the README
   - This demonstrates MLOps maturity — you don't just build, you measure

8. **Add hybrid search (semantic + keyword)**
   - ChromaDB supports metadata filtering, but adding BM25 keyword search
     alongside vector search improves retrieval for specific names/dates
   - Use LangChain's EnsembleRetriever to combine both

### Phase 3: Production Polish

9. **Add proper logging and monitoring**
   - Structured JSON logging with loguru
   - Query latency tracking
   - Token usage tracking per query
   - This shows "I think about production systems" not just "I can make it work"

10. **Add async support**
    - The FastAPI routes are async but the pipeline is synchronous
    - Make the Ollama calls async using `httpx` or `ollama`'s async client
    - This matters for the CGI role which mentions production deployment

11. **Write integration tests**
    - Test the full pipeline with a small test corpus
    - Test API endpoints with httpx test client
    - Test that metadata filters actually filter correctly

12. **AWS deployment option**
    - Add Terraform or CDK config for deploying to AWS (EC2 + S3)
    - Or at minimum, document the deployment architecture
    - CGI explicitly wants AWS/Azure/GCP experience

## Key Architecture Decisions (Don't Change These)

- **Source-weighted reranking**: This is the main differentiator. The combined score formula `(relevance * 0.65) + (reliability * 0.35)` means government documents outrank news articles at the same semantic similarity. This is intentional for documentary research where provenance matters.

- **Structured JSON output from LLM**: The prompts force the LLM to return JSON with citations, contradictions, timeline events, and confidence. This is harder to implement than free-text RAG but demonstrates prompt engineering maturity.

- **Manifest-based ingestion**: Documents are ingested via a JSON manifest that carries rich metadata (source type, dates, subjects, classification). This is how a real production system handles heterogeneous source material — not just dumping PDFs into a folder.

- **Evaluation pipeline**: The test_queries.json + evaluator.py setup shows that Chris thinks about quality measurement, not just building features.

## Tech Stack Mapping to CGI Job Requirements

| CGI Requirement | Project Implementation |
|---|---|
| RAG and latest techniques | Full RAG pipeline with source-weighted reranking |
| Vector Databases | ChromaDB with cosine similarity search |
| Open source model deployment (LLaMA, Mistral) | Ollama with Mistral 7B (swappable) |
| Prompt Engineering | 4 engineered prompt templates with structured JSON output |
| Python, SQL | Python throughout, structured data models |
| Azure, AWS, or GCP | Docker Compose + deployment docs for AWS |
| Fine-tune LLMs with LoRA/QLoRA | [STRETCH] Add a fine-tuning script for domain adaptation |
| MLOps — build, tune, deploy, monitor | Evaluation pipeline, logging, Docker deployment |
| 5+ years hands-on Advanced Analytics | This project + his 7 years of production ML |

## What NOT To Do

- Don't make this generic. Every prompt, every model name, every test query should reference S4/Lazar/Area 51. This is a domain-specific tool, not a generic RAG template.
- Don't over-engineer the frontend. A simple Streamlit app is fine. The backend is the star.
- Don't use OpenAI. The whole point is open-source model deployment (Mistral/LLaMA via Ollama). CGI explicitly lists this as a requirement.
- Don't skip the evaluation pipeline. This is what separates a senior engineer from a tutorial follower.

## GitHub Repo Setup

When pushing to GitHub:
- Repo name: `s4-research-intelligence`
- Description: "RAG-powered documentary research assistant with source-weighted retrieval, contradiction detection, and timeline extraction. Built for S4: The Bob Lazar Story."
- Add topics: `rag`, `llm`, `langchain`, `chromadb`, `documentary`, `python`, `fastapi`, `ollama`
- Include a demo GIF or screenshot in the README showing the CLI or Streamlit interface
- License: MIT
