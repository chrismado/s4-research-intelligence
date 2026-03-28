# Claude Code Handoff — S4 Research Intelligence

## Who You're Building This For

Chris Matteau — Montreal-based ML engineer, self-taught, 7 years production experience. Built ML pipelines for a feature documentary called "S4: The Bob Lazar Story" (environment generation via U-Net + ControlNet conditioning, de-aging via StyleGAN2-ADA, neural matting, Gaussian Splatting). He's applying for a **Generative AI Specialist** role at **CGI Montreal** that requires RAG, LLMs, vector databases, prompt engineering, and Python. This project bridges his existing CV/generative ML work with the NLP/LLM skills the role demands.

## The Story This Project Tells

"I built the visual AI pipelines for S4 — environment generation, de-aging, neural compositing. But I also had thousands of pages of research material for the documentary: interview transcripts, FOIA documents, archival references, scientific papers. So I built a RAG system to query my own research corpus with source-weighted retrieval, contradiction detection, and timeline extraction. Both the visual AI and the knowledge AI serve the same film."

This is NOT a tutorial clone. It's a production tool grounded in real documentary research.

## Current Project Structure

See `CLAUDE.md` for the authoritative module map and architecture details. The README has the full directory tree.

## What's Been Built (Completed)

All original phases are done. Here's what exists:

- **Full RAG pipeline** — ingestion, chunking, embedding, source-weighted retrieval, hybrid search (BM25 + semantic)
- **Multi-agent orchestration** — LangGraph StateGraph with 4 agents (corpus_search, cross_reference, timeline, fact_check), conditional routing, retry logic
- **Streamlit frontend** — chat UI with agent/RAG mode toggle, SSE streaming
- **Evaluation framework** — 6 modules (hallucination, adversarial, benchmarks, regression, quantization, A/B), 64 dedicated tests
- **Static demo** — GitHub Pages site (`docs/`) with pre-recorded agent traces, zero backend required
- **Observability** — Langfuse/JSON tracing, agent metrics
- **Docker** — compose files for full stack and demo mode
- **CI/CD** — GitHub Actions for lint+test (ci.yml), eval suite (eval.yml), Pages deployment (pages.yml)
- **129 tests passing**, ruff clean

## What Could Be Done Next

These are stretch goals, not blockers:

1. **Record real traces** — run `scripts/record_demo_traces.py` on a GPU box with Ollama to replace hand-crafted demo JSON with actual pipeline output
2. **Parallel agent execution** — sub-agents currently run sequentially via LangGraph edges; cross_reference + timeline could run in parallel
3. **Agent conversation memory** — multi-turn not supported for agent queries (RAG mode has it)
4. **AWS/cloud deployment** — Terraform or CDK for EC2 + S3 deployment (CGI wants cloud experience)
5. **Langfuse integration** — tracer is wired but needs `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` env vars
6. **Demo GIF/screenshot** — add a visual to the README showing the Streamlit or static demo

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
