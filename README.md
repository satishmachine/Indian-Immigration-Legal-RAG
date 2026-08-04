# 🏛️ Bureau of Immigration – Statutory AI Assistant

> **Executive-Grade Legal RAG Platform** for Indian Immigration, Citizenship, Passport, Emigration & Foreigners Acts & Rules.  
> *Grounded Legal Question Answering · Hybrid Vector Search · Interactive PDF Highlighting · Multi-LLM Provider Support.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant_Cloud-Vector_DB-red?logo=qdrant)](https://qdrant.tech)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)](https://python.langchain.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-orange)](https://smith.langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features & UI Capabilities](#key-features--ui-capabilities)
- [Legal Corpus](#legal-corpus)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Quick Start](#local-quick-start)
- [Developer Tooling](#developer-tooling)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [API Reference](#api-reference)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [License](#license)

---

## Overview

The **Bureau of Immigration – Statutory AI Assistant** is a production-grade Retrieval-Augmented Generation (RAG) platform purpose-built for legal counsel, immigration officers, researchers, and consultants handling Indian statutory immigration matters.

Unlike generic PDF chatbots, this system:

- Understands the **hierarchical structure** of Indian statutory instruments (Acts, Sections, Rules, Orders)
- Operates on a **Qdrant Cloud Vector Database** with 1,536-dimensional dense embeddings (295+ chunks indexed)
- Executes **hybrid retrieval** fusing Dense Vector search (Qdrant Cloud) with Sparse Keyword search (BM25) via Reciprocal Rank Fusion (RRF)
- Reranks statutory candidates using a **cross-encoder reranker** (`ms-marco-MiniLM-L-12-v2` by default, configurable)
- Features an **Interactive Title-Aware PDF Viewer** (`pdf_viewer.html`) that opens statutory citations in new tabs, navigates to the exact section page, and renders **a single glowing highlight box** over the section title heading
- Implements an **LLM Provider Factory** (`LLMFactory`) with primary support for **Groq** (`llama-3.3-70b-versatile`) and fallback to **OpenAI / Euri Proxy** & Mock modes
- Supports **multiple embedding providers** (BGE/BAAI, Sentence Transformers, OpenAI, Voyage AI, Jina, Ollama)
- Provides a **FastAPI REST backend** alongside the Streamlit UI for programmatic integration
- Includes optional **LangSmith tracing & observability** for full pipeline visibility

---

## Key Features & UI Capabilities

### 1. ☁️ Qdrant Cloud Vector Indexing
- Cloud-hosted Qdrant cluster storing dense vector embeddings (default: `BAAI/bge-m3`).
- Configurable vector size, collection name, and retrieval thresholds.
- Automatic document parsing, chunking, and cloud vector upsert pipeline.

### 2. 📄 Title-Aware PDF Viewer (`pdf_viewer.html`)
- Opens cited statutory documents in a separate web tab with a clean executive header.
- **Multifactor Page Locator**: Locates section pages via title keyword scoring, skipping Hindi pages and Table of Contents index lists.
- **Single-Heading Highlight Overlay**: Ranks candidate matches and draws **exactly ONE** glowing highlight box over the true section heading — no spurious secondary boxes.
- **Hindi Page & Devanagari Filtering**: Automatically skips Hindi pages and strips non-English Devanagari text during client-side scanning.

### 3. ⚖️ LexisNexis-Style Citation Cards
- Displays statutory Act titles, section titles, page references, and similarity match score badges.
- Every card includes a direct **`📄 Open Act/Rules PDF (Sec. X) in New Tab ↗`** link.

### 4. ⚡ Real-Time Streaming & Penal Clause Warnings
- Word-by-word streaming response generation via LCEL streaming.
- **Penal Clause Warning Banners**: Highlights criminal/penal statutory consequences in prominent warning cards.

### 5. 📂 PDF Upload & Automated Cloud Vector Ingestion
- Upload custom statutory PDFs via the sidebar.
- Auto-saves the file, computes embeddings, and upserts vectors directly into Qdrant Cloud.

### 6. 🤖 Flexible LLM Provider Architecture
- High-speed inference using **Groq** (`llama-3.3-70b-versatile`) with seamless fallback to **OpenAI / Euri Proxy** (`gpt-4o-mini` / `gpt-4.1-mini`) and offline Mock mode via `LLMFactory`.
- Wrapped uniformly behind a shared LangChain `BaseChatModel` interface.

### 7. 🔍 Hybrid Retrieval Pipeline
- **Dense Retrieval**: Semantic similarity via Qdrant Cloud.
- **Sparse Retrieval**: BM25 keyword retrieval.
- **RRF Fusion**: Reciprocal Rank Fusion merges both result lists.
- **Cross-Encoder Reranking**: FlashRank/ms-marco reranker for final candidate selection.
- Configurable `TOP_K` at each stage and a `HYBRID_ALPHA` weight knob.

### 8. 📊 Observability
- Optional **LangSmith tracing** via `LANGCHAIN_TRACING_V2=true` — records every chain step, retrieval call, and LLM response.
- Structured JSON logging via `structlog` with configurable log level and format.
- Disk-based or Redis-backed response caching.

---

## Legal Corpus

The indexed legal corpus covers **7 core Indian statutory instruments**:

| Statutory Act / Document | Year | Domain |
|---|---|---|
| **The Immigration and Foreigners Act, 2025** | 2025 | Immigration, Deportation & Penalties |
| **Immigration and Foreigners Rules 2025** | 2025 | Visa Authorities & Immigration Rules |
| **The Citizenship Act, 1955** | 1955 | Citizenship, Naturalisation & OCI |
| **The Passports Act, 1967** | 1967 | Passport Issuance & Impounding |
| **The Foreigners Act, 1946** | 1946 | Restrictions & Entry Controls |
| **The Emigration Act, 1983** | 1983 | Emigration & Recruitment Agents |
| **Registration of Foreigners Act, 1939** | 1939 | Registration & Arrival Intimations |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│              Streamlit UI  (Frontend · Port 8501)              │
│   Tri-Color Banner │ Citation Cards │ Sidebar PDF Uploader    │
│   Chat Interface   │ Source Panel   │ Title-Aware PDF Viewer  │
└───────────────────────┬────────────────────────────────────────┘
                        │ HTTP / direct Python call
┌───────────────────────▼────────────────────────────────────────┐
│              FastAPI REST API  (Backend · Port 8000)           │
│   POST /api/v1/query  │  POST /api/v1/ingest  │  JWT Auth     │
└───────────────────────┬────────────────────────────────────────┘
                        │ LCEL Chain execution
┌───────────────────────▼────────────────────────────────────────┐
│                    Legal RAG Pipeline (LCEL)                   │
│  1. Standalone Query Rewriter (History-Aware)                  │
│  2. Hybrid Retrieval  (Qdrant Dense + BM25 Sparse → RRF)       │
│  3. Cross-Encoder Reranker  (ms-marco-MiniLM-L-12-v2)          │
│  4. Statutory-Grounded LLM Generation  (multi-provider)        │
│  5. Citation Formatter  +  Penal Clause Detection              │
└───────────────────────┬────────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────────┐
│                  Cloud & Storage Layer                         │
│  Qdrant Cloud Vector DB  │  Disk Cache  │  LangSmith Traces   │
│  BAAI/BGE-M3 Embeddings  │  structlog JSON Logs               │
└────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **UI Framework** | Streamlit `1.40+` |
| **REST API** | FastAPI `0.115+` + Uvicorn |
| **Orchestration** | LangChain LCEL (`langchain`, `langchain-core`, `langchain-community`) |
| **Vector Database** | Qdrant Cloud (`langchain-qdrant`, `qdrant-client`) |
| **Embeddings** | BAAI/BGE-M3 (default) · OpenAI · Voyage · Jina · Ollama |
| **LLM Providers** | OpenAI · Anthropic Claude · Google Gemini · Groq · Ollama |
| **Reranker** | FlashRank / ms-marco-MiniLM-L-12-v2 · Cohere |
| **PDF Engine** | PyMuPDF (`fitz`) · PDF.js 3.11 (browser) · pypdfium2 |
| **Config** | Pydantic Settings v2 |
| **Logging** | `structlog` (console / JSON) |
| **Caching** | `diskcache` (disk) / Redis |
| **Observability** | LangSmith tracing |
| **Code Quality** | Ruff · Black · MyPy · pre-commit |
| **Testing** | Pytest + pytest-cov (≥ 80 % coverage gate) |

---

## Project Structure

```
Immigration_RAG/
├── Data_Set/                        # Statutory Act PDFs (source corpus)
├── assets/                          # Branding assets & BOI seal logo
├── static/
│   ├── pdf_viewer.html              # Title-Aware PDF Viewer with Highlight Overlay
│   └── pdfs/                        # Web-served PDF copies
├── uploads/                         # User-uploaded PDFs (runtime)
├── processed/                       # Post-ingestion processed docs
├── logs/                            # Structured log files
├── reports/                         # Coverage & evaluation reports
├── configs/
│   └── logging.yaml                 # Structured logging configuration
├── src/
│   ├── api/
│   │   ├── main.py                  # FastAPI app factory
│   │   └── v1/                      # Versioned API routes
│   ├── app/
│   │   ├── components/
│   │   │   ├── chat_interface.py    # Streaming chat UI
│   │   │   ├── citation_card.py     # LexisNexis-style citation cards
│   │   │   ├── header.py            # Tri-color banner header
│   │   │   ├── pdf_viewer.py        # PDF viewer iframe bridge
│   │   │   ├── sidebar.py           # PDF upload & settings sidebar
│   │   │   └── source_panel.py      # Retrieved source panel
│   │   ├── state.py                 # Global Streamlit session state
│   │   └── theme.py                 # Dark theme CSS injection
│   ├── chains/
│   │   ├── legal_rag_chain.py       # Main LCEL RAG chain
│   │   ├── history_aware_retriever.py # Standalone query rewriter
│   │   ├── prompts.py               # All system & user prompts
│   │   ├── citation_formatter.py    # Citation extraction & formatting
│   │   ├── memory.py                # Conversation memory management
│   │   └── output_parser.py         # Structured LLM output parsing
│   ├── core/
│   │   ├── config/                  # Pydantic Settings (all env vars)
│   │   ├── models/                  # Domain models & schemas
│   │   ├── interfaces/              # Abstract base classes
│   │   ├── exceptions/              # Custom exception hierarchy
│   │   ├── logging/                 # structlog setup
│   │   ├── citation/                # Citation domain logic
│   │   └── utils/                   # Shared utilities
│   ├── ingestion/
│   │   ├── parsers/                 # PDF & document parsers (PyMuPDF)
│   │   ├── chunkers/                # Statutory-aware text chunkers
│   │   ├── metadata/                # Metadata enrichment
│   │   └── pipeline/                # End-to-end ingestion orchestration
│   ├── retrieval/
│   │   ├── dense/                   # Qdrant dense retriever
│   │   ├── sparse/                  # BM25 sparse retriever
│   │   ├── hybrid/                  # RRF fusion retriever
│   │   ├── reranker/                # Cross-encoder reranker
│   │   └── compression_retriever.py # LangChain compression wrapper
│   └── services/
│       ├── vector_store/            # Qdrant Cloud vector store service
│       ├── embedding/               # Multi-provider embedding service
│       ├── chat/                    # Chat session service
│       └── pdf_service.py           # PDF routing & page locator service
├── scripts/
│   └── migrate_to_qdrant_cloud.py   # Standalone Qdrant Cloud migration runner
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures
│   └── unit/                        # Unit test suite
├── evaluation/                      # RAGAS evaluation datasets & runners
├── .streamlit/
│   └── config.toml                  # Streamlit theme & static file serving
├── .pre-commit-config.yaml          # Pre-commit hooks (Ruff, Black, MyPy)
├── streamlit_app.py                 # Main Streamlit entry point
├── pyproject.toml                   # Project metadata, deps & tool config
├── requirements.txt                 # Production Python dependencies
├── Makefile                         # Developer convenience commands
└── .env.example                     # Environment variables template
```

---

## Local Quick Start

### Prerequisites
- Python 3.12+
- A [Qdrant Cloud](https://cloud.qdrant.io/) account (free tier works)
- At least one LLM API key (OpenAI, Anthropic, Groq, or Google AI Studio)

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/<your-username>/Immigration_RAG.git
cd Immigration_RAG

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

> **Tip (uv):** If you use `uv`, simply run `uv sync` — the `uv.lock` file is committed.

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in the required values:

```env
# --- Active LLM Provider ---
ACTIVE_LLM_PROVIDER=openai          # openai | anthropic | google | groq | ollama

# --- OpenAI ---
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# --- Qdrant Cloud ---
QDRANT_URL=https://<cluster-id>.cloud.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_COLLECTION_NAME=legal_documents

# --- Embedding Provider ---
EMBEDDING_PROVIDER=bge              # bge | sentence_transformers | openai | ollama
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

See [`.env.example`](.env.example) for the full list of all supported variables.

### 3. Ingest the Legal Corpus

Place your statutory PDFs in `Data_Set/` then run the ingestion pipeline:

```bash
python scripts/migrate_to_qdrant_cloud.py
```

### 4. Run the Application

**Streamlit UI only:**
```bash
streamlit run streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

**FastAPI backend only:**
```bash
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs at **`http://localhost:8000/docs`**.

**Both services (two terminals):**
```bash
make run-api    # Terminal 1 – FastAPI on port 8000
make run-app    # Terminal 2 – Streamlit on port 8501
```

---

## Developer Tooling

All common developer commands are available via `make`:

```bash
make help              # Show all available commands

# Installation
make install           # Install production dependencies
make install-dev       # Install all dependencies (incl. dev extras)
make setup-pre-commit  # Install & configure pre-commit hooks

# Code Quality
make lint              # Run Ruff linter
make format            # Auto-format with Black + Ruff
make typecheck         # Run MyPy static type checker

# Testing
make test              # Run the full test suite
make test-unit         # Run unit tests only
make test-integration  # Run integration tests only
make test-cov          # Run tests with HTML coverage report

# Services
make run-api           # Start FastAPI dev server (port 8000)
make run-app           # Start Streamlit frontend (port 8501)
make docker-qdrant     # Start a local Qdrant instance via Docker

# Cleanup
make clean             # Remove __pycache__, build artifacts, reports
```

### Local Qdrant (Docker)

For local development without a Qdrant Cloud account:

```bash
make docker-qdrant
# Qdrant dashboard: http://localhost:6333/dashboard
```

Then in `.env`:
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_URL=                         # leave empty for local
QDRANT_API_KEY=                     # leave empty for local
```

---

## Configuration & Environment Variables

All configuration is managed via **Pydantic Settings v2** (`src/core/config/`). Every variable can be set in `.env` or as a real environment variable. Key groups:

| Group | Key Variables |
|---|---|
| **Application** | `APP_NAME`, `APP_ENV`, `DEBUG`, `APP_VERSION` |
| **LLM** | `ACTIVE_LLM_PROVIDER`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `OLLAMA_BASE_URL` |
| **Qdrant** | `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`, `QDRANT_VECTOR_SIZE` |
| **Embeddings** | `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_DEVICE` |
| **Retrieval** | `RETRIEVAL_TOP_K_DENSE`, `RETRIEVAL_TOP_K_SPARSE`, `RETRIEVAL_TOP_K_RERANK`, `HYBRID_ALPHA` |
| **Reranker** | `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANKER_MAX_LENGTH` |
| **Ingestion** | `INGESTION_CHUNK_SIZE`, `INGESTION_CHUNK_OVERLAP`, `INGESTION_BATCH_SIZE` |
| **API** | `API_HOST`, `API_PORT`, `API_PREFIX`, `API_SECRET_KEY`, `API_CORS_ORIGINS` |
| **Logging** | `LOG_LEVEL`, `LOG_FORMAT`, `LOG_DIR` |
| **Caching** | `CACHE_BACKEND`, `CACHE_TTL_SECONDS`, `REDIS_URL` |
| **LangSmith** | `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` |

See [`.env.example`](.env.example) for descriptions and default values for every variable.

---

## API Reference

The FastAPI backend exposes a versioned REST API. After starting with `make run-api`, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

Key endpoints (under `/api/v1/`):

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Submit a legal query and receive a grounded answer with citations |
| `POST` | `/ingest` | Upload and ingest a PDF into the vector store |
| `GET` | `/health` | Service health check |

All endpoints require a valid JWT bearer token when `API_SECRET_KEY` is configured.

---

## Streamlit Cloud Deployment

This project is optimized for **Streamlit Community Cloud** deployment:

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "feat: initial commit – Bureau of Immigration Legal AI"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/Immigration_RAG.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Log into **[share.streamlit.io](https://share.streamlit.io)** → **New App**
2. Select your repo (`Immigration_RAG`), branch (`main`), main file: `streamlit_app.py`

### 3. Configure Secrets

In **App Settings → Secrets**, paste your environment variables in TOML format:

```toml
ACTIVE_LLM_PROVIDER    = "openai"
OPENAI_API_KEY         = "sk-..."
OPENAI_CHAT_MODEL      = "gpt-4o-mini"

QDRANT_URL             = "https://<cluster-id>.cloud.qdrant.io"
QDRANT_API_KEY         = "<your-qdrant-api-key>"
QDRANT_COLLECTION_NAME = "legal_documents"

EMBEDDING_PROVIDER     = "bge"
EMBEDDING_MODEL_NAME   = "BAAI/bge-m3"
```

### 4. Deploy

Click **Deploy**. Streamlit Cloud will install `requirements.txt` and host your app live.

> **Note:** To use Groq as the LLM provider, set `ACTIVE_LLM_PROVIDER = "groq"` and add `GROQ_API_KEY` in secrets.

---


## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
