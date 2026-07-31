"""
scripts/build_all_docs.py (Part 2)
===================================
Appends the complete logic to build:
1. docs/01_Project_Overview.md through docs/15_Developer_Guide.md
2. PROJECT_DOCUMENTATION.md (Comprehensive single file)
3. PROJECT_DOCUMENTATION.pdf (ReportLab canvas with custom header/footer)
"""

import json, os, sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

manifest_path = ROOT_DIR / "scratch" / "full_code_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

# Sort manifest alphabetically by path
manifest = sorted(manifest, key=lambda x: x["path"])

# Helper to format docstring
def fmt_doc(doc):
    if not doc:
        return "No explicit docstring provided."
    return doc.strip().replace("\n", " ")

# -----------------------------------------------------------------------------
# 02_Architecture.md
# -----------------------------------------------------------------------------
doc_02 = """# Architecture & System Design — Indian Legal RAG Platform

## 1. High-Level Architecture Overview

The system follows a clean **Layered Architecture** adhering to **Domain-Driven Design (DDD)** and the **Repository Pattern**. It strictly separates UI representation, API controllers, RAG orchestration, retrieval strategies, vector storage, and configuration management.

```mermaid
graph TB
    subgraph Presentation Layer
        UI[Streamlit Glassmorphic Frontend]
        API[FastAPI v1 REST Engine]
    end

    subgraph Orchestration & Chain Layer
        Chain[LegalRAGChain LCEL]
        HistRet[History Aware Legal Retriever]
        Mem[InMemoryChatMessageHistory]
        Parser[LegalRAGOutputParser]
    end

    subgraph Retrieval & Reranking Layer
        CompRet[LegalCompressionRetriever]
        Hybrid[LegalHybridRetriever]
        Dense[DenseRetriever]
        Sparse[BM25Retriever]
        Reranker[BGEReranker / CrossEncoder]
    end

    subgraph Service & Infrastructure Layer
        EmbServ[BGEEmbeddingService]
        QdrantRepo[QdrantVectorRepository]
        LLMFact[LLM Factory Groq/OpenAI/Gemini]
    end

    subgraph Data & Persistence Layer
        QdrantDB[(Qdrant Vector Store)]
        PDFCorpus[(Data_Set Statutory PDFs)]
        DiskCache[(DiskCache / Redis)]
    end

    UI --> Chain
    API --> Chain
    Chain --> HistRet
    Chain --> Mem
    HistRet --> CompRet
    CompRet --> Hybrid
    Hybrid --> Dense
    Hybrid --> Sparse
    CompRet --> Reranker
    Dense --> EmbServ
    Dense --> QdrantRepo
    QdrantRepo --> QdrantDB
    Chain --> LLMFact
    Chain --> Parser
```

---

## 2. Package Dependency Graph

```mermaid
graph LR
    app --> chains
    app --> core
    api --> chains
    api --> core
    chains --> retrieval
    chains --> services
    chains --> core
    retrieval --> services
    retrieval --> core
    ingestion --> services
    ingestion --> core
    services --> core
    evaluation --> chains
    evaluation --> core
```

---

## 3. Core Architectural Principles
1. **Repository Pattern for Vector Storage**: Abstract interface `VectorStoreRepository` decouples the core domain from Qdrant implementation details.
2. **Factory Pattern for LLMs & Embeddings**: `get_llm()` and `get_embedding_service()` instantiate model providers dynamically based on Pydantic environment configurations.
3. **LangChain Expression Language (LCEL)**: RAG pipeline components are composed as immutable `Runnable` steps supporting async execution and memory tracking.
4. **Single Source of Truth Configuration**: `Settings` class using Pydantic v2 loads and validates all environment variables and secrets at application startup.
5. **Zero-Hallucination Legal Citation Verification**: Extracted citations are cross-checked against retrieved statutory chunks before presentation.
"""

with open(DOCS_DIR / "02_Architecture.md", "w", encoding="utf-8") as f:
    f.write(doc_02)

print("Created 02_Architecture.md")

# -----------------------------------------------------------------------------
# 03_Execution_Flow.md
# -----------------------------------------------------------------------------
doc_03 = """# Execution Flow & Call Hierarchy

## 1. Application Startup Flow (`streamlit run streamlit_app.py`)

When the user launches the application, the execution proceeds sequentially:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as streamlit_app.py
    participant Settings as core.config.settings
    participant State as app.state.AppState
    participant Theme as app.theme
    participant Repo as services.vector_store
    participant Chain as chains.legal_rag_chain
    participant UI as app.components

    User->>App: Launch Application
    App->>Settings: get_settings()
    Settings-->>App: Validated Settings Singleton
    App->>State: AppState.initialize()
    State-->>App: Session State Ready
    App->>Theme: apply_custom_theme()
    Theme-->>App: Custom Glassmorphic CSS Applied
    App->>Repo: ensure_data_set_indexed()
    Repo-->>App: Vector Index Verified
    App->>UI: render_sidebar()
    UI-->>App: Active Filters Selected
    App->>Chain: get_rag_chain()
    Chain-->>App: LegalRAGChain Singleton Initialized
    App->>UI: render_chat_interface()
    UI-->>User: Interactive Chat Screen Rendered
```

---

## 2. Query Execution Call Flow

When a user submits a legal question in the Chat UI:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Chat as render_chat_interface
    participant Chain as LegalRAGChain
    participant HistRet as HistoryAwareRetriever
    participant CompRet as LegalCompressionRetriever
    participant Hybrid as LegalHybridRetriever
    participant Qdrant as QdrantVectorRepository
    participant BGE as BGEEmbeddingService
    participant Reranker as BGEReranker
    participant LLM as Groq / OpenAI LLM
    participant Parser as LegalRAGOutputParser

    User->>Chat: Submit Question ("What is penalty under Section 10 Emigration Act?")
    Chat->>Chain: query(question, session_id, filters)
    Chain->>HistRet: invoke({"input": question, "chat_history": history})
    HistRet->>CompRet: get_relevant_documents(reformulated_question)
    CompRet->>Hybrid: _get_relevant_documents()
    Hybrid->>BGE: embed_query(question)
    BGE-->>Hybrid: 1024-dim Vector
    Hybrid->>Qdrant: search_dense(vector, top_k=20)
    Qdrant-->>Hybrid: Top-20 Dense Chunks
    Hybrid->>Hybrid: BM25 Sparse Search (top_k=20)
    Hybrid->>Hybrid: Convex Fusion Score Calculation
    Hybrid-->>CompRet: Top Candidate Chunks
    CompRet->>Reranker: rerank(query, candidates, top_n=5)
    Reranker-->>CompRet: Top-5 Cross-Attention Scored Chunks
    CompRet-->>HistRet: Compressed LCDocuments
    HistRet-->>Chain: Grounded Context Documents
    Chain->>LLM: invoke(LEGAL_QA_PROMPT)
    LLM-->>Chain: Raw LLM Generation
    Chain->>Parser: parse(question, answer_text, docs)
    Parser-->>Chain: LegalRAGResponse (Citations, Sections, Grounded Answer)
    Chain-->>Chat: Render Response with Expandable Citations
    Chat-->>User: Visual Output Displayed
```
"""

with open(DOCS_DIR / "03_Execution_Flow.md", "w", encoding="utf-8") as f:
    f.write(doc_03)

print("Created 03_Execution_Flow.md")

# -----------------------------------------------------------------------------
# 04_Folder_Structure.md
# -----------------------------------------------------------------------------
doc_04 = """# Complete Folder Structure & Module Directory

## 1. Directory Tree Overview

```
Immigration_RAG/
├── .env                              # Active environment variable configurations
├── .env.example                      # Comprehensive environment template
├── Makefile                          # Developer command automation targets
├── README.md                         # Project introduction and quickstart
├── pyproject.toml                    # Poetry/Ruff dependency & tool specs
├── requirements.txt                  # Production Python dependencies
├── streamlit_app.py                  # Root Streamlit UI entrypoint
├── Data_Set/                         # Raw Statutory Legal PDF Corpus
│   ├── The Citizenship Act, 1955.pdf
│   ├── The Emigration Act, 1983.pdf
│   ├── The Foreigners Act, 1946.pdf
│   ├── The Passport (Entry into India) Act, 1920.pdf
│   └── The Passports Act, 1967.pdf
├── configs/                          # YAML / JSON configuration files
├── docs/                             # Full technical documentation package
├── evaluation/                       # RAGAS metrics & synthetic datasets
├── logs/                             # Rotating application log files
├── processed/                        # Post-ingestion extracted text/JSON
├── qdrant_db/                        # Local Qdrant vector database storage
├── scripts/                          # Diagnostic & testing scripts
├── src/                              # Main Application Source Code
│   ├── api/                          # FastAPI REST Web Server
│   │   ├── main.py                   # FastAPI Application Factory
│   │   ├── dependencies/             # Dependency injection routines
│   │   ├── middleware/             # CORS & Error handling middleware
│   │   └── v1/                       # API Version 1 Routers & Endpoints
│   ├── app/                          # Streamlit Frontend Modules
│   │   ├── state.py                  # Session State Manager
│   │   ├── theme.py                  # Glassmorphic CSS Custom Theme
│   │   └── components/               # Modular UI Widgets (Chat, Sidebar, PDF)
│   ├── chains/                       # LangChain LCEL RAG Pipelines
│   │   ├── citation_formatter.py     # Prompt Document Formatter
│   │   ├── history_aware_retriever.py# Conversational Query Reformulator
│   │   ├── legal_rag_chain.py        # Master RAG Chain Assembly
│   │   ├── memory.py                 # Chat History Session Store
│   │   ├── output_parser.py          # Grounded Citation Parser
│   │   └── prompts.py                # Statutory QA System Prompts
│   ├── core/                         # Enterprise Infrastructure Core
│   │   ├── citation/                 # Statutory Citation Extractor
│   │   ├── config/                   # Pydantic v2 Settings Singleton
│   │   ├── exceptions/               # Domain Exception Hierarchy
│   │   ├── interfaces/               # Abstract Base Repository Classes
│   │   ├── logging/                  # Structlog Logger Factory
│   │   └── models/                   # Pydantic Domain Entities & DTOs
│   ├── evaluation/                   # RAGAS Pipeline Evaluator
│   ├── ingestion/                    # Document Processing Engine
│   │   ├── chunkers/                 # Custom Statutory Legal Chunker
│   │   ├── metadata/                 # Regex Legal Metadata Extractor
│   │   ├── parsers/                  # PDF, DOCX, TXT Document Parsers
│   │   └── pipeline/                 # End-to-End Ingestion Pipeline
│   ├── retrieval/                    # Search & Reranking Architecture
│   │   ├── compression_retriever.py  # LangChain BaseRetriever Wrapper
│   │   ├── dense/                    # Qdrant Dense Similarity Search
│   │   ├── hybrid/                   # BM25 + Dense Score Fusion
│   │   ├── reranker/                 # BGE & FlashRank Cross-Encoders
│   │   └── sparse/                   # BM25 Keyword Search
│   └── services/                     # Business Logic Services
│       ├── chat/                     # LLM Provider Factory (Groq/OpenAI)
│       ├── embedding/                # BGE & SentenceTransformer Services
│       └── vector_store/             # Qdrant Repository Implementation
└── tests/                            # Pytest Unit & Integration Suite
```

## 2. Directory Responsibilities & Interactions

| Directory | Primary Purpose | Key Interactions |
| :--- | :--- | :--- |
| `src/core/` | Single source of truth for config, logging, interfaces, and domain models. | Imported by ALL other packages. Has ZERO dependencies on other `src/` modules. |
| `src/services/` | Concrete drivers for vector DBs, embedding models, and LLMs. | Implements interfaces from `src/core/interfaces/`. Used by `retrieval/` and `chains/`. |
| `src/ingestion/` | Parses PDFs from `Data_Set/`, extracts statutory metadata, chunks text, and embeds vectors into Qdrant. | Uses `services/embedding/` and `services/vector_store/`. |
| `src/retrieval/` | Executes hybrid search and cross-encoder reranking. | Uses `services/vector_store/` and `services/embedding/`. Used by `chains/`. |
| `src/chains/` | Assembles LangChain LCEL pipeline for conversational statutory Q&A. | Uses `retrieval/`, `services/chat/`, `core/models/`. |
| `src/app/` | Renders Streamlit Glassmorphic web UI. | Calls `chains/legal_rag_chain.py` and `ingestion/pipeline/`. |
| `src/api/` | Exposes REST HTTP endpoints for enterprise integrations. | Wraps `chains/` and `ingestion/`. |
"""

with open(DOCS_DIR / "04_Folder_Structure.md", "w", encoding="utf-8") as f:
    f.write(doc_04)

print("Created 04_Folder_Structure.md")
