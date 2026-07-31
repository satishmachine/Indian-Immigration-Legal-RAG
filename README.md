# 🏛️ Bureau of Immigration – Statutory AI Assistant

> **Executive-Grade Legal RAG Platform** for Indian Immigration, Citizenship, Passport, Emigration, and Foreigners Laws.  
> *Grounded Legal Question Answering for Immigration Acts, Rules, Orders & Notifications.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)](https://python.langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange)](https://groq.com)
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
- [Streamlit Cloud Deployment Guide](#streamlit-cloud-deployment-guide)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [API Reference](#api-reference)
- [License](#license)

---

## Overview

The **Bureau of Immigration – Statutory AI Assistant** is a production-grade Retrieval-Augmented Generation (RAG) platform purpose-built for legal counsel, immigration officers, researchers, and consultants handling Indian statutory immigration matters.

Unlike generic PDF chatbots, this system:

- Understands the **hierarchical structure** of Indian statutory instruments (Acts, Sections, Rules, Orders)
- Executes **hybrid retrieval** fusing Dense Vector search (Qdrant) with Sparse Keyword search (BM25) via Reciprocal Rank Fusion (RRF)
- Reranks statutory candidates using **cross-encoder reranking** (`BGE-Reranker-Large`)
- Highlights criminal and penal consequences (**Penal Clause Warnings**) under Indian law
- Features an **Executive Split-Screen PDF Viewer** for direct statutory verification

---

## Key Features & UI Capabilities

### 1. 🏛️ Indian National Flag Tri-Color Banner
- Executive dark-mode banner incorporating official **India Flag Saffron (`#FF9933`)**, **Chakra Navy**, and **India Flag Green (`#22C55E`)**.
- Features the official **Bureau of Immigration (Ministry of Home Affairs)** seal logo.

### 2. 📄 Split-Screen Interactive PDF Viewer
- View real statutory PDFs directly alongside the chat interface (`7:5` split layout).
- Includes Base64 PDF iframe embedding, zoom controls (`100%`, `125%`, `150%`), page selector sliders, and highlighted statutory passage fallback readers.

### 3. ⚖️ LexisNexis & Westlaw Style Citation Cards
- Displays official Act titles, section titles, page references, and similarity match score badges.
- Every citation card features a direct **`📄 Open PDF in Viewer`** button to open that exact statutory document in the right panel.

### 4. ⚡ Real-Time Streaming & Action Toolbar
- Real-time word-by-word streaming generator via `st.write_stream()`.
- **Action Toolbar**: **📋 Copy to Clipboard** (JS clipboard write), **📥 Export Answer** (JSON download), and **👍 / 👎 Feedback buttons**.
- **Suggested Follow-up Questions**: Dynamic question pills below answers.
- **Penal Clause Warning Banners**: Highlights criminal/penal statutory consequences in prominent warning cards.

### 5. 📂 Drag & Drop PDF Uploader & Domain Scope
- Drag & drop custom statutory PDFs into `uploads/` with instant vector indexing into Qdrant.
- **Retrieval Domain Scope** dropdown filter targeting specific legal domains (*Citizenship*, *Passport*, *Visa*, *Emigration*, *Foreigners*).
- Live **Vector DB Active / Inactive** health metrics indicator.

---

## Legal Corpus

The core indexed legal instruments include:

| Statutory Act / Document | Enactment Year | Legal Domain |
|---|---|---|
| **The Citizenship Act, 1955** | 1955 | Citizenship & OCI Registration |
| **The Passports Act, 1967** | 1967 | Passport Issuance & Impounding |
| **The Foreigners Act, 1946** | 1946 | Restrictions & Deportation |
| **The Emigration Act, 1983** | 1983 | Emigration & Recruitment Agents |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│              Executive Streamlit UI (Frontend)                 │
│   Tri-Color Banner │ Split PDF Viewer │ Lexis Citation Cards   │
└───────────────────────┬────────────────────────────────────────┘
                        │ RAG Execution
┌───────────────────────▼────────────────────────────────────────┐
│                   Legal RAG Pipeline (LCEL)                    │
│                                                                │
│  1. Query Rewriter & Standalone Query Normalization            │
│  2. Hybrid Retrieval (Qdrant Dense Vector + BM25 Sparse)       │
│  3. Reciprocal Rank Fusion (RRF) & Metadata Filtering          │
│  4. Cross-Encoder Reranker (BGE-Reranker-Large)               │
│  5. Statutory Grounded LLM Generation (Groq Llama 3.3 70B)     │
└───────────────────────┬────────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────────┐
│                   Data & Storage Layer                         │
│   Qdrant Vector Store  │  Statutory PDFs (Data_Set/)           │
└────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **UI Framework**: Streamlit `1.57+`
- **Orchestration**: LangChain LCEL
- **LLM Engine**: Groq Llama 3.3 70B Versatile
- **Vector Database**: Qdrant Vector Store
- **Hybrid Search**: Dense Vector Embeddings + BM25 Sparse Search (RRF Fusion)
- **Reranker Engine**: BGE-Reranker-Large
- **API Framework**: FastAPI `0.115+`
- **Environment & Settings**: Pydantic Settings v2

---

## Project Structure

```
Immigration_RAG/
├── Data_Set/                  # Pre-indexed Statutory Act PDFs
│   ├── Citizenship_Act_1955.pdf
│   ├── Passport_Act_1967.pdf
│   ├── Foreigners_Act_1946.pdf
│   └── Emigration_Act_1983.pdf
├── assets/                    # Branding assets & BOI seal logo
│   └── boi_logo.png
├── src/
│   ├── app/                   # Streamlit UI Components & Theme
│   │   ├── components/
│   │   │   ├── chat_interface.py
│   │   │   ├── citation_card.py
│   │   │   ├── header.py
│   │   │   ├── pdf_viewer.py
│   │   │   └── sidebar.py
│   │   ├── state.py           # Global App State Manager
│   │   └── theme.py           # Executive Dark Theme CSS
│   ├── chains/                # LangChain RAG LCEL Chains & Prompts
│   ├── core/                  # Domain Models, Config & Exceptions
│   ├── ingestion/             # PDF Parsing, Chunking & Ingestion Pipeline
│   ├── retrieval/             # Hybrid Retriever, RRF & BGE Reranker
│   └── services/              # Vector Store & Embedding Services
├── .streamlit/
│   └── secrets.toml.example   # Streamlit Cloud secrets template
├── streamlit_app.py           # Main Streamlit Entry Orchestrator
├── pyproject.toml             # Project dependencies & tool configs
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Local Quick Start

### 1. Clone Repository & Environment Setup
```bash
git clone https://github.com/your-username/Immigration_RAG.git
cd Immigration_RAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```
Key variables required:
```env
GROQ_API_KEY=your_groq_api_key
VOYAGE_API_KEY=your_voyage_or_embedding_api_key
```

### 3. Run the Streamlit Application
```bash
streamlit run streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Streamlit Cloud Deployment Guide

This project is fully optimized for **Streamlit Community Cloud** deployment:

1. **Push to GitHub**: Ensure your code is pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "feat: bureau of immigration legal assistant"
   git push origin main
   ```
2. **Deploy on Streamlit Cloud**:
   - Go to **[share.streamlit.io](https://share.streamlit.io)** and click **New App**.
   - Select your repository, branch (`main`), and set Main file path to `streamlit_app.py`.
3. **Set Secrets**:
   - Go to **App Settings ➔ Secrets** and paste your API keys (see [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example)):
     ```toml
     GROQ_API_KEY = "your_actual_groq_api_key"
     VOYAGE_API_KEY = "your_actual_voyage_api_key"
     OPENAI_API_KEY = "your_actual_openai_api_key"
     QDRANT__LOCATION = ":memory:"
     ```
4. **Launch**: Click **Deploy**. Streamlit Cloud will auto-install dependencies, auto-index the statutory PDFs in `Data_Set/` on first launch, and host your app live!

---

## Configuration & Environment Variables

Key configuration parameters managed by `src/core/config/settings.py`:

| Parameter | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | None | Secret API key for Groq Llama 3.3 LLM |
| `VOYAGE_API_KEY` | None | Secret API key for Voyage Legal Embeddings |
| `QDRANT__LOCATION` | `qdrant_db` | Path for local embedded Qdrant DB or `:memory:` |
| `LEGAL_RETRIEVAL__TOP_K` | `5` | Final top documents retrieved for RAG context |
| `RERANKER__TOP_K` | `5` | Top reranked statutory passages passed to LLM |

---

## API Reference

The platform also exposes a RESTful FastAPI service for programmatic legal retrieval:

```bash
# Start FastAPI backend
uvicorn src.api.main:app --reload --port 8000
```
- **Swagger Docs**: `http://localhost:8000/docs`
- **POST `/api/v1/query`**: Execute RAG legal question answering
- **POST `/api/v1/ingest`**: Trigger document ingestion pipeline
- **GET `/api/v1/health`**: Check system & vector store health status

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.
