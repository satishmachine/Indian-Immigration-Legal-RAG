# 🏛️ Bureau of Immigration – Statutory AI Assistant

> **Executive-Grade Legal RAG Platform** for Indian Immigration, Citizenship, Passport, Emigration & Foreigners Acts & Rules.  
> *Grounded Legal Question Answering with Interactive Single-Heading PDF Highlighting & Qdrant Cloud Vector Search.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Qdrant Cloud](https://img.shields.io/badge/Qdrant_Cloud-Cluster-red?logo=qdrant)](https://qdrant.tech)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)](https://python.langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
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
- Operates on a **Qdrant Cloud Vector Database** hosting 1,536-dimensional embeddings (295 chunks / 60,017 words indexed)
- Executes **hybrid retrieval** fusing Dense Vector search (Qdrant Cloud) with Sparse Keyword search (BM25) via Reciprocal Rank Fusion (RRF)
- Reranks statutory candidates using **cross-encoder reranking** (`BGE-Reranker-Large`)
- Features an **Interactive Title-Aware PDF Viewer (`pdf_viewer.html`)** that automatically opens statutory citations in new web tabs, navigates to the exact section page, and renders **a single glowing highlight box** over the section title heading
- Automatically resolves **Act vs. Rules PDF routing** (e.g., routing Rule 9 to *Immigration and Foreigners Rules 2025.pdf* Page 25)

---

## Key Features & UI Capabilities

### 1. ☁️ Qdrant Cloud Vector Indexing
- Cloud-hosted Qdrant cluster (`Immigration_RAG`) storing 1536-dimensional dense vector embeddings.
- Automatic document parsing, chunking, and cloud vector upsert pipeline.

### 2. 📄 Title-Aware PDF Viewer (`pdf_viewer.html`)
- Opens cited statutory documents in a separate web tab with clean executive header controls.
- **Multifactor Page Locator**: Locates section pages using title keyword scoring, skipping Hindi pages and Table of Contents index lists.
- **Single-Heading Highlight Overlay**: Ranks candidate matches and draws **exactly ONE** glowing yellow highlight box over the true Section Heading (eliminating unwanted secondary boxes).
- **Hindi Page & Devanagari Filtering**: Automatically skips Hindi pages (Pages 1–22 of Rules 2025) and strips non-English Devanagari text during client-side scanning.

### 3. ⚖️ LexisNexis & Westlaw Style Citation Cards
- Displays statutory Act titles, section titles, page references, and similarity match score badges.
- Every citation card features a direct **`📄 Open Act/Rules PDF (Sec. X) in New Web Tab ↗`** link button.

### 4. ⚡ Real-Time Streaming & Penal Clause Warnings
- Real-time word-by-word streaming response generation.
- **Penal Clause Warning Banners**: Highlights criminal/penal statutory consequences in prominent warning cards.

### 5. 📂 PDF Upload & Automated Cloud Vector Ingestion
- Upload custom statutory PDFs through the sidebar interface.
- Automatically saves the file, computes embeddings, and stores vectors directly in **Qdrant Cloud**.

---

## Legal Corpus

The indexed legal corpus covers **7 core Indian statutory instruments**:

| Statutory Act / Document | Enactment Year | Legal Domain |
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
│              Executive Streamlit UI (Frontend)                 │
│   Tri-Color Banner │ Citation Cards │ Sidebar PDF Uploader    │
└───────────────────────┬────────────────────────────────────────┘
                        │ RAG Execution
┌───────────────────────▼────────────────────────────────────────┐
│                   Legal RAG Pipeline (LCEL)                    │
│                                                                │
│  1. Query Rewriter & Standalone Query Normalization            │
│  2. Hybrid Retrieval (Qdrant Cloud Vector + BM25 Sparse)       │
│  3. Reciprocal Rank Fusion (RRF) & Metadata Filtering          │
│  4. Cross-Encoder Reranker (BGE-Reranker-Large)               │
│  5. Statutory Grounded LLM Generation (OpenAI / Euri / Groq)   │
└───────────────────────┬────────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────────┐
│                   Cloud & Storage Layer                        │
│   Qdrant Cloud Vector DB  │  Title-Aware PDF Viewer HTML       │
└────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **UI Framework**: Streamlit `1.40+`
- **Vector Database**: Qdrant Cloud (`legal_documents` collection, 1536-dim vectors)
- **Embeddings**: OpenAI `text-embedding-ada-002` via Euri API Proxy
- **Orchestration**: LangChain LCEL
- **PDF Engine**: PyMuPDF (`fitz`), PDF.js 3.11
- **API Framework**: FastAPI `0.115+`
- **Configuration**: Pydantic Settings v2

---

## Project Structure

```
Immigration_RAG/
├── Data_Set/                  # Statutory Act PDFs
├── assets/                    # Branding assets & BOI seal logo
├── static/
│   ├── pdf_viewer.html        # Custom Title-Aware PDF Viewer with Highlighting
│   └── pdfs/                  # Web-served PDF copies
├── src/
│   ├── app/                   # Streamlit UI Components & Theme
│   │   ├── components/        # Chat, Header, Citation Cards, Sidebar
│   │   └── state.py           # Global App State Manager
│   ├── chains/                # RAG LCEL Chains & Prompts
│   ├── core/                  # Domain Models, Config & Settings
│   ├── ingestion/             # PDF Parsing, Chunking & Cloud Ingestion Pipeline
│   ├── retrieval/             # Hybrid Retriever (Qdrant Cloud + BM25) & Reranker
│   └── services/              # Vector Store & PDF Locator Services
├── scripts/
│   └── migrate_to_qdrant_cloud.py  # Standalone Qdrant Cloud Migration Runner
├── .streamlit/
│   └── config.toml            # Streamlit theme & static file serving config
├── streamlit_app.py           # Main Streamlit App Entry Point
├── pyproject.toml             # Project dependencies & tool configs
├── requirements.txt           # Production Python dependencies
└── README.md
```

---

## Local Quick Start

### 1. Clone Repository & Environment Setup
```bash
git clone https://github.com/<your-username>/Immigration_RAG.git
cd Immigration_RAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Qdrant Cloud and LLM API credentials:
```bash
cp .env.example .env
```
Key variables required:
```env
OPENAI_API_KEY=your_openai_api_key
EURI_API_KEY=your_euri_api_key
QDRANT_URL=https://4a24b67a-f6d3-4ddb-9bf6-c84eaa803e60.eu-central-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_cloud_api_key
QDRANT_COLLECTION=legal_documents
```

### 3. Run the Application
```bash
streamlit run streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Streamlit Cloud Deployment Guide

This project is fully optimized for **Streamlit Community Cloud** deployment:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Bureau of Immigration Legal AI Assistant"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/Immigration_RAG.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Cloud**:
   - Log into **[share.streamlit.io](https://share.streamlit.io)** and click **New App**.
   - Select your repository (`Immigration_RAG`), branch (`main`), and set Main file path to `streamlit_app.py`.
3. **Configure Secrets**:
   - Go to **App Settings ➔ Secrets** and paste your environment keys:
     ```toml
     OPENAI_API_KEY = "your_openai_api_key"
     EURI_API_KEY = "your_euri_api_key"
     QDRANT_URL = "https://4a24b67a-f6d3-4ddb-9bf6-c84eaa803e60.eu-central-1-0.aws.cloud.qdrant.io"
     QDRANT_API_KEY = "your_qdrant_cloud_api_key"
     QDRANT_COLLECTION = "legal_documents"
     ```
4. **Launch**: Click **Deploy**. Streamlit Cloud will host your application live!

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.
