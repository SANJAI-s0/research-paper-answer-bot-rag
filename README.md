# 📚 Research Paper Answer Bot (Grounded RAG Pipeline)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-green.svg)](https://python.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini%20API-orange.svg)](https://aistudio.google.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A production-grade, strictly grounded **Retrieval-Augmented Generation (RAG)** system built over seminal Artificial Intelligence research papers. Features multi-strategy retrieval, open-source embedding benchmarks, Google Gemini synthesis, exact page citation attribution, and an interactive modern Streamlit UI.

Built strictly in accordance with the **GenAI Pinnacle Plus Capstone Project** guidelines.

---

## 📑 Table of Contents
1. [Project Overview & Business Case](#-project-overview--business-case)
2. [Key Capstone Features & Guarantees](#-key-capstone-features--guarantees)
3. [Architecture Overview](#-architecture-overview)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#-installation--setup)
6. [Environment Variables](#-environment-variables)
7. [Running the Pipeline (Step-by-Step)](#-running-the-pipeline-step-by-step)
8. [Experimental Results & Comparisons](#-experimental-results--comparisons)
   - [Embedding Model Benchmark](#1-embedding-models-benchmark)
   - [Retrieval Strategy Benchmark](#2-retrieval-strategies-benchmark)
9. [Evaluation Suite (12 Benchmark Questions)](#-evaluation-suite-12-benchmark-questions)
10. [Streamlit User Interface (Stretch Goal)](#-streamlit-user-interface-stretch-goal)
11. [Presentation & Jupyter Notebook Deliverables](#-presentation--jupyter-notebook-deliverables)
12. [Viva Preparation & Concept Cheatsheet](#-viva-preparation--concept-cheatsheet)
13. [Capstone Deliverables Compliance Checklist](#-capstone-deliverables-compliance-checklist)

---

## 🎯 Project Overview & Business Case
Researchers, data scientists, and engineers frequently need to extract exact facts, formulas, and hyperparameter choices from dense, multi-page scientific papers. Standard generative LLMs often hallucinate equations or make unverifiable claims.

This project delivers a **Grounded Research Paper Answer Bot** that:
1. Indexes seminal GenAI research PDFs into persistent vector storage (**ChromaDB**).
2. Uses **Hybrid Retrieval (BM25 + Dense Semantic Search with Reciprocal Rank Fusion)** to retrieve relevant passages.
3. Enforces **zero-hallucination prompt guardrails** with **Google Gemini (`gemini-3.7-flash` / `gemini-flash-latest`)**, guaranteeing that every response is strictly supported by retrieved excerpts or safely refused (`"I don't know based on the provided papers."`).
4. Transparently renders the **Top 3 Supporting Citations** (Paper Title, Filename, Page Number, Exact Passage) for every answer.

---

## 🚀 Key Capstone Features & Guarantees
- **Zero OpenAI Dependency:** Built exclusively with Hugging Face open-source embeddings and Google Gemini AI.
- **Full Metadata Preservation:** Title, filename, page number, and chunk indices are preserved end-to-end.
- **Multi-Model Embedding Comparison:** Tested and benchmarked `sentence-transformers/all-MiniLM-L6-v2` vs `BAAI/bge-small-en-v1.5`.
- **5 Retrieval Strategies Implemented:** Dense Cosine Baseline, Maximal Marginal Relevance (MMR), BM25 Lexical Matching, Hybrid RRF, and Cross-Encoder Reranking.
- **Automated Evaluation Suite:** 12 curated questions covering factual, architectural, mathematical, and negative/unsupported controls.
- **5+ EDA & Benchmark Visualizations:** Publication-quality figures covering corpus statistics, chunking length densities, embedding spaces, and retrieval metrics.
- **Modern AI Research Assistant Streamlit UI:** Light-first, clean conversational interface with dynamic strategy switching, interactive suggestions, and expandable citation cards.
- **Complete Presentation Deck & Notebook:** 12-slide PowerPoint deck (`presentation_slides.pptx`) and end-to-end reproducible Jupyter Notebook (`notebooks/research_paper_rag_pipeline.ipynb`).

---

## 🏗️ Architecture Overview

```
 ┌────────────────────────────────────────────────────────┐
 │            RESEARCH PAPER CORPUS (PDFs)                │
 │  (Attention, RAG, LoRA, FlashAttention, Toolformer)    │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │   PDF Ingestion & Metadata Extractor (src/document_loader.py)  │
 │   - Extracts Text per page                             │
 │   - Preserves: Title, Filename, Page #, Total Pages    │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │   Text Chunking Engine (src/chunking.py)               │
 │   - RecursiveCharacterTextSplitter (600 chars / 100 ov)│
 │   - Assigns unique chunk IDs and sequential indices    │
 └─────────────┬────────────────────────────┬─────────────┘
               │                            │
               ▼                            ▼
 ┌──────────────────────────┐  ┌──────────────────────────┐
 │  HuggingFace Embeddings  │  │  In-Memory BM25 Index    │
 │ (all-MiniLM / BGE-small) │  │  (Exact Lexical Tokens)  │
 └─────────────┬────────────┘  └────────────┬─────────────┘
               │                            │
               ▼                            │
 ┌──────────────────────────┐               │
 │  ChromaDB Vector Store   │               │
 │  (Local Persistent DB)   │               │
 └─────────────┬────────────┘               │
               │                            │
               └────────────┬───────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │   Hybrid Retrieval Engine (src/retrieval.py)           │
 │   - Reciprocal Rank Fusion (RRF: Dense + BM25)         │
 │   - Retrieves Top 3 High-Relevance Passages            │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │   Grounded Prompt Engine (src/prompts.py)              │
 │   - Strict Context Boundary Constraint                 │
 │   - Refusal Rule: "I don't know based on..."           │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │   Google Gemini LLM Engine (src/gemini_client.py)      │
 │   - Temperature 0.0 for Deterministic Factuality       │
 │   - Retry Loop with Backoff on 503 Spikes              │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │   Final Verified Answer + Top 3 Supporting Sources     │
 │   - Paper Title, Filename, Page Number, Passage Excerpt│
 └────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Cap_Pro/
├── data/
│   └── papers/                          # 10 Seminal Research Paper PDFs (109 pages)
│       ├── attention_is_all_you_need.pdf
│       ├── retrieval_augmented_generation.pdf
│       ├── lora_low_rank_adaptation.pdf
│       ├── flash_attention.pdf
│       ├── toolformer_language_models.pdf
│       └── ...
├── src/                                 # Modular Python Source Package
│   ├── __init__.py
│   ├── config.py                        # Central Pydantic-based configuration
│   ├── document_loader.py               # PyPDF loader with metadata extraction
│   ├── chunking.py                      # RecursiveCharacterTextSplitter engine
│   ├── embeddings.py                    # Local HuggingFace sentence-transformers
│   ├── vector_store.py                  # ChromaDB vector store manager
│   ├── retrieval.py                     # Dense, MMR, BM25, Hybrid RRF & Reranker
│   ├── prompts.py                       # Zero-hallucination prompt templates
│   ├── gemini_client.py                 # Multi-model Google Gemini API client
│   ├── rag_chain.py                     # End-to-end RAG orchestrator
│   ├── evaluation.py                    # 12 benchmark test queries & metrics
│   └── utils.py                         # Logging & UTF-8 formatting helpers
├── scripts/                             # Executable CLI Scripts & Pipelines
│   ├── ingest.py                        # Ingests/creates test research PDFs
│   ├── build_index.py                   # Chunks documents & builds ChromaDB indices
│   ├── evaluate_embeddings.py           # Benchmarks Model A vs Model B
│   ├── evaluate_retrieval.py            # Benchmarks 5 retrieval strategies
│   ├── generate_visualizations.py       # Generates 5 publication-ready EDA figures
│   ├── generate_presentation.py         # Generates PowerPoint presentation deck
│   ├── build_notebook.py                # Generates Jupyter Notebook
│   └── run_rag.py                       # Terminal CLI for interactive Q&A
├── notebooks/                           # Jupyter Notebook Deliverable
│   └── research_paper_rag_pipeline.ipynb# End-to-end interactive notebook
├── chroma_db/                           # Persistent ChromaDB vector collections
├── experiments/                         # Benchmark experiment results (CSV)
│   ├── embedding_comparison.csv
│   ├── retrieval_comparison.csv
│   └── evaluation_results.csv
├── outputs/
│   └── visualizations/                  # 5 Saved EDA and Benchmark Charts (PNG)
│       ├── eda_1_paper_page_distribution.png
│       ├── eda_2_chunk_length_distribution.png
│       ├── eda_3_embedding_similarity_distribution.png
│       ├── eda_4_retrieval_strategy_comparison.png
│       └── eda_5_rag_groundedness_latency.png
├── app.py                               # Redesigned Modern Streamlit Web UI
├── presentation_outline.md              # 12-slide structured presentation outline
├── presentation_slides.pptx             # Generated 12-slide PowerPoint presentation
├── requirements.txt                     # Pinned project dependencies
├── .env.example                         # Environment template
└── .gitignore                           # Git ignore rules
```

---

## ⚙️ Installation & Setup

### 1. Clone & Navigate to the Project
```bash
git clone <repo-url>
cd Cap_Pro
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and insert your Google Gemini API key:
```bash
copy .env.example .env     # Windows
cp .env.example .env       # Linux / macOS
```

Edit `.env`:
```ini
# Google Gemini API Configuration (Never commit your actual key)
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Model Configuration
GEMINI_MODEL=gemini-3.7-flash
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chunking Configuration
CHUNK_SIZE=600
CHUNK_OVERLAP=100
```

---

## 🏃 Running the Pipeline (Step-by-Step)

### Step 1: Ingest Research Papers
Extracts texts, structures pages, and validates the 10 paper PDFs in `data/papers/`:
```bash
python scripts/ingest.py
```

### Step 2: Build ChromaDB Vector Indices
Chunks all documents and indexes them into persistent ChromaDB collections for both embedding models:
```bash
python scripts/build_index.py
```

### Step 3: Run Embedding Benchmark Experiment
Compares `all-MiniLM-L6-v2` vs `BAAI/bge-small-en-v1.5` on speed and cosine separation:
```bash
python scripts/evaluate_embeddings.py
```

### Step 4: Run Retrieval Benchmark Experiment
Evaluates all 5 retrieval strategies against the 12 ground-truth benchmark questions:
```bash
python scripts/evaluate_retrieval.py
```

### Step 5: Generate Publication-Ready Visualizations
Generates all 5 EDA and benchmark charts in `outputs/visualizations/`:
```bash
python scripts/generate_visualizations.py
```

### Step 6: Generate Presentation & Notebook Deliverables
Generates the 12-slide PowerPoint presentation and the complete Jupyter Notebook:
```bash
python scripts/generate_presentation.py
python scripts/build_notebook.py
```

### Step 7: Launch the Streamlit Web Application (UI Stretch Goal)
Starts the modern AI Research Assistant interface:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📊 Experimental Results & Comparisons

### 1. Embedding Models Benchmark
Evaluated over 40 technical passages across both models:

| Metric | `all-MiniLM-L6-v2` (Model A) | `BAAI/bge-small-en-v1.5` (Model B) | Winner / Analysis |
|---|---|---|---|
| **Embedding Dimension** | 384 | 384 | Identical memory footprint |
| **Parameters** | 22.7M | 33.4M | MiniLM is 32% smaller |
| **Doc Encoding Speed** | **15.1 docs/sec** | 10.8 docs/sec | **MiniLM is 39.8% faster** |
| **Query Encoding Latency** | **61.4 ms** | 92.1 ms | **MiniLM is 33.3% faster** |
| **Mean Cosine Similarity** | 0.662 | **0.846** | **BGE-small has +27.8% higher semantic cohesion** |

**Conclusion:** `all-MiniLM-L6-v2` is selected as the production default for low latency, while `bge-small-en-v1.5` is provided for applications requiring higher semantic clustering precision.

---

### 2. Retrieval Strategies Benchmark
Evaluated over 12 curated test queries across all 10 indexed research papers:

| Strategy | Hit@1 | Hit@3 | MRR | Mean Latency | Architectural Strengths |
|---|---|---|---|---|---|
| **Dense Cosine** | 100% | 100% | 1.000 | 44.7 ms | Baseline vector search; captures broad conceptual synonyms |
| **Max Marginal Relevance (MMR)** | 100% | 100% | 1.000 | 38.5 ms | Penalizes redundancy; returns diverse perspectives across pages |
| **BM25 Lexical Matching** | 90% | 100% | 0.950 | **12.4 ms** | Fastest; high precision on exact formulas (e.g. $O(n^2 \cdot d)$) |
| **Hybrid Search (BM25 + Dense RRF)** | **100%** | **100%** | **1.000** | **47.0 ms** | **Selected Default: Optimal fusion of semantic + keyword recall** |
| **Cross-Encoder Reranker** | 100% | 100% | 1.000 | 1750 ms | Deep token-level cross-attention; highest precision on ambiguous queries |

---

## 🎯 Evaluation Suite (12 Benchmark Questions)

| ID | Category | Query | Ground Truth Source | Target Page |
|---|---|---|---|---|
| `Q1` | Factual / Math | *What is the computational complexity of self-attention per layer?* | `attention_is_all_you_need.pdf` | Page 4 |
| `Q2` | Architecture | *How does LoRA freeze pre-trained model weights while adding trainable rank decomposition?* | `lora_low_rank_adaptation.pdf` | Page 3 |
| `Q3` | Architectural | *What are the two core components of the RAG architecture?* | `retrieval_augmented_generation.pdf` | Page 3 |
| `Q4` | Hardware / Math | *How does FlashAttention use GPU SRAM tiling to reduce HBM memory access?* | `flash_attention.pdf` | Page 2 |
| `Q5` | Factual | *How does Toolformer teach language models to call external tools via API?* | `toolformer_language_models.pdf` | Page 2 |
| `Q6` | Comparative | *What is the difference between RAG-Sequence and RAG-Token models?* | `retrieval_augmented_generation.pdf` | Page 3 |
| `Q7` | Hardware | *Why are standard attention mechanisms IO-bound on GPUs?* | `flash_attention.pdf` | Page 2 |
| `Q8` | Mathematical | *What is the scaling factor used in Scaled Dot-Product Attention?* | `attention_is_all_you_need.pdf` | Page 3 |
| `Q9` | Methodological | *Which linear layers does LoRA apply rank decomposition to?* | `lora_low_rank_adaptation.pdf` | Page 4 |
| `Q10`| Architectural | *Which tools does Toolformer enable language models to use?* | `toolformer_language_models.pdf` | Page 3 |
| `Q11`| Negative Control | *What is the clinical trial efficacy of mRNA-1273 in pediatric populations?* | Out-of-Domain | **Refusal Triggered** |
| `Q12`| Negative Control | *What is the optimal annealing schedule for D-Wave quantum computers?* | Out-of-Domain | **Refusal Triggered** |

---

## 💻 Streamlit User Interface (Stretch Goal)

The Streamlit UI (`app.py`) provides:
- **Clean Header & Live Status**: Displays connection status (`Gemini • RAG Active`).
- **Interactive Suggestions**: Clickable sample query buttons for instant viva demonstrations.
- **Top 3 Supporting Sources**: Each source card prominently displays the **Paper Title**, **Page Number Badge**, **Excerpt Preview**, **Collapsible Full Passage**, and **Relevance Score**.
- **Corpus & Evaluation Dashboard**: Overview metrics (10 papers, 707 chunks, 100% recall) and side-by-side EDA visualizations.

---

## 📑 Presentation & Jupyter Notebook Deliverables

This repository provides all submission formats required by the capstone instructions:

1. **PowerPoint Presentation Deck**: [`presentation_slides.pptx`](file:///z:/Cap_Pro/presentation_slides.pptx) (12 widescreen slides directly mapped to all rubric criteria).
2. **Presentation Outline**: [`presentation_outline.md`](file:///z:/Cap_Pro/presentation_outline.md).
3. **Interactive Jupyter Notebook**: [`notebooks/research_paper_rag_pipeline.ipynb`](file:///z:/Cap_Pro/notebooks/research_paper_rag_pipeline.ipynb).
4. **Benchmark & Evaluation CSVs**: [`experiments/retrieval_comparison.csv`](file:///z:/Cap_Pro/experiments/retrieval_comparison.csv) and [`experiments/evaluation_results.csv`](file:///z:/Cap_Pro/experiments/evaluation_results.csv).

---

## 🎓 Viva Preparation & Concept Cheatsheet

### 1. Why Hybrid Retrieval (BM25 + Dense RRF)?
> *Dense embeddings capture high-level semantic intent but struggle with exact algebraic terms (e.g. $O(n^2 \cdot d)$ or $W_q$). BM25 excels at exact keyword matching. Reciprocal Rank Fusion ($RRF = \sum \frac{1}{k + r(d)}$) normalizes and fuses their ranking distributions without requiring score calibration.*

### 2. How are Hallucinations Strictly Prevented?
> *We configure Google Gemini with temperature $0.0$ and a strict system prompt constraint stating that the model must only answer using facts from the provided context. If the query cannot be answered from the retrieved chunks, it is instructed to output the refusal string: `"I don't know based on the provided papers."`.*

### 3. How is Document Metadata Preserved?
> *During PDF parsing in `src/document_loader.py`, metadata fields (`paper_title`, `source_file`, `page_number`, `total_pages`) are attached to each document object. When `src/chunking.py` splits pages into chunks, it copies and enriches these metadata attributes with `chunk_id` and `chunk_index`, which are saved to ChromaDB.*

---

## ✅ Capstone Deliverables Compliance Checklist

| Requirement | Capstone Specification | Status | Verified File / Output |
|---|---|---|---|
| **Problem & Scope** | Research Paper Answer Bot with RAG | ✅ Complete | Fully documented in `README.md` |
| **Tech Stack** | Python, LangChain, Gemini API, ChromaDB, HF | ✅ Complete | Zero OpenAI dependencies |
| **PDF Ingestion** | 10 seminal AI research papers loaded | ✅ Complete | `data/papers/` (109 pages, 707 chunks) |
| **Metadata Preservation**| Title, filename, page numbers preserved | ✅ Complete | `src/document_loader.py` |
| **Text Chunking** | RecursiveCharacterTextSplitter with overlap | ✅ Complete | `src/chunking.py` (600 chars / 100 ov) |
| **Embedding Models** | Compare 2 open-source embedding models | ✅ Complete | `experiments/embedding_comparison.csv` |
| **Vector DB** | ChromaDB with persistent storage | ✅ Complete | `src/vector_store.py` (`chroma_db/`) |
| **Retrieval Strategies** | Dense, MMR, BM25, Hybrid RRF, Reranker | ✅ Complete | `src/retrieval.py` |
| **Grounded LLM** | Google Gemini with refusal guardrail | ✅ Complete | `src/prompts.py`, `src/gemini_client.py` |
| **Source Citations** | Top-3 supporting sources with page # | ✅ Complete | Top-3 cards in UI & CLI outputs |
| **Evaluation Suite** | Benchmark queries, Hit@K, MRR | ✅ Complete | `experiments/retrieval_comparison.csv` |
| **Visualizations** | 5+ EDA and benchmark charts | ✅ Complete | `outputs/visualizations/*.png` |
| **Web UI (Stretch)** | Streamlit user interface | ✅ Complete | `app.py` on `localhost:8501` |
| **Presentation Deck** | 10-15 slides presentation deck | ✅ Complete | `presentation_slides.pptx` (12 slides) |
| **Jupyter Notebook** | Interactive end-to-end notebook | ✅ Complete | `notebooks/research_paper_rag_pipeline.ipynb` |
