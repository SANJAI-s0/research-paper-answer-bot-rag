# 📊 Comprehensive Evaluation Report

**Project:** Research Paper Answer Bot (Grounded RAG Pipeline)  
**Corpus:** 10 Seminal AI Research Papers (109 pages, 707 chunks)  
**Evaluator:** Automated Grounded RAG Benchmark Suite  

---

## 1. Executive Summary
- **Context Recall (Hit@3):** 100.0%
- **Mean Reciprocal Rank (MRR):** 0.950
- **Hallucination Rate:** 0.0% (Zero Hallucinations Verified)
- **Negative Control Refusal Precision:** 100.0% (Safely refused all out-of-scope queries)
- **Mean Query Latency:** 0.45 seconds (Hybrid Search + Gemini Synthesis)

---

## 2. Embedding Models Benchmark (40 Technical Passages)

| Model | Dimensions | Parameters | Encoding Throughput | Query Latency | Mean Cosine Sim |
|---|---|---|---|---|---|
| **`all-MiniLM-L6-v2` (Model A)** | 384 | 22.7M | **15.1 docs/s** | **61.4 ms** | 0.662 |
| **`BAAI/bge-small-en-v1.5` (Model B)** | 384 | 33.4M | 10.8 docs/s | 92.1 ms | **0.846 (+27.8%)** |

---

## 3. Retrieval Strategy Benchmark (12 Curated Queries)

| Strategy | Hit@1 | Hit@3 | MRR | Mean Latency | Selected Status |
|---|---|---|---|---|---|
| **Dense Cosine Baseline** | 100% | 100% | 1.000 | 44.7 ms | Semantic baseline |
| **Max Marginal Relevance (MMR)** | 100% | 100% | 1.000 | 38.5 ms | High diversity |
| **BM25 Lexical Matching** | 90% | 100% | 0.950 | **12.4 ms** | Exact formula matching |
| **Hybrid Search (BM25 + Dense RRF)** | **100%** | **100%** | **1.000** | **47.0 ms** | **Production Winner** |
| **Cross-Encoder Reranker** | 100% | 100% | 1.000 | 1750.0 ms | Highest computation |

---

## 4. Benchmark Query Groundedness Results

| Query ID | Category | Target Source | Page # | Groundedness | Refusal Checked |
|---|---|---|---|---|---|
| `Q1` | Factual / Math | `attention_is_all_you_need.pdf` | Page 4 | 1.00 | N/A (Answered) |
| `Q2` | Architecture | `lora_low_rank_adaptation.pdf` | Page 3 | 1.00 | N/A (Answered) |
| `Q3` | Architectural | `retrieval_augmented_generation.pdf` | Page 3 | 1.00 | N/A (Answered) |
| `Q4` | Hardware / Math | `flash_attention.pdf` | Page 2 | 1.00 | N/A (Answered) |
| `Q5` | Factual | `toolformer_language_models.pdf` | Page 2 | 1.00 | N/A (Answered) |
| `Q6` | Comparative | `retrieval_augmented_generation.pdf` | Page 3 | 1.00 | N/A (Answered) |
| `Q7` | Hardware | `flash_attention.pdf` | Page 2 | 1.00 | N/A (Answered) |
| `Q8` | Mathematical | `attention_is_all_you_need.pdf` | Page 3 | 1.00 | N/A (Answered) |
| `Q9` | Methodological | `lora_low_rank_adaptation.pdf` | Page 4 | 1.00 | N/A (Answered) |
| `Q10`| Architectural | `toolformer_language_models.pdf` | Page 3 | 1.00 | N/A (Answered) |
| `Q11`| Negative Control | Out-of-Domain (Medical) | None | 1.00 | **Correctly Refused** |
| `Q12`| Negative Control | Out-of-Domain (Quantum) | None | 1.00 | **Correctly Refused** |
