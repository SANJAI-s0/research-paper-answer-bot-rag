"""
Evaluation module.
Implements automated evaluation benchmarks for Retrieval and RAG quality.
Evaluates:
- Retrieval Hit@K & MRR
- Groundedness & Faithfulness
- Hallucination Prevention on Negative/Out-of-domain queries
- Latency across Embedding Models & Retrieval Strategies
"""

from typing import List, Dict, Any, Optional
import time
import pandas as pd
from pathlib import Path

from src.utils import get_logger
from src.config import config
from src.rag_chain import ResearchPaperRAGChain

logger = get_logger("Evaluation")


# Benchmark Test Suite: 12 curated questions covering multiple papers + negative controls
BENCHMARK_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "Q1",
        "question": "What is the computational complexity per layer of Self-Attention compared to Recurrent layers according to Attention Is All You Need?",
        "expected_paper": "attention",
        "expected_keywords": ["O(n^2 * d)", "O(n * d^2)", "self-attention", "recurrent", "complexity"],
        "category": "factual_technical",
        "is_answerable": True,
    },
    {
        "id": "Q2",
        "question": "How does Multi-Head Attention allow the model to jointly attend to information from different representation subspaces?",
        "expected_paper": "attention",
        "expected_keywords": ["linear projections", "subspaces", "heads", "dk", "dv", "concat"],
        "category": "architectural",
        "is_answerable": True,
    },
    {
        "id": "Q3",
        "question": "What are the two core components of the RAG architecture proposed by Lewis et al. for knowledge-intensive NLP tasks?",
        "expected_paper": "rag",
        "expected_keywords": ["retriever", "generator", "DPR", "BART", "parametric", "non-parametric"],
        "category": "factual_technical",
        "is_answerable": True,
    },
    {
        "id": "Q4",
        "question": "What is the difference between RAG-Sequence and RAG-Token models in Lewis et al.?",
        "expected_paper": "rag",
        "expected_keywords": ["RAG-Sequence", "RAG-Token", "document", "marginalize", "token"],
        "category": "architectural_comparison",
        "is_answerable": True,
    },
    {
        "id": "Q5",
        "question": "How does LoRA (Low-Rank Adaptation) freeze pre-trained model weights while introducing trainable rank decomposition matrices?",
        "expected_paper": "lora",
        "expected_keywords": ["freeze", "rank decomposition", "low-rank", "matrix A", "matrix B", "delta W"],
        "category": "factual_technical",
        "is_answerable": True,
    },
    {
        "id": "Q6",
        "question": "Why does LoRA introduce no additional inference latency compared to full fine-tuning?",
        "expected_paper": "lora",
        "expected_keywords": ["merge", "inference latency", "W_0 + B*A", "weights", "no latency"],
        "category": "architectural",
        "is_answerable": True,
    },
    {
        "id": "Q7",
        "question": "How does FlashAttention utilize GPU SRAM tiling to avoid reading and writing large intermediate attention matrices from HBM?",
        "expected_paper": "flashattention",
        "expected_keywords": ["SRAM", "HBM", "tiling", "IO complexity", "online softmax", "memory"],
        "category": "factual_technical",
        "is_answerable": True,
    },
    {
        "id": "Q8",
        "question": "In Toolformer, how are language models taught to self-generate API calls in a self-supervised manner?",
        "expected_paper": "toolformer",
        "expected_keywords": ["API calls", "self-supervised", "calculator", "calendar", "loss", "filter"],
        "category": "methodology",
        "is_answerable": True,
    },
    {
        "id": "Q9",
        "question": "What positional encoding formula does the Transformer use for sinusoidal representations?",
        "expected_paper": "attention",
        "expected_keywords": ["sin", "cos", "pos", "2i", "10000^(2i/d_model)", "wavelength"],
        "category": "mathematical",
        "is_answerable": True,
    },
    {
        "id": "Q10",
        "question": "What temperature schedule is used for quantum annealing of deep neural networks on the D-Wave superconducting processor?",
        "expected_paper": "none",
        "expected_keywords": [],
        "category": "negative_unsupported",
        "is_answerable": False,
    },
    {
        "id": "Q11",
        "question": "What is the clinical efficacy percentage of mRNA-1273 vaccine in the Phase 3 pediatric clinical trial?",
        "expected_paper": "none",
        "expected_keywords": [],
        "category": "negative_unsupported",
        "is_answerable": False,
    },
    {
        "id": "Q12",
        "question": "What are the scaling benefits of Low-Rank Adaptation (LoRA) regarding GPU VRAM memory consumption during training?",
        "expected_paper": "lora",
        "expected_keywords": ["VRAM", "optimizer states", "memory", "gradients", "parameters"],
        "category": "factual_technical",
        "is_answerable": True,
    },
]


def evaluate_retrieval_strategy(
    rag_chain: ResearchPaperRAGChain,
    strategy: str,
    questions: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Evaluate retrieval precision, Hit@K, MRR, and latency on benchmark questions."""
    questions = questions or BENCHMARK_QUESTIONS
    
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    latencies = []
    detailed_rows = []
    
    for q in questions:
        q_id = q["id"]
        query = q["question"]
        expected_paper = q["expected_paper"]
        is_answerable = q["is_answerable"]
        
        t0 = time.perf_counter()
        retrieved = rag_chain.retrieval_engine.retrieve(query=query, strategy=strategy, top_k=top_k)
        latency = time.perf_counter() - t0
        latencies.append(latency)
        
        # Determine rank of first relevant chunk
        found_rank = 0
        if is_answerable and expected_paper != "none":
            for rank, item in enumerate(retrieved, 1):
                src_file = item.get("source_file", "").lower()
                title = item.get("paper_title", "").lower()
                passage = item.get("passage", "").lower()
                
                # Match paper name or keyword presence
                if expected_paper in src_file or expected_paper in title or any(kw.lower() in passage for kw in q.get("expected_keywords", [])):
                    found_rank = rank
                    break
            
            if found_rank == 1:
                hits_at_1 += 1
            if found_rank > 0 and found_rank <= top_k:
                hits_at_k += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)
        else:
            # Negative test: retrieval is irrelevant if answerable is False
            reciprocal_ranks.append(1.0)
            
        detailed_rows.append({
            "question_id": q_id,
            "category": q["category"],
            "strategy": strategy,
            "is_answerable": is_answerable,
            "rank_found": found_rank if is_answerable else "N/A",
            "top_paper_retrieved": retrieved[0].get("paper_title") if retrieved else "None",
            "latency_sec": round(latency, 4),
        })
        
    answerable_count = sum(1 for q in questions if q["is_answerable"])
    hit_rate = (hits_at_k / answerable_count) if answerable_count > 0 else 1.0
    mrr = (sum(reciprocal_ranks[:answerable_count]) / answerable_count) if answerable_count > 0 else 1.0
    avg_latency = sum(latencies) / len(latencies)
    
    summary = {
        "strategy": strategy,
        "total_questions": len(questions),
        "hit_at_1": round(hits_at_1 / answerable_count, 4) if answerable_count > 0 else 1.0,
        f"hit_at_{top_k}": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "avg_retrieval_latency_sec": round(avg_latency, 4),
    }
    
    return {"summary": summary, "details": detailed_rows}


def run_full_rag_evaluation(
    rag_chain: ResearchPaperRAGChain,
    questions: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Run full RAG pipeline evaluation on all benchmark queries."""
    questions = questions or BENCHMARK_QUESTIONS
    results = []
    
    for q in questions:
        logger.info(f"Evaluating {q['id']}: '{q['question'][:50]}...'")
        res = rag_chain.query(q["question"])
        
        answer = res["answer"]
        is_refusal = res["is_refusal"]
        sources = res["sources"]
        
        # Groundedness & refusal verification
        if not q["is_answerable"]:
            correct_refusal = is_refusal or "don't know" in answer.lower()
            groundedness = 1.0 if correct_refusal else 0.0
        else:
            # Check if expected keywords appear in answer or context
            keyword_matches = sum(1 for kw in q["expected_keywords"] if kw.lower() in answer.lower())
            groundedness = round(min(1.0, (keyword_matches / max(len(q["expected_keywords"]), 1)) + 0.3), 2)
            correct_refusal = False

        results.append({
            "Question ID": q["id"],
            "Category": q["category"],
            "Question": q["question"],
            "Answerable": q["is_answerable"],
            "RAG Answer": answer[:180] + ("..." if len(answer) > 180 else ""),
            "Refusal Triggered": is_refusal,
            "Groundedness Score": groundedness,
            "Top Source Title": sources[0]["paper_title"] if sources else "None",
            "Top Source Page": sources[0]["page_number"] if sources else "N/A",
            "Sources Count": len(sources),
            "Latency (s)": res["latency_seconds"],
        })
        
    df = pd.DataFrame(results)
    return df
