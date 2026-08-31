"""
Embedding Models Comparison & Evaluation Script.
Compares Embedding Model A (all-MiniLM-L6-v2) vs Model B (bge-small-en-v1.5).
Generates benchmark results and exports `experiments/embedding_comparison.csv`.
"""

import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger
from src.config import config
from src.document_loader import load_all_papers
from src.chunking import ChunkingEngine
from src.embeddings import benchmark_embedding_models
from src.evaluation import BENCHMARK_QUESTIONS

logger = get_logger("EvalEmbeddings")


def run_embedding_benchmarks():
    """Run comparative benchmark between Embedding Model A and Model B."""
    logger.info("Loading document chunks for embedding evaluation...")
    docs = load_all_papers(config.PAPERS_DIR)
    
    if not docs:
        logger.error("No documents found in data/papers/! Run scripts/ingest.py first.")
        return
        
    chunk_engine = ChunkingEngine(strategy="recursive", chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = chunk_engine.split_documents(docs)
    
    sample_passages = [c.page_content for c in chunks[:40]]  # Representative passage sample
    sample_queries = [q["question"] for q in BENCHMARK_QUESTIONS if q["is_answerable"]]
    
    models_to_test = [
        config.EMBEDDING_MODEL_A,
        config.EMBEDDING_MODEL_B,
    ]
    
    logger.info(f"Benchmarking {len(models_to_test)} models on {len(sample_passages)} chunks and {len(sample_queries)} queries...")
    results = benchmark_embedding_models(sample_queries, sample_passages, models=models_to_test)
    
    # Add qualitative notes and selection rationale
    enriched_results = []
    for r in results:
        m_name = r["model_name"]
        if "all-MiniLM" in m_name:
            r["architecture"] = "6-layer MiniLM (22.7M params)"
            r["strengths"] = "Extremely lightweight, ultra-low CPU latency, small memory footprint"
            r["limitations"] = "Lower semantic granularity on dense technical domain nuances"
            r["selection_status"] = "Baseline / High-Throughput Choice"
        else:
            r["architecture"] = "12-layer BERT-based (33.4M params)"
            r["strengths"] = "Trained with retro-contrastive loss, superior MTEB retrieval accuracy"
            r["limitations"] = "Slightly higher compute requirement per token"
            r["selection_status"] = "Selected Optimal Model (High Precision)"
        enriched_results.append(r)
        
    df = pd.DataFrame(enriched_results)
    
    # Save to CSV
    output_path = config.EXPERIMENTS_DIR / "embedding_comparison.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved embedding comparison table to '{output_path}'.")
    
    print("\n" + "=" * 80)
    print("EMBEDDING MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(tabulate(df[["model_name", "dimension", "doc_throughput_per_sec", "query_encode_time_sec", "avg_max_cosine_similarity", "selection_status"]], headers="keys", tablefmt="pretty"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_embedding_benchmarks()
