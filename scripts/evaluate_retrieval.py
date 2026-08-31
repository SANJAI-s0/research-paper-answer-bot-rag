"""
Retrieval Strategies Comparison & Evaluation Script.
Compares:
1. Dense Cosine Similarity (Baseline)
2. Maximal Marginal Relevance (MMR)
3. BM25 Lexical Keyword Search
4. Hybrid Search (BM25 + Dense RRF)
5. Cross-Encoder Reranking
Evaluates on all 12 benchmark questions and exports results to CSV.
"""

import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger
from src.config import config
from src.rag_chain import ResearchPaperRAGChain
from src.evaluation import BENCHMARK_QUESTIONS, evaluate_retrieval_strategy, run_full_rag_evaluation

logger = get_logger("EvalRetrieval")


def run_retrieval_benchmarks():
    """Execute comparative benchmarks across all retrieval strategies."""
    logger.info("Initializing RAG chain for retrieval evaluation...")
    rag_chain = ResearchPaperRAGChain()
    
    strategies = ["dense", "mmr", "bm25", "hybrid", "rerank"]
    summary_results = []
    
    for strat in strategies:
        logger.info(f"Evaluating strategy: '{strat}'...")
        eval_res = evaluate_retrieval_strategy(rag_chain, strategy=strat, top_k=config.TOP_K_RETRIEVAL)
        summary = eval_res["summary"]
        
        if strat == "dense":
            summary["description"] = "Cosine vector similarity baseline"
            summary["strengths"] = "Fastest, standard semantic search"
            summary["tradeoffs"] = "May miss exact acronyms/equations"
        elif strat == "mmr":
            summary["description"] = "Maximal Marginal Relevance"
            summary["strengths"] = "Penalizes redundant neighboring chunks, increases diversity"
            summary["tradeoffs"] = "Marginally lower precision on singular topic queries"
        elif strat == "bm25":
            summary["description"] = "BM25 keyword search"
            summary["strengths"] = "High precision for exact terms (e.g., 'O(n^2*d)', 'SRAM')"
            summary["tradeoffs"] = "Fails when vocabulary differs from query"
        elif strat == "hybrid":
            summary["description"] = "Reciprocal Rank Fusion (BM25 + Dense)"
            summary["strengths"] = "Optimal balance of exact keywords and deep semantics"
            summary["tradeoffs"] = "Slightly higher computation (combines 2 rank lists)"
        elif strat == "rerank":
            summary["description"] = "Cross-Encoder neural reranker"
            summary["strengths"] = "Highest context precision on complex queries"
            summary["tradeoffs"] = "Higher latency due to pairwise cross-attention"
            
        summary_results.append(summary)
        
    df_summary = pd.DataFrame(summary_results)
    
    # Save comparison to CSV
    retrieval_csv_path = config.EXPERIMENTS_DIR / "retrieval_comparison.csv"
    df_summary.to_csv(retrieval_csv_path, index=False)
    logger.info(f"Saved retrieval comparison table to '{retrieval_csv_path}'.")
    
    print("\n" + "=" * 90)
    print("RETRIEVAL STRATEGY BENCHMARK RESULTS")
    print("=" * 90)
    display_cols = ["strategy", "hit_at_1", "hit_at_3", "mrr", "avg_retrieval_latency_sec"]
    print(tabulate(df_summary[display_cols], headers="keys", tablefmt="pretty"))
    print("=" * 90 + "\n")
    
    # Run full RAG evaluation across benchmark questions
    logger.info("Running complete RAG pipeline evaluation on benchmark questions...")
    eval_df = run_full_rag_evaluation(rag_chain)
    
    eval_csv_path = config.EXPERIMENTS_DIR / "evaluation_results.csv"
    eval_df.to_csv(eval_csv_path, index=False)
    logger.info(f"Saved detailed evaluation results to '{eval_csv_path}'.")
    
    print("\n" + "=" * 90)
    print("SAMPLE EVALUATION RESULTS ACROSS 12 BENCHMARK QUESTIONS")
    print("=" * 90)
    print(tabulate(eval_df[["Question ID", "Category", "Answerable", "Refusal Triggered", "Groundedness Score", "Latency (s)"]], headers="keys", tablefmt="pretty"))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_retrieval_benchmarks()
