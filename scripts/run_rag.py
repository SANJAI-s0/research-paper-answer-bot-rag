"""
CLI RAG Runner Script.
Allows asking questions to the Research Paper Answer Bot from the terminal.
Outputs:
- Generated Grounded Answer
- Top 3 Supporting Sources (Title, Filename, Page Number, Passage Snippet)
- Refusal / Grounding status
- Execution Latency
"""

import sys
import argparse
from pathlib import Path

# Ensure Windows stdout handles mathematical characters safely
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, format_top_sources
from src.config import config
from src.rag_chain import ResearchPaperRAGChain

logger = get_logger("RunRAG")


def main():
    parser = argparse.ArgumentParser(description="Query the Research Paper Answer Bot via CLI.")
    parser.add_argument("--query", "-q", type=str, help="Question to ask the RAG bot.")
    parser.add_argument("--strategy", "-s", type=str, default="hybrid", choices=["dense", "mmr", "bm25", "hybrid", "rerank"], help="Retrieval strategy.")
    parser.add_argument("--embedding", "-e", type=str, default=config.DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Number of supporting sources to retrieve.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive CLI session.")
    
    args = parser.parse_args()
    
    logger.info(f"Initializing RAG pipeline (Model: {args.embedding}, Strategy: {args.strategy})...")
    rag = ResearchPaperRAGChain(embedding_model_name=args.embedding, retrieval_strategy=args.strategy)
    
    if args.interactive:
        print("\n" + "=" * 80)
        print("[*] RESEARCH PAPER ANSWER BOT - INTERACTIVE CLI")
        print(f"Strategy: {args.strategy.upper()} | Embedding: {args.embedding}")
        print("Type 'exit' or 'quit' to stop.")
        print("=" * 80 + "\n")
        
        while True:
            try:
                user_q = input("Question > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break
                    
                result = rag.query(user_q, strategy=args.strategy, top_k=args.top_k)
                
                print("\n" + "-" * 80)
                print(f"[ANSWER] (Latency: {result['latency_seconds']}s):")
                print(result["answer"])
                print("-" * 80)
                print(format_top_sources(result["sources"]))
                print("-" * 80 + "\n")
            except KeyboardInterrupt:
                print("\nExiting.")
                break
    else:
        query_text = args.query or "What is the computational complexity of self-attention per layer?"
        print(f"\nEvaluating Query: '{query_text}'\n")
        result = rag.query(query_text, strategy=args.strategy, top_k=args.top_k)
        
        print("=" * 80)
        print(f"[ANSWER] (Strategy: {result['retrieval_strategy']}, Latency: {result['latency_seconds']}s):")
        print(result["answer"])
        print("=" * 80)
        print(format_top_sources(result["sources"]))
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
