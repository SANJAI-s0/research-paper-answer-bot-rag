"""
RAG Chain Orchestrator module.
Coordinates document retrieval, prompt building, LLM synthesis via Google Gemini,
and extraction of top-3 supporting citations.
"""

from typing import List, Dict, Any, Optional, Literal
import time
from pathlib import Path

from src.utils import get_logger, format_top_sources
from src.config import config
from src.vector_store import ChromaVectorStoreManager, get_vector_store
from src.retrieval import AdvancedRetrievalEngine
from src.prompts import build_rag_prompt
from src.gemini_client import GeminiLLMClient, get_gemini_client

logger = get_logger("RAGChain")


class ResearchPaperRAGChain:
    """End-to-End RAG Pipeline for answering research paper queries."""
    
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        retrieval_strategy: Optional[str] = None,
        gemini_client: Optional[GeminiLLMClient] = None,
    ):
        self.embedding_model_name = embedding_model_name or config.DEFAULT_EMBEDDING_MODEL
        self.retrieval_strategy = retrieval_strategy or config.DEFAULT_RETRIEVAL_STRATEGY
        
        self.vector_store_manager = get_vector_store(embedding_model_name=self.embedding_model_name)
        self.retrieval_engine = AdvancedRetrievalEngine(vector_store_manager=self.vector_store_manager)
        self.gemini_client = gemini_client or get_gemini_client()

    def query(
        self,
        question: str,
        strategy: Optional[str] = None,
        top_k: int = config.TOP_K_RETRIEVAL,
    ) -> Dict[str, Any]:
        """Execute complete RAG pipeline for a user question."""
        start_time = time.perf_counter()
        active_strategy = strategy or self.retrieval_strategy
        
        # 1. Retrieve supporting sources
        retrieved_sources = self.retrieval_engine.retrieve(
            query=question,
            strategy=active_strategy,
            top_k=top_k,
        )
        
        # 2. Check if context is completely empty
        if not retrieved_sources:
            return {
                "question": question,
                "answer": config.REFUSAL_RESPONSE,
                "sources": [],
                "is_refusal": True,
                "retrieval_strategy": active_strategy,
                "embedding_model": self.embedding_model_name,
                "latency_seconds": round(time.perf_counter() - start_time, 4),
            }
            
        # 3. Build grounded prompt
        prompt_payload = build_rag_prompt(question, retrieved_sources)
        
        # 4. Generate answer with Gemini
        answer = self.gemini_client.generate_response(
            system_instruction=prompt_payload["system_instruction"],
            user_prompt=prompt_payload["user_prompt"],
        )
        
        # 5. Check if LLM refused or answered
        is_refusal = (
            config.REFUSAL_RESPONSE.lower() in answer.lower()
            or "don't know" in answer.lower()
            or "insufficient information" in answer.lower()
        )
        
        latency = round(time.perf_counter() - start_time, 4)
        
        result = {
            "question": question,
            "answer": answer,
            "sources": retrieved_sources[:top_k],
            "is_refusal": is_refusal,
            "retrieval_strategy": active_strategy,
            "embedding_model": self.embedding_model_name,
            "latency_seconds": latency,
            "context_passages_formatted": prompt_payload["context_text"],
        }
        
        return result


def create_rag_pipeline(
    embedding_model_name: Optional[str] = None,
    retrieval_strategy: Optional[str] = None,
) -> ResearchPaperRAGChain:
    """Factory helper to instantiate a configured RAG pipeline."""
    return ResearchPaperRAGChain(
        embedding_model_name=embedding_model_name,
        retrieval_strategy=retrieval_strategy,
    )
