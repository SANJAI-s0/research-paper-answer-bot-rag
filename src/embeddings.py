"""
Embeddings module.
Supports open-source Hugging Face embedding models via sentence-transformers:
- Model A: sentence-transformers/all-MiniLM-L6-v2 (384 dims, fast, lightweight baseline)
- Model B: BAAI/bge-small-en-v1.5 (384 dims, optimized for dense passage retrieval)
Provides embedding generation, cosine similarity, and comparative benchmarking.
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from src.utils import get_logger
from src.config import config

logger = get_logger("Embeddings")


class LocalHuggingFaceEmbeddings(Embeddings):
    """LangChain compatible Hugging Face Embeddings wrapper."""
    
    def __init__(self, model_name: str = config.DEFAULT_EMBEDDING_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading embedding model: '{self.model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_embedding_dimension() if hasattr(self.model, "get_embedding_dimension") else self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded '{self.model_name}' successfully. Embedding dimension: {self.dimension}.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()


_EMBEDDING_CACHE: Dict[str, LocalHuggingFaceEmbeddings] = {}


def get_embedding_model(model_name: Optional[str] = None) -> LocalHuggingFaceEmbeddings:
    """Factory to retrieve a cached embedding model instance."""
    selected_name = model_name or config.DEFAULT_EMBEDDING_MODEL
    if selected_name not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[selected_name] = LocalHuggingFaceEmbeddings(model_name=selected_name)
    return _EMBEDDING_CACHE[selected_name]


def benchmark_embedding_models(
    sample_queries: List[str],
    sample_passages: List[str],
    models: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Compare embedding models on latency, dimension, and average semantic similarity."""
    if models is None:
        models = [config.EMBEDDING_MODEL_A, config.EMBEDDING_MODEL_B]

    benchmark_results = []
    
    for model_name in models:
        logger.info(f"Benchmarking embedding model: {model_name}")
        emb_wrapper = LocalHuggingFaceEmbeddings(model_name=model_name)
        
        # Measure document embedding time
        t0 = time.perf_counter()
        doc_vectors = np.array(emb_wrapper.embed_documents(sample_passages))
        t_docs = time.perf_counter() - t0
        
        # Measure query embedding time
        t0 = time.perf_counter()
        query_vectors = np.array(emb_wrapper.embed_documents(sample_queries))
        t_queries = time.perf_counter() - t0
        
        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(query_vectors, doc_vectors)
        avg_max_sim = float(np.mean(np.max(sim_matrix, axis=1)))
        
        benchmark_results.append({
            "model_name": model_name,
            "dimension": emb_wrapper.dimension,
            "docs_encoded": len(sample_passages),
            "doc_encode_time_sec": round(t_docs, 4),
            "doc_throughput_per_sec": round(len(sample_passages) / max(t_docs, 1e-5), 2),
            "query_encode_time_sec": round(t_queries, 4),
            "avg_max_cosine_similarity": round(avg_max_sim, 4),
        })
        
    return benchmark_results
