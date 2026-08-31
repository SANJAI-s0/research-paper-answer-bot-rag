"""
Retrieval module.
Implements and benchmarks multiple retrieval strategies:
1. Dense Retrieval (Cosine Similarity) - Baseline
2. Maximal Marginal Relevance (MMR) - Relevance + Diversity
3. BM25 Keyword Search - Exact lexical matching
4. Hybrid Retrieval - Reciprocal Rank Fusion (BM25 + Dense)
5. Cross-Encoder Reranking - High-precision neural reranker
"""

from typing import List, Dict, Any, Optional, Tuple, Literal
import time
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.utils import get_logger, clean_text
from src.config import config
from src.vector_store import ChromaVectorStoreManager

logger = get_logger("Retrieval")


class BM25Retriever:
    """In-memory BM25 lexical retriever over document chunks."""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None
        logger.info(f"Initialized BM25Retriever over {len(documents)} document chunks.")

    def _tokenize(self, text: str) -> List[str]:
        return clean_text(text).lower().split()

    def get_relevant_documents(self, query: str, k: int = config.TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort by BM25 score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]


class HybridRetriever:
    """Combines BM25 lexical scores and ChromaDB dense cosine similarity via Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, vector_store_manager: ChromaVectorStoreManager, all_chunks: List[Document], rrf_k: int = 60):
        self.vector_store_manager = vector_store_manager
        self.bm25_retriever = BM25Retriever(all_chunks)
        self.rrf_k = rrf_k
        logger.info(f"Initialized HybridRetriever (Dense + BM25 with RRF constant k={rrf_k}).")

    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL, fetch_k: int = 10) -> List[Dict[str, Any]]:
        # 1. Dense retrieval
        dense_results = self.vector_store_manager.similarity_search_with_score(query, k=fetch_k)
        
        # 2. BM25 retrieval
        bm25_results = self.bm25_retriever.get_relevant_documents(query, k=fetch_k)
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Dense ranks (Chroma returns distance, smaller is closer)
        for rank, (doc, dist) in enumerate(dense_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:40])
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        # BM25 ranks
        for rank, (doc, score) in enumerate(bm25_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:40])
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        # Sort documents by combined RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        
        output = []
        for cid in sorted_ids:
            doc = doc_map[cid]
            output.append({
                "document": doc,
                "score": float(rrf_scores[cid]),
                "paper_title": doc.metadata.get("paper_title", "Unknown"),
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page_number": doc.metadata.get("page_number", 1),
                "passage": doc.page_content,
                "strategy": "hybrid_rrf",
            })
            
        return output


class AdvancedRetrievalEngine:
    """Unified retrieval interface offering Dense, MMR, BM25, Hybrid, and Rerank."""
    
    def __init__(
        self,
        vector_store_manager: ChromaVectorStoreManager,
        all_chunks: Optional[List[Document]] = None,
    ):
        self.vsm = vector_store_manager
        self.all_chunks = all_chunks or []
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._reranker = None

    def _get_bm25(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            if not self.all_chunks:
                # Load all chunks from Chroma collection
                docs_dict = self.vsm.vector_store._collection.get(include=["documents", "metadatas"])
                loaded_docs = []
                for content, meta in zip(docs_dict.get("documents", []), docs_dict.get("metadatas", [])):
                    loaded_docs.append(Document(page_content=content, metadata=meta or {}))
                self.all_chunks = loaded_docs
            self._bm25_retriever = BM25Retriever(self.all_chunks)
        return self._bm25_retriever

    def _get_hybrid(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            _ = self._get_bm25()
            self._hybrid_retriever = HybridRetriever(self.vsm, self.all_chunks)
        return self._hybrid_retriever

    def retrieve(
        self,
        query: str,
        strategy: Literal["dense", "mmr", "hybrid", "bm25", "rerank"] = "hybrid",
        top_k: int = config.TOP_K_RETRIEVAL,
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k documents according to the chosen strategy."""
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []

        if strategy == "dense":
            raw = self.vsm.similarity_search_with_score(query, k=top_k)
            for doc, distance in raw:
                # In Chroma with cosine similarity, similarity = 1 - distance
                sim = max(0.0, 1.0 - float(distance))
                results.append({
                    "document": doc,
                    "score": round(sim, 4),
                    "paper_title": doc.metadata.get("paper_title", "Unknown"),
                    "source_file": doc.metadata.get("source_file", "Unknown"),
                    "page_number": doc.metadata.get("page_number", 1),
                    "passage": doc.page_content,
                    "strategy": "dense_cosine",
                })

        elif strategy == "mmr":
            raw_docs = self.vsm.max_marginal_relevance_search(query, k=top_k, fetch_k=config.RETRIEVAL_FETCH_K)
            for i, doc in enumerate(raw_docs):
                results.append({
                    "document": doc,
                    "score": round(1.0 - (i * 0.1), 4),  # MMR rank weight
                    "paper_title": doc.metadata.get("paper_title", "Unknown"),
                    "source_file": doc.metadata.get("source_file", "Unknown"),
                    "page_number": doc.metadata.get("page_number", 1),
                    "passage": doc.page_content,
                    "strategy": "mmr",
                })

        elif strategy == "bm25":
            bm25 = self._get_bm25()
            bm25_raw = bm25.get_relevant_documents(query, k=top_k)
            for doc, score in bm25_raw:
                results.append({
                    "document": doc,
                    "score": round(score, 4),
                    "paper_title": doc.metadata.get("paper_title", "Unknown"),
                    "source_file": doc.metadata.get("source_file", "Unknown"),
                    "page_number": doc.metadata.get("page_number", 1),
                    "passage": doc.page_content,
                    "strategy": "bm25",
                })

        elif strategy == "hybrid":
            hybrid = self._get_hybrid()
            results = hybrid.retrieve(query, top_k=top_k)

        elif strategy == "rerank":
            # Retrieve initial pool with dense similarity, then rerank with cross-encoder
            try:
                from sentence_transformers import CrossEncoder
                if self._reranker is None:
                    self._reranker = CrossEncoder(config.RERANKER_MODEL)
                
                candidates = self.vsm.similarity_search_with_score(query, k=config.RETRIEVAL_FETCH_K)
                if not candidates:
                    return []
                
                pairs = [[query, doc.page_content] for doc, _ in candidates]
                rerank_scores = self._reranker.predict(pairs)
                
                ranked_indices = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)[:top_k]
                for idx in ranked_indices:
                    doc, _ = candidates[idx]
                    results.append({
                        "document": doc,
                        "score": round(float(rerank_scores[idx]), 4),
                        "paper_title": doc.metadata.get("paper_title", "Unknown"),
                        "source_file": doc.metadata.get("source_file", "Unknown"),
                        "page_number": doc.metadata.get("page_number", 1),
                        "passage": doc.page_content,
                        "strategy": "cross_encoder_rerank",
                    })
            except Exception as e:
                logger.warning(f"Reranking fallback to dense search: {str(e)}")
                return self.retrieve(query, strategy="dense", top_k=top_k)
        
        latency = round(time.perf_counter() - t0, 4)
        logger.info(f"Retrieved {len(results)} chunks for query '{query[:30]}...' using strategy '{strategy}' in {latency}s.")
        return results
