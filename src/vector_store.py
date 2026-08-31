"""
Vector Store module.
Manages persistent ChromaDB vector collections for chunked research papers.
Supports model-specific collections and metadata persistence.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma

from src.utils import get_logger
from src.config import config
from src.embeddings import LocalHuggingFaceEmbeddings, get_embedding_model

logger = get_logger("VectorStore")


def sanitize_collection_name(name: str) -> str:
    """Ensure collection name meets ChromaDB requirements (3-63 chars, alphanumeric/underscores)."""
    clean = name.replace("/", "_").replace("-", "_").replace(".", "_")
    return clean[:63]


class ChromaVectorStoreManager:
    """Manages ChromaDB vector store creation, loading, and queries."""
    
    def __init__(
        self,
        persist_dir: Path = config.CHROMA_PERSIST_DIR,
        embedding_model_name: str = config.DEFAULT_EMBEDDING_MODEL,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self.embeddings = get_embedding_model(self.embedding_model_name)
        self.collection_name = sanitize_collection_name(f"rag_{self.embedding_model_name}")
        
        logger.info(f"Initializing ChromaDB vector store at '{self.persist_dir}', collection: '{self.collection_name}'")
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def index_documents(self, documents: List[Document], batch_size: int = 64) -> int:
        """Index chunked documents with full metadata into ChromaDB in batches."""
        if not documents:
            logger.warning("No documents provided for indexing.")
            return 0
            
        logger.info(f"Indexing {len(documents)} chunk documents into collection '{self.collection_name}'...")
        
        # Add documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vector_store.add_documents(batch)
            
        count = self.get_document_count()
        logger.info(f"Indexing complete. Total chunks in '{self.collection_name}': {count}")
        return count

    def get_document_count(self) -> int:
        """Return the number of items in the vector store collection."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def get_indexed_papers_summary(self) -> List[Dict[str, Any]]:
        """Retrieve a summary of papers indexed in this collection."""
        try:
            results = self.vector_store._collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            papers: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                if not meta:
                    continue
                filename = meta.get("source_file", "unknown")
                if filename not in papers:
                    papers[filename] = {
                        "paper_title": meta.get("paper_title", "Unknown"),
                        "source_file": filename,
                        "total_pages": meta.get("total_pages", "N/A"),
                        "chunk_count": 0,
                    }
                papers[filename]["chunk_count"] += 1
                
            return list(papers.values())
        except Exception as e:
            logger.warning(f"Could not retrieve paper summary: {str(e)}")
            return []

    def similarity_search_with_score(
        self, query: str, k: int = config.TOP_K_RETRIEVAL
    ) -> List[Tuple[Document, float]]:
        """Execute dense cosine similarity search returning docs and distance scores."""
        return self.vector_store.similarity_search_with_score(query, k=k)

    def max_marginal_relevance_search(
        self, query: str, k: int = config.TOP_K_RETRIEVAL, fetch_k: int = config.RETRIEVAL_FETCH_K
    ) -> List[Document]:
        """Execute MMR search to balance relevance and diversity."""
        return self.vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None):
        """Return a standard LangChain retriever interface."""
        kwargs = search_kwargs or {"k": config.TOP_K_RETRIEVAL}
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)


_VECTOR_STORE_CACHE: Dict[str, ChromaVectorStoreManager] = {}


def get_vector_store(
    embedding_model_name: Optional[str] = None,
    persist_dir: Path = config.CHROMA_PERSIST_DIR,
) -> ChromaVectorStoreManager:
    """Vector store manager factory with instance caching."""
    model_name = embedding_model_name or config.DEFAULT_EMBEDDING_MODEL
    cache_key = f"{model_name}_{str(persist_dir)}"
    if cache_key not in _VECTOR_STORE_CACHE:
        _VECTOR_STORE_CACHE[cache_key] = ChromaVectorStoreManager(persist_dir=persist_dir, embedding_model_name=model_name)
    return _VECTOR_STORE_CACHE[cache_key]
