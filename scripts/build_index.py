"""
Build Vector Index Script.
Loads all research paper PDFs, splits them into semantic chunks with full metadata,
and indexes into ChromaDB collections for both embedding models.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger
from src.config import config
from src.document_loader import load_all_papers
from src.chunking import ChunkingEngine, compare_chunking_strategies
from src.vector_store import ChromaVectorStoreManager

logger = get_logger("BuildIndex")


def build_all_indices():
    """Build persistent vector indices for both embedding models."""
    logger.info("Step 1: Loading PDF research papers from data/papers/...")
    documents = load_all_papers(config.PAPERS_DIR)
    
    if not documents:
        logger.error("No documents found! Run scripts/ingest.py first to populate data/papers/.")
        return
        
    logger.info(f"Loaded {len(documents)} page documents.")
    
    # Chunking comparison & statistics
    logger.info("Step 2: Performing text chunking comparison...")
    chunk_stats = compare_chunking_strategies(documents)
    logger.info("Chunking Comparison Results:")
    for stat in chunk_stats:
        logger.info(
            f" - {stat['Strategy']}: {stat['Total Chunks']} chunks (avg: {stat['Avg Chunk Chars']:.1f} chars, "
            f"min: {stat['Min Chunk Chars']}, max: {stat['Max Chunk Chars']})"
        )
        
    # Primary chunking using RecursiveCharacterTextSplitter
    chunk_engine = ChunkingEngine(
        strategy="recursive",
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = chunk_engine.split_documents(documents)
    logger.info(f"Generated {len(chunks)} enriched chunks with metadata.")
    
    # Step 3: Index for Model A (all-MiniLM-L6-v2)
    logger.info(f"Step 3: Indexing chunks into ChromaDB with Model A ({config.EMBEDDING_MODEL_A})...")
    vsm_a = ChromaVectorStoreManager(
        persist_dir=config.CHROMA_PERSIST_DIR,
        embedding_model_name=config.EMBEDDING_MODEL_A,
    )
    vsm_a.index_documents(chunks)
    
    # Step 4: Index for Model B (BAAI/bge-small-en-v1.5)
    logger.info(f"Step 4: Indexing chunks into ChromaDB with Model B ({config.EMBEDDING_MODEL_B})...")
    vsm_b = ChromaVectorStoreManager(
        persist_dir=config.CHROMA_PERSIST_DIR,
        embedding_model_name=config.EMBEDDING_MODEL_B,
    )
    vsm_b.index_documents(chunks)
    
    logger.info("All vector indices built successfully and persisted to chroma_db/.")


if __name__ == "__main__":
    build_all_indices()
