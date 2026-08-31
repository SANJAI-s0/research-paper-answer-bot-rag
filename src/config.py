"""
Configuration module for the Research Paper Answer Bot RAG system.
Loads settings from environment variables and provides structured access.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env file from project root if present
load_dotenv()

# Base project paths (Windows and POSIX compatible)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_RESULTS_DIR = OUTPUTS_DIR / "evaluation_results"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Ensure runtime directories exist
for directory in [DATA_DIR, PAPERS_DIR, CHROMA_PERSIST_DIR, EXPERIMENTS_DIR, OUTPUTS_DIR, EVAL_RESULTS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class AppConfig:
    """Central configuration class."""
    
    # Storage Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = DATA_DIR
    PAPERS_DIR: Path = PAPERS_DIR
    CHROMA_PERSIST_DIR: Path = CHROMA_PERSIST_DIR
    EXPERIMENTS_DIR: Path = EXPERIMENTS_DIR
    OUTPUTS_DIR: Path = OUTPUTS_DIR
    EVAL_RESULTS_DIR: Path = EVAL_RESULTS_DIR
    VISUALIZATIONS_DIR: Path = VISUALIZATIONS_DIR

    # LLM Configuration (Google Gemini only)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
    
    # Embedding Models (Hugging Face sentence-transformers)
    EMBEDDING_MODEL_A: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_B: str = "BAAI/bge-small-en-v1.5"
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", EMBEDDING_MODEL_A)
    
    # Chunking Parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # Retrieval Configuration
    DEFAULT_RETRIEVAL_STRATEGY: str = os.getenv("RETRIEVAL_STRATEGY", "hybrid")  # "dense", "mmr", "hybrid", "rerank"
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "3"))
    RETRIEVAL_FETCH_K: int = int(os.getenv("RETRIEVAL_FETCH_K", "8"))
    
    # Reranker Model (optional cross-encoder)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Groundedness & Refusal
    REFUSAL_RESPONSE: str = "I don't know based on the provided papers."
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))


config = AppConfig()
