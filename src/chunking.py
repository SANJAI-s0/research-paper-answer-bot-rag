"""
Chunking strategies for research paper texts.
Supports:
- RecursiveCharacterTextSplitter (Recommended for hierarchical academic structure)
- FixedSizeTextSplitter (Baseline comparison)
Ensures metadata integrity (paper title, source file, page number, chunk index).
"""

from typing import List, Dict, Any, Literal
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document

from src.utils import get_logger

logger = get_logger("Chunking")


class ChunkingEngine:
    """Handles text splitting with metadata preservation and strategy comparison."""
    
    def __init__(
        self,
        strategy: Literal["recursive", "fixed"] = "recursive",
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._init_splitter()

    def _init_splitter(self):
        """Initialize the selected text splitter."""
        if self.strategy == "recursive":
            # Hierarchical separators for academic papers: sections, paragraphs, sentences, words
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
                length_function=len,
                is_separator_regex=False,
            )
        elif self.strategy == "fixed":
            self.splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.strategy}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks while preserving and enriching metadata."""
        if not documents:
            return []
            
        raw_chunks = self.splitter.split_documents(documents)
        enriched_chunks: List[Document] = []
        
        # Track chunk indices per source file and page
        chunk_counter: Dict[str, int] = {}
        
        for doc in raw_chunks:
            source_file = doc.metadata.get("source_file", "unknown")
            page_num = doc.metadata.get("page_number", 1)
            file_key = f"{source_file}_p{page_num}"
            
            chunk_idx = chunk_counter.get(file_key, 0) + 1
            chunk_counter[file_key] = chunk_idx
            
            # Enrich metadata
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = chunk_idx
            metadata["chunk_id"] = f"{source_file}_p{page_num}_c{chunk_idx}"
            metadata["chunk_char_length"] = len(doc.page_content)
            metadata["strategy"] = self.strategy
            
            enriched_chunks.append(Document(page_content=doc.page_content, metadata=metadata))
            
        logger.info(
            f"Split {len(documents)} page documents into {len(enriched_chunks)} chunks "
            f"using '{self.strategy}' strategy (size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return enriched_chunks


def compare_chunking_strategies(documents: List[Document]) -> Dict[str, Any]:
    """Compare chunking strategies on the provided documents."""
    configs = [
        {"name": "Recursive-600", "strategy": "recursive", "size": 600, "overlap": 100},
        {"name": "Recursive-1000", "strategy": "recursive", "size": 1000, "overlap": 150},
        {"name": "Fixed-600", "strategy": "fixed", "size": 600, "overlap": 100},
        {"name": "Fixed-1000", "strategy": "fixed", "size": 1000, "overlap": 150},
    ]
    
    results = []
    for cfg in configs:
        engine = ChunkingEngine(strategy=cfg["strategy"], chunk_size=cfg["size"], chunk_overlap=cfg["overlap"])
        chunks = engine.split_documents(documents)
        lengths = [len(c.page_content) for c in chunks] if chunks else [0]
        
        results.append({
            "Strategy": cfg["name"],
            "Type": cfg["strategy"],
            "Chunk Size": cfg["size"],
            "Overlap": cfg["overlap"],
            "Total Chunks": len(chunks),
            "Avg Chunk Chars": sum(lengths) / len(lengths) if lengths else 0,
            "Min Chunk Chars": min(lengths) if lengths else 0,
            "Max Chunk Chars": max(lengths) if lengths else 0,
        })
        
    return results
