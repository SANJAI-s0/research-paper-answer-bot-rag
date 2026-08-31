"""
Document Loader module.
Extracts text from research paper PDFs and preserves rich metadata:
- paper_title
- source_file (filename)
- page_number (1-indexed)
- total_pages
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
from langchain_core.documents import Document

from src.utils import get_logger, clean_text

logger = get_logger("DocumentLoader")


class ResearchPaperLoader:
    """PDF Document Loader preserving research paper metadata."""
    
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.file_path}")
        self.filename = self.file_path.name

    def _extract_title_from_text(self, first_page_text: str) -> str:
        """Heuristically extract paper title from the top lines of the first page."""
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
        for line in lines[:5]:
            # Filter out arXiv headers, dates, or very short lines
            if len(line) > 8 and not line.lower().startswith(("arxiv:", "published", "http", "vol.", "page")):
                return line
        # Fallback to sanitized filename
        return self.filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()

    def load(self) -> List[Document]:
        """Load the PDF and return LangChain Documents per page with full metadata."""
        documents: List[Document] = []
        try:
            reader = pypdf.PdfReader(str(self.file_path))
            total_pages = len(reader.pages)
            
            if total_pages == 0:
                logger.warning(f"Empty PDF document: {self.filename}")
                return []
            
            # Infer title from first page
            first_page_text = reader.pages[0].extract_text() or ""
            paper_title = self._extract_title_from_text(first_page_text)
            
            for page_idx, page in enumerate(reader.pages):
                page_number = page_idx + 1
                raw_text = page.extract_text() or ""
                cleaned = clean_text(raw_text)
                
                if not cleaned:
                    logger.debug(f"Skipping empty text on page {page_number} of {self.filename}")
                    continue
                
                metadata: Dict[str, Any] = {
                    "source": str(self.file_path),
                    "source_file": self.filename,
                    "paper_title": paper_title,
                    "page_number": page_number,
                    "total_pages": total_pages,
                }
                
                documents.append(Document(page_content=cleaned, metadata=metadata))
            
            logger.info(f"Loaded '{paper_title}' ({self.filename}): {len(documents)}/{total_pages} pages extracted.")
            return documents

        except Exception as e:
            logger.error(f"Failed to load PDF '{self.filename}': {str(e)}")
            raise e


def load_all_papers(papers_dir: Path) -> List[Document]:
    """Load all PDF research papers found in the specified directory."""
    papers_dir = Path(papers_dir)
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in directory: {papers_dir}")
        return []
    
    all_docs: List[Document] = []
    for pdf_path in sorted(pdf_files):
        loader = ResearchPaperLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
        
    logger.info(f"Total documents loaded across {len(pdf_files)} PDFs: {len(all_docs)} pages.")
    return all_docs
