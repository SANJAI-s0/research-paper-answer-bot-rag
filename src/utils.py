"""
Utility helpers for logging, formatting, text cleanup, and citation display.
"""

import logging
import sys
from typing import List, Dict, Any

# Ensure Windows stdout handles mathematical and unicode characters gracefully
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_logger(name: str = "RAGBot") -> logging.Logger:
    """Configure and return a standard logger with encoding-safe output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


logger = get_logger("Utils")


def clean_text(text: str) -> str:
    """Normalize whitespace and remove unwanted control characters."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(line for line in lines if line)
    return " ".join(cleaned.split())


def format_source_citation(source_dict: Dict[str, Any], index: int = 1) -> str:
    """Format a single source citation for terminal or report display."""
    title = source_dict.get("paper_title", "Unknown Title")
    page = source_dict.get("page_number", "N/A")
    filename = source_dict.get("source_file", "Unknown File")
    passage = source_dict.get("passage", "").strip()
    score = source_dict.get("score")
    
    score_str = f" (Score: {score:.4f})" if score is not None else ""
    
    output = [
        f"--- [Source {index}] {title}{score_str} ---",
        f"File: {filename} | Page: {page}",
        f"Passage: \"{passage}\"",
    ]
    return "\n".join(output)


def format_top_sources(sources: List[Dict[str, Any]]) -> str:
    """Format top-3 supporting sources."""
    if not sources:
        return "No supporting sources found."
    
    formatted = [f"=== TOP {len(sources)} SUPPORTING SOURCES ==="]
    for i, src in enumerate(sources, 1):
        formatted.append(format_source_citation(src, index=i))
    return "\n\n".join(formatted)
