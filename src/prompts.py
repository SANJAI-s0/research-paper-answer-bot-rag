"""
Prompt templates module for Grounded RAG.
Enforces strict hallucination prevention, citation grounding, and refusal instructions.
"""

from typing import List, Dict, Any
from src.config import config


RAG_SYSTEM_PROMPT = """You are an expert AI Research Assistant specialized in answering scientific and technical questions strictly from provided research paper excerpts.

CRITICAL OPERATIONAL RULES:
1. Grounded Answers: Answer the question ONLY and EXCLUSIVELY using the facts directly mentioned in the provided Context Passages below.
2. No Extrapolation: Do NOT extrapolate, speculate, make assumptions, or introduce outside knowledge from prior training.
3. Strict Refusal: If the provided Context Passages do not contain sufficient information to answer the question factually and completely, you MUST respond EXACTLY with:
   "{refusal_response}"
4. In-text Citations: Whenever you state a fact, cite the supporting paper title and page number like: [Paper Title, Page X].
5. Tone: Be concise, direct, professional, and mathematically/scientifically accurate.
""".format(refusal_response=config.REFUSAL_RESPONSE)


RAG_USER_PROMPT_TEMPLATE = """CONTEXT PASSAGES:
{context_text}

USER QUESTION:
{question}

Please provide your grounded answer below:"""


def format_context_passages(retrieved_sources: List[Dict[str, Any]]) -> str:
    """Format retrieved document chunks into clean numbered context blocks for prompt injection."""
    if not retrieved_sources:
        return "No relevant passages found in indexed papers."
    
    formatted_blocks = []
    for i, src in enumerate(retrieved_sources, 1):
        title = src.get("paper_title", "Unknown Title")
        page = src.get("page_number", "N/A")
        filename = src.get("source_file", "Unknown File")
        passage = src.get("passage", "").strip()
        
        block = f"[Passage {i}] (Paper: \"{title}\", File: {filename}, Page {page})\n{passage}"
        formatted_blocks.append(block)
        
    return "\n\n".join(formatted_blocks)


def build_rag_prompt(question: str, retrieved_sources: List[Dict[str, Any]]) -> Dict[str, str]:
    """Construct full system prompt and user prompt for Gemini."""
    context_text = format_context_passages(retrieved_sources)
    user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
        context_text=context_text,
        question=question.strip(),
    )
    return {
        "system_instruction": RAG_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "context_text": context_text,
    }
