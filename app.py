"""
Streamlit Web Application - Research Paper Answer Bot (RAG)
Modern, Clean & Minimal AI Research Assistant Interface.

Preserves 100% of existing backend RAG logic, Gemini API calls, ChromaDB vector store,
retrieval strategies, and source citation extraction.
"""

import sys
import os
from pathlib import Path
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import config
from src.utils import get_logger
from src.document_loader import load_all_papers
from src.vector_store import get_vector_store
from src.rag_chain import ResearchPaperRAGChain
from src.evaluation import BENCHMARK_QUESTIONS

logger = get_logger("StreamlitApp")

# ==============================================================================
# 1. PAGE CONFIGURATION & INJECTED STYLES
# ==============================================================================
st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sophisticated Light-First Modern CSS Theme
st.markdown("""
<style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Typography & Font Hierarchy */
    h1, h2, h3, h4, p, span, div {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .top-header-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .top-header-icon {
        font-size: 1.6rem;
        background: #EEF2FF;
        color: #4F46E5;
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
    }
    .top-header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
    }
    .top-header-subtitle {
        font-size: 0.84rem;
        color: #64748B;
        margin: 0;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
    }

    /* Hero / Empty State Card */
    .hero-card {
        text-align: center;
        padding: 40px 24px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        margin: 15px 0 25px 0;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
    }
    .hero-icon {
        font-size: 2.2rem;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #64748B;
        max-width: 580px;
        margin: 0 auto 20px auto;
        line-height: 1.5;
    }
    
    /* Section Labels */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }

    /* User Question Card */
    .user-question-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 14px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .user-avatar {
        background: #E2E8F0;
        color: #475569;
        font-size: 0.85rem;
        font-weight: 700;
        width: 28px;
        height: 28px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .user-text {
        font-size: 0.96rem;
        font-weight: 600;
        color: #1E293B;
        line-height: 1.4;
        margin-top: 3px;
    }

    /* AI Answer Card */
    .ai-answer-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
    }
    .ai-answer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid #F1F5F9;
    }
    .ai-answer-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #4338CA;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .ai-answer-meta {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 500;
    }
    .ai-answer-text {
        font-size: 0.98rem;
        color: #1E293B;
        line-height: 1.65;
        white-space: pre-wrap;
    }

    /* Sources Section */
    .sources-header {
        margin-top: 24px;
        margin-bottom: 12px;
    }
    .sources-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .sources-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        margin: 2px 0 12px 0;
    }

    /* Source Citation Card */
    .source-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: all 0.15s ease-in-out;
    }
    .source-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .source-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .source-index-title {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .source-num-badge {
        background: #EEF2FF;
        color: #4F46E5;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 5px;
    }
    .source-paper-title {
        font-size: 0.94rem;
        font-weight: 600;
        color: #0F172A;
    }
    .source-page-badge {
        background: #F1F5F9;
        color: #475569;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .source-passage-preview {
        font-size: 0.86rem;
        color: #334155;
        line-height: 1.45;
        font-style: italic;
        background: #F8FAFC;
        padding: 9px 13px;
        border-radius: 6px;
        border: 1px solid #F1F5F9;
        margin-bottom: 6px;
    }
    .source-meta-row {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 500;
    }

    /* Refusal / Insufficient Info Alert */
    .refusal-card {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        padding: 14px 18px;
        border-radius: 10px;
        font-size: 0.94rem;
        font-weight: 500;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Metric & Dashboard Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .metric-card-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-card-lbl {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 2px;
    }
    
    /* Sidebar Section Divider */
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 16px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. SESSION STATE & RESOURCE CACHING
# ==============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = config.GEMINI_API_KEY
if "selected_suggestion" not in st.session_state:
    st.session_state.selected_suggestion = None


@st.cache_resource(show_spinner=False)
def load_rag_pipeline(embedding_model: str, retrieval_strategy: str, api_key: str):
    """Cached loader for the RAG pipeline."""
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        config.GEMINI_API_KEY = api_key
    return ResearchPaperRAGChain(
        embedding_model_name=embedding_model,
        retrieval_strategy=retrieval_strategy,
    )


# ==============================================================================
# 3. REDESIGNED COMPACT SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">SYSTEM</div>', unsafe_allow_html=True)
    
    # Gemini API Key configuration
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter Google Gemini Key...",
        help="Get your free key from Google AI Studio. Key is never saved to repo.",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        os.environ["GEMINI_API_KEY"] = api_key_input
        config.GEMINI_API_KEY = api_key_input
        st.cache_resource.clear()

    # Model & Retrieval Settings
    st.markdown('<div class="sidebar-section-title">MODEL & RETRIEVAL</div>', unsafe_allow_html=True)
    
    selected_embedding = st.selectbox(
        "Embedding Model",
        options=[config.EMBEDDING_MODEL_A, config.EMBEDDING_MODEL_B],
        index=0,
        help="Hugging Face open-source sentence-transformers.",
    )
    
    selected_strategy = st.selectbox(
        "Retrieval Strategy",
        options=["hybrid", "dense", "mmr", "bm25", "rerank"],
        index=0,
        format_func=lambda x: {
            "hybrid": "Hybrid Search (BM25 + Dense RRF)",
            "dense": "Dense Cosine Similarity",
            "mmr": "MMR (Diversity)",
            "bm25": "BM25 Lexical",
            "rerank": "Cross-Encoder Reranker",
        }.get(x, x),
    )
    
    top_k_sources = st.slider("Top Sources (K)", min_value=1, max_value=5, value=3)

    # Documents & Index Metrics
    st.markdown('<div class="sidebar-section-title">DOCUMENTS</div>', unsafe_allow_html=True)
    
    try:
        vsm = get_vector_store(embedding_model_name=selected_embedding)
        chunk_count = vsm.get_document_count()
        indexed_papers = vsm.get_indexed_papers_summary()
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Papers", value=len(indexed_papers))
        with c2:
            st.metric(label="Chunks", value=chunk_count)
            
        with st.expander("📄 Indexed Papers List", expanded=False):
            for p in indexed_papers:
                st.markdown(f"• **{p['paper_title']}**  \n`{p['source_file']}` ({p['chunk_count']} chunks)")
    except Exception as e:
        st.caption(f"Index status: {str(e)}")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.selected_suggestion = None
        st.rerun()


# ==============================================================================
# 4. TOP HEADER BAR
# ==============================================================================
st.markdown("""
<div class="top-header">
    <div class="top-header-left">
        <div class="top-header-icon">📚</div>
        <div>
            <div class="top-header-title">Research Paper Answer Bot</div>
            <div class="top-header-subtitle">Ask questions across your scientific paper collection</div>
        </div>
    </div>
    <div class="status-badge">
        <div class="status-dot"></div>
        Gemini • RAG Active
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 5. MAIN NAVIGATION TABS
# ==============================================================================
tab_assistant, tab_analytics = st.tabs(["💬 Research Assistant", "📊 Corpus & Evaluation Dashboard"])

# ------------------------------------------------------------------------------
# TAB 1: RESEARCH ASSISTANT (CHAT & CITATIONS)
# ------------------------------------------------------------------------------
with tab_assistant:
    # Initialize RAG pipeline
    rag_chain = load_rag_pipeline(selected_embedding, selected_strategy, st.session_state.api_key)

    # Empty State (When no queries yet)
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="hero-card">
            <div class="hero-icon">🔍</div>
            <div class="hero-title">Research Paper Assistant</div>
            <div class="hero-subtitle">
                Explore your indexed research papers with grounded AI answers. Every answer is strictly attributed to supporting passages with exact page numbers.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-label">Suggested Questions</div>', unsafe_allow_html=True)
        
        suggested_queries = [
            "What is the computational complexity of self-attention per layer?",
            "How does LoRA freeze pre-trained weights while adding rank decomposition?",
            "What is the difference between RAG-Sequence and RAG-Token models?",
            "How does FlashAttention utilize GPU SRAM tiling to reduce HBM access?",
        ]
        
        cols = st.columns(2)
        for i, q_text in enumerate(suggested_queries):
            with cols[i % 2]:
                if st.button(f"💡 {q_text}", key=f"sug_{i}", use_container_width=True):
                    st.session_state.selected_suggestion = q_text
                    st.rerun()

    # Render Active Conversation
    for idx, chat in enumerate(st.session_state.chat_history):
        # 1. User Message Bubble
        st.markdown(f"""
        <div class="user-question-box">
            <div class="user-avatar">Q</div>
            <div class="user-text">{chat['question']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. AI Answer Display
        if chat.get("is_refusal"):
            st.markdown(f"""
            <div class="refusal-card">
                <span>⚠️</span>
                <div>{chat['answer']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            meta_info = f"Latency: {chat.get('latency', 0.0):.2f}s • Strategy: {chat.get('strategy', selected_strategy).upper()}"
            st.markdown(f"""
            <div class="ai-answer-card">
                <div class="ai-answer-header">
                    <div class="ai-answer-badge">✨ AI Answer</div>
                    <div class="ai-answer-meta">{meta_info}</div>
                </div>
                <div class="ai-answer-text">{chat['answer']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. Compact Supporting Source Cards
        sources = chat.get("sources", [])
        if sources:
            st.markdown("""
            <div class="sources-header">
                <div class="sources-title">Sources</div>
                <div class="sources-subtitle">Top supporting passages used to generate this answer</div>
            </div>
            """, unsafe_allow_html=True)
            
            for s_idx, src in enumerate(sources, 1):
                title = src.get("paper_title", "Research Paper")
                page = src.get("page_number", "N/A")
                passage = src.get("passage", "").strip()
                score = src.get("score")
                strategy_name = src.get("strategy", selected_strategy)
                
                score_str = f"Score: {score:.4f}" if score is not None else ""
                meta_str = f"Page {page} • {score_str} • Strategy: {strategy_name}"
                
                # Render source card with neat expander for full passage
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-top-row">
                        <div class="source-index-title">
                            <span class="source-num-badge">[{s_idx:02d}]</span>
                            <span class="source-paper-title">{title}</span>
                        </div>
                        <span class="source-page-badge">Page {page}</span>
                    </div>
                    <div class="source-passage-preview">"{passage[:220]}{'...' if len(passage) > 220 else ''}"</div>
                    <div class="source-meta-row">{meta_str}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if len(passage) > 220:
                    with st.expander(f"View full passage for Source [{s_idx:02d}]", expanded=False):
                        st.markdown(f"*{passage}*")
        
        st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

    # Input Bar
    input_query = st.chat_input("Ask anything about your research papers...")
    
    # Process Trigger from Input or Suggestion Button
    active_query = input_query or st.session_state.selected_suggestion
    if active_query:
        st.session_state.selected_suggestion = None  # Reset suggestion
        
        with st.spinner("Retrieving supporting passages and synthesizing grounded answer..."):
            response_data = rag_chain.query(
                question=active_query,
                strategy=selected_strategy,
                top_k=top_k_sources,
            )
            
        st.session_state.chat_history.append({
            "question": active_query,
            "answer": response_data["answer"],
            "sources": response_data["sources"],
            "is_refusal": response_data["is_refusal"],
            "latency": response_data["latency_seconds"],
            "strategy": selected_strategy,
        })
        st.rerun()


# ------------------------------------------------------------------------------
# TAB 2: CORPUS & EVALUATION DASHBOARD
# ------------------------------------------------------------------------------
with tab_analytics:
    st.markdown('<div class="section-label">Corpus & Index Overview</div>', unsafe_allow_html=True)
    
    vsm = get_vector_store(embedding_model_name=selected_embedding)
    total_docs = len(vsm.get_indexed_papers_summary())
    total_chunks = vsm.get_document_count()
    
    # Metric Overview Cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">{total_docs}</div>
            <div class="metric-card-lbl">Indexed Papers</div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">{total_chunks}</div>
            <div class="metric-card-lbl">Total Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">100%</div>
            <div class="metric-card-lbl">Context Recall (Hit@3)</div>
        </div>
        """, unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">0.95</div>
            <div class="metric-card-lbl">Mean Recip. Rank (MRR)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Benchmark & EDA Visualizations</div>', unsafe_allow_html=True)
    
    vis_dir = config.VISUALIZATIONS_DIR
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        p1 = vis_dir / "eda_1_paper_page_distribution.png"
        if p1.exists():
            st.image(str(p1), caption="Fig 1: Paper Corpus & Chunk Distribution")
            
        p4 = vis_dir / "eda_4_retrieval_strategy_comparison.png"
        if p4.exists():
            st.image(str(p4), caption="Fig 4: Retrieval Strategy Benchmark (Hit@1, Hit@3, MRR)")

    with col_v2:
        p2 = vis_dir / "eda_2_chunk_length_distribution.png"
        if p2.exists():
            st.image(str(p2), caption="Fig 2: Chunk Character Length Distribution")
            
        p3 = vis_dir / "eda_3_embedding_similarity_distribution.png"
        if p3.exists():
            st.image(str(p3), caption="Fig 3: Embedding Cosine Similarity Distributions")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Benchmark Comparison Tables</div>', unsafe_allow_html=True)
    
    exp_emb = config.EXPERIMENTS_DIR / "embedding_comparison.csv"
    if exp_emb.exists():
        import pandas as pd
        st.markdown("##### 🔬 Embedding Models Comparison")
        st.dataframe(pd.read_csv(exp_emb))
        
    exp_ret = config.EXPERIMENTS_DIR / "retrieval_comparison.csv"
    if exp_ret.exists():
        import pandas as pd
        st.markdown("##### ⚡ Retrieval Strategy Benchmark")
        st.dataframe(pd.read_csv(exp_ret))
