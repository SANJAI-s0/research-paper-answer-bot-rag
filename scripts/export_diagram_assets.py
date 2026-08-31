"""
Comprehensive Diagram Generator and Exporter.
Generates publication-quality .SVG and .PNG graphics for all system design,
flowchart, and architecture diagrams in:
- docs/Flow/
- docs/architecture/
- docs/designs/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Create directories
flow_dir = Path("docs/Flow")
arch_dir = Path("docs/architecture")
des_dir = Path("docs/designs")

for d in [flow_dir, arch_dir, des_dir]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'


def save_fig_dual(fig, base_path: Path):
    """Save both SVG and PNG versions of a figure."""
    plt.tight_layout()
    fig.savefig(base_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(base_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Exported: {base_path.name}.png and {base_path.name}.svg")


# ==============================================================================
# 1. FLOW DIAGRAMS
# ==============================================================================
def export_end_to_end_flow():
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(6.5, 5.6, "End-to-End Grounded RAG Pipeline Flow", ha='center', fontsize=14, fontweight='bold', color='#0F172A')

    boxes = [
        (0.5, 2.2, 2.2, 2.0, "User Query", "Text input / Suggestion\n(e.g. Self-attention complexity)", "#EFF6FF", "#3B82F6"),
        (3.2, 2.2, 2.8, 2.0, "Hybrid Retrieval", "ChromaDB (Dense) +\nBM25 (Sparse) via RRF", "#F0FDF4", "#10B981"),
        (6.5, 2.2, 2.8, 2.0, "Grounded Prompt", "Injected Top-3 Excerpts +\nMandatory Refusal Rule", "#FEF3C7", "#D97706"),
        (9.8, 2.2, 2.7, 2.0, "Gemini LLM (0.0)", "Greedy Factual Synthesis +\nTop-3 Citation Cards", "#EEF2FF", "#6366F1"),
    ]
    for x, y, w, h, t, sub, bg, border in boxes:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*0.7, t, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0F172A')
        ax.text(x + w/2, y + h*0.35, sub, ha='center', va='center', fontsize=8.5, color='#475569')

    arrows = [((2.7, 3.2), (3.2, 3.2)), ((6.0, 3.2), (6.5, 3.2)), ((9.3, 3.2), (9.8, 3.2))]
    for s, e in arrows:
        ax.annotate('', xy=e, xytext=s, arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=2, headwidth=7))

    save_fig_dual(fig, flow_dir / "end_to_end_flow")


def export_frontend_flow():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(6.0, 5.6, "Streamlit Frontend UI State Flow", ha='center', fontsize=14, fontweight='bold', color='#0F172A')

    steps = [
        (0.5, 2.2, 2.3, 2.0, "App Loaded", "Check Session State\nRender Top Header & Sidebar", "#F8FAFC", "#94A3B8"),
        (3.3, 2.2, 2.4, 2.0, "Empty / Active State", "Suggested Query Pills or\nPrior Chat History", "#EEF2FF", "#6366F1"),
        (6.2, 2.2, 2.4, 2.0, "RAG Execution", "Show UI Spinner\nCall rag_chain.query()", "#FEF3C7", "#D97706"),
        (9.1, 2.2, 2.4, 2.0, "Render Output", "Display AI Answer Card +\n3 Source Citation Cards", "#F0FDF4", "#10B981"),
    ]
    for x, y, w, h, t, sub, bg, border in steps:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*0.7, t, ha='center', va='center', fontsize=10, fontweight='bold', color='#0F172A')
        ax.text(x + w/2, y + h*0.35, sub, ha='center', va='center', fontsize=8.2, color='#475569')

    arrows = [((2.8, 3.2), (3.3, 3.2)), ((5.7, 3.2), (6.2, 3.2)), ((8.6, 3.2), (9.1, 3.2))]
    for s, e in arrows:
        ax.annotate('', xy=e, xytext=s, arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=2, headwidth=7))

    save_fig_dual(fig, flow_dir / "frontend_flow")


def export_backend_flow():
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis('off')
    ax.text(6.5, 6.0, "Backend Processing Sequence & Retrieval Pipeline", ha='center', fontsize=14, fontweight='bold', color='#0F172A')

    nodes = [
        (0.6, 3.8, 3.2, 1.6, "1. Ingestion & Splitter", "PyPDF Loader -> Metadata Engine\nRecursive Split (600c / 100ov)", "#EFF6FF", "#3B82F6"),
        (4.8, 3.8, 3.4, 1.6, "2. Dual Vector/BM25 Index", "Sentence-Transformers (384-D)\nChromaDB + In-Memory BM25", "#F0FDF4", "#10B981"),
        (9.2, 3.8, 3.2, 1.6, "3. Query Dispatcher", "Dispatches Query to Dense &\nSparse Retrievers in parallel", "#FEF3C7", "#D97706"),
        (2.8, 1.0, 3.4, 1.6, "4. Reciprocal Rank Fusion", "RRF Score = sum 1 / (60 + Rank)\nSelects Top 3 Grounded Passages", "#EEF2FF", "#6366F1"),
        (7.2, 1.0, 3.4, 1.6, "5. Grounded LLM Synthesis", "Google Gemini (gemini-3.7-flash)\nTemperature 0.0 -> Grounded Answer", "#FAF5FF", "#A855F7"),
    ]
    for x, y, w, h, t, sub, bg, border in nodes:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*0.7, t, ha='center', va='center', fontsize=10, fontweight='bold', color='#0F172A')
        ax.text(x + w/2, y + h*0.35, sub, ha='center', va='center', fontsize=8.2, color='#475569')

    arrows = [
        ((3.8, 4.6), (4.8, 4.6)),
        ((8.2, 4.6), (9.2, 4.6)),
        ((10.8, 3.8), (5.5, 2.6)),
        ((6.2, 1.8), (7.2, 1.8)),
    ]
    for s, e in arrows:
        ax.annotate('', xy=e, xytext=s, arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=2, headwidth=7))

    save_fig_dual(fig, flow_dir / "backend_flow")


def export_model_flow():
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(6.5, 5.6, "Embedding & Generative Model Data Flow", ha='center', fontsize=14, fontweight='bold', color='#0F172A')

    m_boxes = [
        (0.5, 2.0, 3.5, 2.5, "Embedding Model Flow", "Model: all-MiniLM-L6-v2\nInput: Text Passages (600 chars)\nOutput: 384-dimensional Vectors\nDistance: Cosine Similarity in HNSW", "#EFF6FF", "#3B82F6"),
        (4.7, 2.0, 3.5, 2.5, "BM25 Scoring Flow", "Algorithm: Okapi BM25\nTokenizer: Regex alphanumeric\nWeights: k1=1.5, b=0.75\nOutput: Lexical Relevance Scores", "#F0FDF4", "#10B981"),
        (8.9, 2.0, 3.6, 2.5, "LLM Synthesis Flow", "Provider: Google Gemini API\nModel: gemini-3.7-flash\nTemperature: 0.0 (Deterministic)\nGuardrail: Strict Refusal on Unknowns", "#EEF2FF", "#6366F1"),
    ]
    for x, y, w, h, t, sub, bg, border in m_boxes:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*0.78, t, ha='center', va='center', fontsize=11, fontweight='bold', color='#0F172A')
        ax.text(x + w/2, y + h*0.42, sub, ha='center', va='center', fontsize=8.5, color='#475569')

    save_fig_dual(fig, flow_dir / "model_flow")


# ==============================================================================
# 2. ARCHITECTURE DIAGRAMS
# ==============================================================================
def export_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
    ax.text(7, 6.6, "System Architecture: End-to-End RAG Pipeline", ha='center', fontsize=15, fontweight='bold', color='#0F172A')

    phases = [
        ("STAGE 1: INGESTION", 0.5, 0.6, 3.8, 5.5, "#EFF6FF", "#3B82F6"),
        ("STAGE 2: DUAL INDEX", 4.6, 0.6, 4.2, 5.5, "#F0FDF4", "#10B981"),
        ("STAGE 3: RETRIEVAL & LLM", 9.1, 0.6, 4.4, 5.5, "#EEF2FF", "#6366F1"),
    ]
    for title, x, y, w, h, bg, border in phases:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", facecolor=bg, edgecolor=border, linewidth=1.5, alpha=0.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.25, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=border)

    def draw_box(x, y, w, h, title, subtitle, bg="#FFFFFF", border="#CBD5E1"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=border, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color="#0F172A")
        ax.text(x + w/2, y + h*0.3, subtitle, ha='center', va='center', fontsize=7.8, color="#64748B")

    draw_box(0.8, 4.4, 3.2, 1.1, "10 Seminal AI Papers", "109 Pages PDF Corpus", border="#93C5FD")
    draw_box(0.8, 2.7, 3.2, 1.1, "PyPDF Document Loader", "Extracts Text + Title + Page #", border="#93C5FD")
    draw_box(0.8, 1.0, 3.2, 1.1, "Recursive Text Splitter", "600 Chars / 100 Overlap (707 Chunks)", border="#93C5FD")

    draw_box(4.9, 4.4, 3.6, 1.1, "Sentence-Transformers", "all-MiniLM / BGE-small (384-D)", border="#86EFAC")
    draw_box(4.9, 2.7, 3.6, 1.1, "Persistent ChromaDB", "Vector Collection + Cosine Index", border="#86EFAC")
    draw_box(4.9, 1.0, 3.6, 1.1, "In-Memory BM25 Index", "Exact Keyword & Equation Tokens", border="#86EFAC")

    draw_box(9.4, 4.4, 3.8, 1.1, "User Natural Query", "e.g. 'Complexity of Self-Attention'", border="#C7D2FE")
    draw_box(9.4, 2.7, 3.8, 1.1, "Reciprocal Rank Fusion", "Hybrid Score = Sum 1 / (60 + Rank)", border="#C7D2FE")
    draw_box(9.4, 1.0, 3.8, 1.1, "Google Gemini LLM", "Temperature 0.0 -> Grounded Answer", border="#C7D2FE")

    arrows = [
        ((2.4, 4.4), (2.4, 3.8)), ((2.4, 2.7), (2.4, 2.1)),
        ((4.0, 1.55), (4.9, 1.55)), ((4.0, 1.55), (4.9, 4.95)),
        ((6.7, 4.4), (6.7, 3.8)),
        ((8.5, 3.25), (9.4, 3.25)), ((8.5, 1.55), (9.4, 3.0)),
        ((11.3, 4.4), (11.3, 3.8)), ((11.3, 2.7), (11.3, 2.1)),
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=6, shrink=0.05))

    save_fig_dual(fig, arch_dir / "system_architecture")


def export_dual_index_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.5); ax.axis('off')
    ax.text(6.0, 6.0, "Dual-Index Architecture: Dense Vector & Sparse Lexical", ha='center', fontsize=14, fontweight='bold', color='#0F172A')

    # Top Chunks
    r_top = patches.FancyBboxPatch((3.5, 4.6), 5.0, 1.0, boxstyle="round,pad=0.08", facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.5)
    ax.add_patch(r_top)
    ax.text(6.0, 5.25, "707 Document Chunks", ha='center', fontsize=11, fontweight='bold', color='#0F172A')
    ax.text(6.0, 4.85, "Preserves: Title, Page #, Chunk ID, Chunk Index", ha='center', fontsize=8.5, color='#475569')

    # Left Dense
    r_dense = patches.FancyBboxPatch((0.8, 1.2), 4.8, 2.8, boxstyle="round,pad=0.08", facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.5)
    ax.add_patch(r_dense)
    ax.text(3.2, 3.6, "Dense Semantic Branch", ha='center', fontsize=11, fontweight='bold', color='#1E40AF')
    ax.text(3.2, 2.5, "• sentence-transformers/all-MiniLM-L6-v2\n• 384-dimensional continuous space\n• Persistent ChromaDB storage\n• HNSW Cosine vector search", ha='center', fontsize=8.8, color='#1E293B')

    # Right Sparse
    r_sparse = patches.FancyBboxPatch((6.4, 1.2), 4.8, 2.8, boxstyle="round,pad=0.08", facecolor="#F0FDF4", edgecolor="#10B981", linewidth=1.5)
    ax.add_patch(r_sparse)
    ax.text(8.8, 3.6, "Sparse Lexical Branch", ha='center', fontsize=11, fontweight='bold', color='#065F46')
    ax.text(8.8, 2.5, "• In-Memory Rank-BM25 Engine\n• Regex Alphanumeric Tokenizer\n• TF-IDF weighting (k1=1.5, b=0.75)\n• Exact keyword & equation matching", ha='center', fontsize=8.8, color='#1E293B')

    ax.annotate('', xy=(3.2, 4.0), xytext=(5.0, 4.6), arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=6))
    ax.annotate('', xy=(8.8, 4.0), xytext=(7.0, 4.6), arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=6))

    save_fig_dual(fig, arch_dir / "dual_index_architecture")


def export_hybrid_retrieval_architecture():
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(6.5, 5.6, "Hybrid Retrieval & Reciprocal Rank Fusion (RRF)", ha='center', fontsize=14, fontweight='bold', color='#0F172A')

    cards = [
        (0.5, 1.5, 3.5, 3.2, "Dense Retrieval", "ChromaDB Cosine Search\nReturns Top 10 Ranked Chunks\nCaptures broad conceptual intent", "#EFF6FF", "#3B82F6"),
        (4.7, 1.5, 3.6, 3.2, "Reciprocal Rank Fusion", "RRF(d) = sum 1 / (60 + Rank)\nk = 60 Smoothing Constant\nFuses ranking positions directly", "#FEF3C7", "#D97706"),
        (8.9, 1.5, 3.6, 3.2, "Sparse BM25 Retrieval", "BM25 Inverted Index Match\nReturns Top 10 Ranked Chunks\nCaptures exact formulas / terms", "#F0FDF4", "#10B981"),
    ]
    for x, y, w, h, t, sub, bg, border in cards:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*0.78, t, ha='center', va='center', fontsize=11, fontweight='bold', color='#0F172A')
        ax.text(x + w/2, y + h*0.42, sub, ha='center', va='center', fontsize=8.5, color='#475569')

    ax.annotate('', xy=(4.7, 3.1), xytext=(4.0, 3.1), arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=2, headwidth=7))
    ax.annotate('', xy=(8.3, 3.1), xytext=(8.9, 3.1), arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=2, headwidth=7))

    save_fig_dual(fig, arch_dir / "hybrid_retrieval_architecture")


# ==============================================================================
# 3. DESIGN DIAGRAMS
# ==============================================================================
def export_rag_design():
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis('off')
    ax.text(6.5, 6.0, "Grounded RAG System Design & Prompt Guardrails", ha='center', fontsize=14, fontweight='bold', color='#0F172A')

    sections = [
        (0.6, 1.2, 3.6, 4.0, "Context Injection", "• Top-3 retrieved passages\n• Exact paper title & page #\n• Wrapped in structured XML:\n  <source id='1' title='...'>\n    Passage content...\n  </source>", "#EFF6FF", "#3B82F6"),
        (4.7, 1.2, 3.6, 4.0, "Grounded Guardrail", "• Persona: Academic Assistant\n• Strict Rule 1: Use ONLY context\n• Strict Rule 2: Mandatory Refusal:\n  'I don't know based on...'\n• Temperature: 0.0 (Greedy)", "#FEF3C7", "#D97706"),
        (8.8, 1.2, 3.6, 4.0, "Verified Output", "• Grounded Synthesis\n• Page-level citation tokens\n• Top-3 Supporting Source Cards\n• Zero Hallucination Guarantee", "#F0FDF4", "#10B981"),
    ]
    for x, y, w, h, t, sub, bg, border in sections:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*0.82, t, ha='center', va='center', fontsize=11, fontweight='bold', color='#0F172A')
        ax.text(x + w/2, y + h*0.45, sub, ha='center', va='center', fontsize=8.8, color='#1E293B')

    arrows = [((4.2, 3.2), (4.7, 3.2)), ((8.3, 3.2), (8.8, 3.2))]
    for s, e in arrows:
        ax.annotate('', xy=e, xytext=s, arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=2, headwidth=7))

    save_fig_dual(fig, des_dir / "rag_design")


def export_ui_component_design():
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.5); ax.axis('off')
    ax.text(7, 7.1, "Streamlit UI Component Design & Layout", ha='center', fontsize=15, fontweight='bold', color='#0F172A')

    # Sidebar Area
    sb = patches.FancyBboxPatch((0.5, 0.6), 3.4, 6.0, boxstyle="round,pad=0.08", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5)
    ax.add_patch(sb)
    ax.text(2.2, 6.2, "SIDEBAR CONTROLS", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')

    sb_elements = [
        ("SYSTEM", "Gemini API Key • Connected Badge", 5.2),
        ("MODEL & RETRIEVAL", "Embedding Model Selector\nStrategy: Hybrid / MMR / BM25\nTop-K Slider (1 to 5)", 3.9),
        ("DOCUMENTS", "10 Papers • 707 Chunks\nCollapsible Paper Index List", 2.4),
        ("ACTIONS", "Clear Conversation Button", 1.2),
    ]
    for title, desc, y_pos in sb_elements:
        c = patches.FancyBboxPatch((0.8, y_pos - 0.4), 2.8, 0.9, boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=1)
        ax.add_patch(c)
        ax.text(2.2, y_pos + 0.25, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2563EB')
        ax.text(2.2, y_pos - 0.05, desc, ha='center', va='center', fontsize=7.2, color='#475569')

    # Main Area
    mb = patches.FancyBboxPatch((4.2, 0.6), 9.3, 6.0, boxstyle="round,pad=0.08", facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5)
    ax.add_patch(mb)

    th = patches.FancyBboxPatch((4.5, 5.7), 8.7, 0.7, boxstyle="round,pad=0.05", facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1)
    ax.add_patch(th)
    ax.text(4.8, 6.05, "[App] Research Paper Answer Bot", fontsize=10.5, fontweight='bold', color='#0F172A', va='center')
    ax.text(12.7, 6.05, "* Gemini | RAG Active", fontsize=8.5, fontweight='bold', color='#059669', va='center', ha='right')

    tb = patches.FancyBboxPatch((4.5, 5.0), 8.7, 0.5, boxstyle="round,pad=0.04", facecolor="#EEF2FF", edgecolor="#C7D2FE", linewidth=1)
    ax.add_patch(tb)
    ax.text(6.0, 5.25, "[Tab 1: Research Assistant]", fontsize=9, fontweight='bold', color='#4338CA', ha='center', va='center')
    ax.text(9.8, 5.25, "[Tab 2: Corpus & Evaluation Dashboard]", fontsize=9, fontweight='bold', color='#64748B', ha='center', va='center')

    uq = patches.FancyBboxPatch((4.5, 4.1), 8.7, 0.65, boxstyle="round,pad=0.04", facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1)
    ax.add_patch(uq)
    ax.text(4.7, 4.42, "Q: What is the computational complexity of self-attention per layer?", fontsize=9, fontweight='bold', color='#1E293B', va='center')

    ans = patches.FancyBboxPatch((4.5, 2.7), 8.7, 1.2, boxstyle="round,pad=0.04", facecolor="#FFFFFF", edgecolor="#93C5FD", linewidth=1.2)
    ax.add_patch(ans)
    ax.text(4.7, 3.65, "AI ANSWER", fontsize=8.5, fontweight='bold', color='#2563EB', va='center')
    ax.text(12.7, 3.65, "Latency: 0.38s • Strategy: HYBRID", fontsize=7.5, color='#64748B', va='center', ha='right')
    ax.text(4.7, 3.15, "The computational complexity of self-attention per layer is O(n^2 * d) [attention_is_all_you_need.pdf, Page 4].", 
            fontsize=8.8, color='#0F172A', va='center')

    src = patches.FancyBboxPatch((4.5, 0.9), 8.7, 1.6, boxstyle="round,pad=0.04", facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1)
    ax.add_patch(src)
    ax.text(4.7, 2.25, "SOURCES • Top Supporting Passages", fontsize=8.5, fontweight='bold', color='#0F172A', va='center')
    ax.text(4.7, 1.85, "[01] Attention Is All You Need                      Page 4 | Score: 0.0325", fontsize=8.2, fontweight='bold', color='#4338CA', va='center')
    ax.text(4.7, 1.45, "\"Computational Complexity Per Layer: Self-Attention: O(n^2 * d), sequential operations O(1)...\"", 
            fontsize=7.8, color='#334155', style='italic', va='center')
    ax.text(4.7, 1.1, "▶ View full passage [Expander] • Page 4 • Strategy: Hybrid (RRF)", fontsize=7.2, color='#64748B', va='center')

    save_fig_dual(fig, des_dir / "ui_component_design")


def export_data_model_design():
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis('off')
    ax.text(6.5, 6.0, "Metadata Schema & Data Model Provenance", ha='center', fontsize=14, fontweight='bold', color='#0F172A')

    entities = [
        (0.6, 1.2, 3.6, 4.2, "DOCUMENT PAGE (PyPDF)", "• page_content: str\n• paper_title: str\n• source_file: str\n• page_number: int (1..N)\n• total_pages: int\n\nPreserves page boundaries\nand document provenance.", "#EFF6FF", "#3B82F6"),
        (4.7, 1.2, 3.6, 4.2, "DOCUMENT CHUNK (Splitter)", "• page_content: str (600c)\n• chunk_id: str (unique hash)\n• chunk_index: int (0..M)\n• paper_title: str\n• source_file: str\n• page_number: int\n• total_pages: int", "#F0FDF4", "#10B981"),
        (8.8, 1.2, 3.6, 4.2, "SUPPORTING CITATION (UI)", "• rank: int (1, 2, 3)\n• paper_title: str\n• page_number: int\n• passage: str\n• score: float (RRF Score)\n• strategy: str ('hybrid')\n\nRendered in UI source cards.", "#EEF2FF", "#6366F1"),
    ]
    for x, y, w, h, t, sub, bg, border in entities:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*0.84, t, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0F172A')
        ax.text(x + w/2, y + h*0.45, sub, ha='center', va='center', fontsize=8.5, color='#1E293B')

    arrows = [((4.2, 3.3), (4.7, 3.3)), ((8.3, 3.3), (8.8, 3.3))]
    for s, e in arrows:
        ax.annotate('', xy=e, xytext=s, arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=2, headwidth=7))

    save_fig_dual(fig, des_dir / "data_model_design")


if __name__ == "__main__":
    print("Exporting all Flow diagrams...")
    export_end_to_end_flow()
    export_frontend_flow()
    export_backend_flow()
    export_model_flow()

    print("\nExporting all Architecture diagrams...")
    export_system_architecture()
    export_dual_index_architecture()
    export_hybrid_retrieval_architecture()

    print("\nExporting all Design diagrams...")
    export_rag_design()
    export_ui_component_design()
    export_data_model_design()
    print("\nAll .svg, .png, and .mmd diagram assets exported successfully!")
