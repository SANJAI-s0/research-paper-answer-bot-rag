"""
Generate visual system architecture, flowchart, and UI design diagrams.
Saved to outputs/visualizations/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

out_dir = Path("outputs/visualizations")
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'

# ==============================================================================
# 1. SYSTEM ARCHITECTURE & FLOWCHART DIAGRAM
# ==============================================================================
def generate_architecture_flowchart():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(7, 6.6, "Research Paper Answer Bot — System Architecture & Retrieval Flowchart", 
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A')
    
    # 4 Main Phase Containers
    phases = [
        ("PHASE 1: INGESTION & CHUNKING", 0.5, 0.6, 3.8, 5.5, "#EFF6FF", "#3B82F6"),
        ("PHASE 2: DUAL-INDEXING", 4.6, 0.6, 4.2, 5.5, "#F0FDF4", "#10B981"),
        ("PHASE 3: HYBRID RETRIEVAL", 9.1, 0.6, 4.4, 5.5, "#EEF2FF", "#6366F1"),
    ]
    for title, x, y, w, h, bg, border in phases:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      facecolor=bg, edgecolor=border, linewidth=1.5, alpha=0.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.25, title, ha='center', va='center', 
                fontsize=9.5, fontweight='bold', color=border)

    # Box Helper
    def draw_box(x, y, w, h, title, subtitle, bg="#FFFFFF", border="#CBD5E1", text_c="#0F172A"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                                     facecolor=bg, edgecolor=border, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', 
                fontsize=9.5, fontweight='bold', color=text_c)
        ax.text(x + w/2, y + h*0.3, subtitle, ha='center', va='center', 
                fontsize=7.8, color="#64748B")

    # Flow Elements
    # Phase 1
    draw_box(0.8, 4.4, 3.2, 1.1, "10 Seminal AI Papers", "109 Pages PDF Corpus", bg="#FFFFFF", border="#93C5FD")
    draw_box(0.8, 2.7, 3.2, 1.1, "PyPDF Document Loader", "Extracts Text + Title + Page #", bg="#FFFFFF", border="#93C5FD")
    draw_box(0.8, 1.0, 3.2, 1.1, "Recursive Text Splitter", "600 Chars / 100 Overlap (707 Chunks)", bg="#FFFFFF", border="#93C5FD")

    # Phase 2
    draw_box(4.9, 4.4, 3.6, 1.1, "Sentence-Transformers", "all-MiniLM / BGE-small (384-D)", bg="#FFFFFF", border="#86EFAC")
    draw_box(4.9, 2.7, 3.6, 1.1, "Persistent ChromaDB", "Vector Collection + Cosine Index", bg="#FFFFFF", border="#86EFAC")
    draw_box(4.9, 1.0, 3.6, 1.1, "In-Memory BM25 Index", "Exact Keyword & Equation Tokens", bg="#FFFFFF", border="#86EFAC")

    # Phase 3
    draw_box(9.4, 4.4, 3.8, 1.1, "User Natural Query", "e.g. 'Complexity of Self-Attention'", bg="#FFFFFF", border="#C7D2FE")
    draw_box(9.4, 2.7, 3.8, 1.1, "Reciprocal Rank Fusion (RRF)", "Hybrid Score = Sum 1 / (60 + Rank)", bg="#FFFFFF", border="#C7D2FE")
    draw_box(9.4, 1.0, 3.8, 1.1, "Google Gemini LLM (Temp 0.0)", "Zero-Hallucination + Top 3 Sources", bg="#4F46E5", border="#4338CA", text_c="#FFFFFF")

    # Connectors
    arrows = [
        ((2.4, 4.4), (2.4, 3.8)),
        ((2.4, 2.7), (2.4, 2.1)),
        ((4.0, 1.55), (4.9, 1.55)),
        ((4.0, 1.55), (4.9, 4.95)),
        ((6.7, 4.4), (6.7, 3.8)),
        ((8.5, 3.25), (9.4, 3.25)),
        ((8.5, 1.55), (9.4, 3.0)),
        ((11.3, 4.4), (11.3, 3.8)),
        ((11.3, 2.7), (11.3, 2.1)),
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=6, shrink=0.05))

    plt.tight_layout()
    fig.savefig(out_dir / "system_architecture_diagram.png", bbox_inches='tight')
    plt.close()
    print("Generated system_architecture_diagram.png")


# ==============================================================================
# 2. UI DESIGN LAYOUT DIAGRAM
# ==============================================================================
def generate_ui_design_layout():
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    
    ax.text(7, 7.1, "Streamlit UI Architecture & Component Hierarchy", 
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A')

    # Sidebar Area
    sb = patches.FancyBboxPatch((0.5, 0.6), 3.4, 6.0, boxstyle="round,pad=0.08", 
                                facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5)
    ax.add_patch(sb)
    ax.text(2.2, 6.2, "SIDEBAR CONTROLS", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    
    sb_elements = [
        ("SYSTEM", "Gemini API Key • Connected Badge", 5.2),
        ("MODEL & RETRIEVAL", "Embedding Model Selector\nStrategy: Hybrid / MMR / BM25\nTop-K Slider (1 to 5)", 3.9),
        ("DOCUMENTS", "10 Papers • 707 Chunks\nCollapsible Paper Index List", 2.4),
        ("ACTIONS", "Clear Conversation Button", 1.2),
    ]
    for title, desc, y_pos in sb_elements:
        c = patches.FancyBboxPatch((0.8, y_pos - 0.4), 2.8, 0.9, boxstyle="round,pad=0.05", 
                                   facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=1)
        ax.add_patch(c)
        ax.text(2.2, y_pos + 0.25, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2563EB')
        ax.text(2.2, y_pos - 0.05, desc, ha='center', va='center', fontsize=7.2, color='#475569')

    # Main Area
    mb = patches.FancyBboxPatch((4.2, 0.6), 9.3, 6.0, boxstyle="round,pad=0.08", 
                                facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5)
    ax.add_patch(mb)

    # Top Header
    th = patches.FancyBboxPatch((4.5, 5.7), 8.7, 0.7, boxstyle="round,pad=0.05", 
                                facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1)
    ax.add_patch(th)
    ax.text(4.8, 6.05, "[App] Research Paper Answer Bot", fontsize=10.5, fontweight='bold', color='#0F172A', va='center')
    ax.text(12.7, 6.05, "* Gemini | RAG Active", fontsize=8.5, fontweight='bold', color='#059669', va='center', ha='right')

    # Tab Bar
    tb = patches.FancyBboxPatch((4.5, 5.0), 8.7, 0.5, boxstyle="round,pad=0.04", 
                                facecolor="#EEF2FF", edgecolor="#C7D2FE", linewidth=1)
    ax.add_patch(tb)
    ax.text(6.0, 5.25, "[Tab 1: Research Assistant]", fontsize=9, fontweight='bold', color='#4338CA', ha='center', va='center')
    ax.text(9.8, 5.25, "[Tab 2: Corpus & Evaluation Dashboard]", fontsize=9, fontweight='bold', color='#64748B', ha='center', va='center')

    # User Query Card
    uq = patches.FancyBboxPatch((4.5, 4.1), 8.7, 0.65, boxstyle="round,pad=0.04", 
                                facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1)
    ax.add_patch(uq)
    ax.text(4.7, 4.42, "Q: What is the computational complexity of self-attention per layer?", fontsize=9, fontweight='bold', color='#1E293B', va='center')

    # AI Answer Card
    ans = patches.FancyBboxPatch((4.5, 2.7), 8.7, 1.2, boxstyle="round,pad=0.04", 
                                 facecolor="#FFFFFF", edgecolor="#93C5FD", linewidth=1.2)
    ax.add_patch(ans)
    ax.text(4.7, 3.65, "AI ANSWER", fontsize=8.5, fontweight='bold', color='#2563EB', va='center')
    ax.text(12.7, 3.65, "Latency: 0.38s • Strategy: HYBRID", fontsize=7.5, color='#64748B', va='center', ha='right')
    ax.text(4.7, 3.15, "The computational complexity of self-attention per layer is O(n^2 * d) [attention_is_all_you_need.pdf, Page 4].", 
            fontsize=8.8, color='#0F172A', va='center')

    # Supporting Sources
    src = patches.FancyBboxPatch((4.5, 0.9), 8.7, 1.6, boxstyle="round,pad=0.04", 
                                 facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1)
    ax.add_patch(src)
    ax.text(4.7, 2.25, "SOURCES • Top Supporting Passages", fontsize=8.5, fontweight='bold', color='#0F172A', va='center')
    ax.text(4.7, 1.85, "[01] Attention Is All You Need                      Page 4 | Score: 0.0325", fontsize=8.2, fontweight='bold', color='#4338CA', va='center')
    ax.text(4.7, 1.45, "\"Computational Complexity Per Layer: Self-Attention: O(n^2 * d), sequential operations O(1)...\"", 
            fontsize=7.8, color='#334155', style='italic', va='center')
    ax.text(4.7, 1.1, "▶ View full passage [Expander] • Page 4 • Strategy: Hybrid (RRF)", fontsize=7.2, color='#64748B', va='center')

    plt.tight_layout()
    fig.savefig(out_dir / "ui_design_layout.png", bbox_inches='tight')
    plt.close()
    print("Generated ui_design_layout.png")


if __name__ == "__main__":
    generate_architecture_flowchart()
    generate_ui_design_layout()
