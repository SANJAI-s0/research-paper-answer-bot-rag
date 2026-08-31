"""
Data Ingestion Script.
Downloads or generates curated seminal GenAI research paper PDFs in `data/papers/`.
Papers included:
1. Attention Is All You Need (Vaswani et al., 2017)
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
3. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
4. FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)
5. Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

from src.utils import get_logger
from src.config import config
from src.document_loader import load_all_papers

logger = get_logger("Ingest")

PAPERS_CONTENT = [
    {
        "filename": "attention_is_all_you_need.pdf",
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin",
        "pages": [
            # Page 1: Abstract & Intro
            """<b>Attention Is All You Need</b><br/>
            <i>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin</i><br/><br/>
            <b>Abstract</b><br/>
            The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.<br/><br/>
            <b>1. Introduction</b><br/>
            Recurrent neural networks, particularly long short-term memory (LSTM) and gated recurrent (GRU) neural networks, have been firmly established as state of the art approaches in sequence modeling. Recurrent models inherently factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time generates a sequence of hidden states h_t, as a function of the previous hidden state h_{t-1} and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths.""",
            
            # Page 2: Model Architecture
            """<b>2. Model Architecture</b><br/>
            The Transformer follows the overall encoder-decoder architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.<br/><br/>
            <b>Encoder:</b> The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism, and a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512.<br/><br/>
            <b>Decoder:</b> The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each sub-layer, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking ensures that predictions for position i can depend only on known outputs at positions less than i.""",

            # Page 3: Attention Mechanism Details
            """<b>3. Attention Mechanisms</b><br/>
            An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.<br/><br/>
            <b>Scaled Dot-Product Attention:</b> We compute the dot products of the query with all keys, divide each by sqrt(d_k), and apply a softmax function to obtain the weights on the values:<br/>
            <i>Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V</i><br/><br/>
            <b>Multi-Head Attention:</b> Instead of performing a single attention function with d_model-dimensional queries, keys and values, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding d_v-dimensional output values. These are concatenated and once again projected.<br/>
            <i>MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O</i> where <i>head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)</i>.<br/>
            In this work we employ h = 8 parallel attention layers, or heads. For each of these we use d_k = d_v = d_model / h = 64.""",

            # Page 4: Positional Encoding & Complexity Comparison
            """<b>4. Positional Encoding & Complexity Analysis</b><br/>
            Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add 'positional encodings' to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed.<br/><br/>
            In this work, we use sine and cosine functions of different frequencies:<br/>
            <i>PE_(pos, 2i) = sin(pos / 10000^(2i/d_model))</i><br/>
            <i>PE_(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</i><br/>
            where pos is the position and i is the dimension.<br/><br/>
            <b>Computational Complexity Per Layer:</b><br/>
            - <b>Self-Attention:</b> Complexity per layer is O(n^2 * d), sequential operations O(1), maximum path length O(1).<br/>
            - <b>Recurrent Layer:</b> Complexity per layer is O(n * d^2), sequential operations O(n), maximum path length O(n).<br/>
            - <b>Convolutional Layer:</b> Complexity per layer is O(k * n * d^2), sequential operations O(1), maximum path length O(log_k(n)).<br/>
            When sequence length n is smaller than representation dimension d, self-attention layers are faster than recurrent layers."""
        ]
    },
    {
        "filename": "rag_retrieval_augmented_generation.pdf",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela",
        "pages": [
            # Page 1: Abstract & Concept
            """<b>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</b><br/>
            <i>Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (NeurIPS 2020)</i><br/><br/>
            <b>Abstract</b><br/>
            Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and they struggle on knowledge-intensive tasks. In this work, we explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation.<br/><br/>
            <b>1. Methods Overview</b><br/>
            RAG models combine a pre-trained sequence-to-sequence model (parametric memory) with a dense vector index of Wikipedia (non-parametric memory) accessed via a neural retriever. We leverage Dense Passage Retrieval (DPR) to provide latent document retrieval, and BART as the generator.""",

            # Page 2: RAG-Sequence vs RAG-Token
            """<b>2. RAG Architectures: RAG-Sequence vs. RAG-Token</b><br/>
            We propose two formulations of RAG models that differ in how the retrieved documents are marginalized across generated output tokens:<br/><br/>
            <b>RAG-Sequence Model:</b> The RAG-Sequence model uses the same retrieved document to generate the complete sequence. Formally, it treats the retrieved document as a single latent variable that is marginalized to get the seq2seq probability p(y|x) via a top-K approximation:<br/>
            <i>p_RAG-Sequence(y|x) = sum_{z in top-k(p(z|x))} p_eta(z|x) * prod_{i=1}^N p_theta(y_i | x, z, y_{1:i-1})</i><br/><br/>
            <b>RAG-Token Model:</b> The RAG-Token model can draw from different latent documents for each target token. This allows the generator to synthesize content from multiple documents when generating an answer:<br/>
            <i>p_RAG-Token(y|x) = prod_{i=1}^N sum_{z in top-k(p(z|x))} p_eta(z|x) * p_theta(y_i | x, z, y_{1:i-1})</i><br/><br/>
            <b>Non-Parametric Retriever (DPR):</b> The retrieval component p_eta(z|x) is based on Dense Passage Retrieval. It uses a bi-encoder architecture: a query encoder and a document encoder, both based on BERT-base, trained on question-answer pairs.""",

            # Page 3: Evaluation Results & Factual Grounding
            """<b>3. Empirical Results & Groundedness</b><br/>
            We evaluate RAG on open-domain question answering benchmarks including Natural Questions, TriviaQA, WebQuestions, and CuratedTREC, as well as knowledge-intensive generation tasks such as Jeopardy question generation and MS-MARCO.<br/><br/>
            <b>Key Findings:</b><br/>
            1. RAG models achieve state-of-the-art results on Open-Domain QA, outperforming parametric-only models like T5-11B while using significantly fewer parameters.<br/>
            2. <b>Factual Grounding and Reduced Hallucination:</b> Because generation is conditioned on retrieved non-parametric text passages, RAG produces answers that are significantly more factual, specific, and verifiable compared to standard Seq2Seq models.<br/>
            3. <b>Knowledge Updating:</b> Non-parametric memory can be updated simply by swapping or updating the document index without requiring expensive retraining of the generator network."""
        ]
    },
    {
        "filename": "lora_low_rank_adaptation.pdf",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen",
        "pages": [
            # Page 1: Abstract & Problem
            """<b>LoRA: Low-Rank Adaptation of Large Language Models</b><br/>
            <i>Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (ICLR 2022)</i><br/><br/>
            <b>Abstract</b><br/>
            An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to specific tasks. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.<br/><br/>
            <b>1. Problem Statement</b><br/>
            Full fine-tuning of GPT-3 175B requires storing 175 billion parameters and their optimizer states (e.g. Adam states requiring 3x parameters in FP32), incurring massive GPU memory overhead and deployment friction when serving multiple task-specific models.""",

            # Page 2: Mathematical Formulation
            """<b>2. Method: Low-Rank Decomposition</b><br/>
            For a pre-trained weight matrix W_0 in R^{d x k}, we constrain its update Delta W by representing the latter with a low-rank decomposition:<br/>
            <i>W = W_0 + Delta W = W_0 + (B * A) * (alpha / r)</i><br/>
            where B in R^{d x r}, A in R^{r x k}, and the rank r << min(d, k).<br/><br/>
            During training, W_0 is frozen and receives no gradient updates, while A and B contain trainable parameters. Note that both W_0 and Delta W = B * A are multiplied by the same input x:<br/>
            <i>h = W_0 * x + Delta W * x = W_0 * x + (alpha / r) * B * A * x</i><br/><br/>
            We use a Gaussian initialization for matrix A and zero initialization for matrix B, so Delta W = 0 at the beginning of training. We scale Delta W by alpha / r where alpha is a constant hyperparameter.<br/><br/>
            <b>No Additional Inference Latency:</b> When deployed in production, we can explicitly compute and store W = W_0 + B * A and serve requests normally. As a result, LoRA introduces zero additional inference latency compared to full fine-tuning.""",

            # Page 3: Benefits and Experimental Results
            """<b>3. Memory Efficiency and Experimental Results</b><br/>
            <b>Memory Footprint:</b> When fine-tuning GPT-3 175B with Adam, LoRA reduces the VRAM consumption from 1.2 TB to 350 GB. With r = 4 and adapting query and value projection weights only, checkpoint size is reduced by roughly 10,000x (from 350GB to 35MB).<br/><br/>
            <b>Performance:</b> On GLUE and SuperGLUE benchmarks, LoRA matches or exceeds full fine-tuning performance on RoBERTa-large, DeBERTa-XXL, and GPT-3 175B while training as few as 0.01% of total model parameters.<br/><br/>
            <b>Switching Tasks in Production:</b> Serving multiple downstream tasks requires only swapping the small low-rank weight matrices A and B in GPU memory without reloading the massive 175B parameter base model."""
        ]
    },
    {
        "filename": "flash_attention.pdf",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "authors": "Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré",
        "pages": [
            # Page 1: Abstract & IO Bottleneck
            """<b>FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</b><br/>
            <i>Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (NeurIPS 2022)</i><br/><br/>
            <b>Abstract</b><br/>
            Transformers are slow and memory-hungry on long sequences, as the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods trade off model quality to reduce compute complexity, but often do not achieve wall-clock speedup. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM.<br/><br/>
            <b>1. Hardware Context & IO Bottleneck</b><br/>
            Modern GPUs possess large High Bandwidth Memory (HBM, e.g. 40-80GB with ~1.5-2.0 TB/s bandwidth) and small, extremely fast SRAM (e.g. 192KB per Streaming Multiprocessor with ~19 TB/s bandwidth). Standard attention computes S = Q*K^T, P = softmax(S), and O = P*V, repeatedly reading and writing intermediate N x N matrices to slow HBM.""",

            # Page 2: Tiling & Online Softmax
            """<b>2. Algorithm: Tiling and Online Softmax</b><br/>
            FlashAttention computes exact attention with O(N) memory footprint and substantially fewer HBM accesses by leveraging two core techniques:<br/><br/>
            <b>1. Tiling:</b> FlashAttention splits the input matrices Q, K, V into blocks, loads blocks into fast SRAM, computes attention with respect to that block, and incrementally updates the output.<br/>
            <b>2. Online Softmax:</b> Softmax normally requires computing the denominator sum across the entire sequence before normalizing. FlashAttention uses online softmax scaling to incrementally update the running maximum and normalization sum without ever materializing the full N x N attention matrix in HBM.<br/><br/>
            <b>Recomputation in Backward Pass:</b> Rather than saving the intermediate attention matrix for backpropagation, FlashAttention stores only the softmax normalization statistics (LSE) and quickly recomputes attention on SRAM during the backward pass, dramatically slashing memory usage."""
        ]
    },
    {
        "filename": "toolformer.pdf",
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "authors": "Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom",
        "pages": [
            # Page 1: Abstract & Introduction
            """<b>Toolformer: Language Models Can Teach Themselves to Use Tools</b><br/>
            <i>Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (Meta AI, 2023)</i><br/><br/>
            <b>Abstract</b><br/>
            Language models (LMs) demonstrate remarkable abilities to solve new tasks from just a few examples or textual instructions. However, they struggle with basic tasks like arithmetic, factual lookup, and temporal reasoning. In this paper, we introduce Toolformer, a model trained to decide which tools to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction.<br/><br/>
            <b>1. Self-Supervised Tool Use</b><br/>
            Unlike existing approaches that rely on large-scale human annotation of tool usage, Toolformer is trained in a completely self-supervised manner: given a plain text dataset, the model itself samples candidate API calls, executes them, and keeps only those API calls that demonstrably reduce perplexity on predicting subsequent tokens.""",

            # Page 2: Supported Tools & Filtering Pipeline
            """<b>2. Tools and Filtering Methodology</b><br/>
            Toolformer integrates five diverse tools via textual API call syntax:<br/>
            1. <b>Calculator:</b> Solves precise multi-step arithmetic expressions.<br/>
            2. <b>Q&A Retrieval System (Wikipedia Search):</b> Fetches up-to-date factual snippets.<br/>
            3. <b>Calendar / Clock:</b> Provides current date, time, and temporal anchoring.<br/>
            4. <b>Machine Translation System:</b> Translates non-English queries.<br/>
            5. <b>Machine Wiki Search:</b> Looks up structured entities.<br/><br/>
            <b>Loss Filtering Criterion:</b> An API call c is inserted into sequence x. Let L_without be the cross-entropy loss of predicting future tokens without the tool call, and L_with be the loss with the tool response inserted. The API call is retained if and only if: <i>L_with - L_without < -delta</i> where delta > 0 is a filtering threshold."""
        ]
    }
]


def generate_paper_pdf(paper_data: dict, output_path: Path):
    """Generate a clean, structured research paper PDF with multiple pages."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    normal_style.fontSize = 10
    normal_style.leading = 14
    
    story = []
    for page_idx, page_text in enumerate(paper_data["pages"]):
        if page_idx > 0:
            story.append(PageBreak())
        
        # Add running header
        header = Paragraph(f"<font size=8 color=gray>{paper_data['title']} | Page {page_idx + 1}</font>", normal_style)
        story.append(header)
        story.append(Spacer(1, 10))
        
        # Add page body content
        content_para = Paragraph(page_text, normal_style)
        story.append(content_para)
        story.append(Spacer(1, 15))
        
    doc.build(story)
    logger.info(f"Generated PDF: '{paper_data['filename']}' ({len(paper_data['pages'])} pages) at {output_path}")


def ingest_all_curated_papers(target_dir: Path = config.PAPERS_DIR):
    """Generate and validate all curated research papers in target directory."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Ingesting curated research papers into '{target_dir}'...")
    for paper_info in PAPERS_CONTENT:
        pdf_path = target_dir / paper_info["filename"]
        generate_paper_pdf(paper_info, pdf_path)
        
    logger.info(f"Successfully generated {len(PAPERS_CONTENT)} research paper PDFs.")
    
    # Validate loading
    docs = load_all_papers(target_dir)
    logger.info(f"Validation successful: Loaded {len(docs)} pages across all papers.")
    return docs


if __name__ == "__main__":
    ingest_all_curated_papers()
