"""
Module 8: Help & Reference
Describes every Playground feature, how it works, and links to further resources.
"""

import streamlit as st


st.title("❓ Help & Reference")
st.markdown("*Everything you need to know about the Playground and its features.*")
st.markdown("---")

# ── Table of Contents ─────────────────────────────────────────────────────
st.markdown("""
**Jump to a section:**
[Document Sources](#document-sources) · [Chunking Strategies](#chunking-strategies) ·
[Embedding Models](#embedding-models) · [Vector Stores](#vector-stores) ·
[Retrieval Strategies](#retrieval-strategies) · [Reasoning Modes](#reasoning-modes) ·
[LLM Reranking](#llm-reranking) · [Guardrails](#guardrails) ·
[Context Management](#context-management) · [Token Tracking](#token-tracking-cost) ·
[Resources](#resources-further-reading)
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📄 Document Sources")

st.markdown("""
The first step in any RAG pipeline is getting text into the system.
The Playground supports four input methods:

| Source | How It Works |
|--------|-------------|
| **File Upload** | Upload PDF, DOCX, VSDX, or TXT files. Text is automatically extracted. |
| **Paste Text** | Type or paste text directly into a text area. |
| **Sample Data** | Load pre-built sample documents from the `data/samples/` folder. |
| **URL** | Fetch raw text content from any public URL. |

After ingestion, the text is split into chunks, embedded, and stored in the vector store
so it can be searched during queries.
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("✂️ Chunking Strategies")

st.markdown("""
Chunking splits your documents into smaller pieces before embedding.
The strategy you pick affects retrieval quality significantly.
""")

strategies = {
    "Character": {
        "how": "Splits text at a fixed character count with configurable overlap.",
        "when": "General purpose. Good default when document structure is unknown.",
        "params": "`chunk_size` (characters per chunk), `overlap` (overlap between chunks).",
    },
    "Sentence": {
        "how": "Groups consecutive sentences together. Uses period/question/exclamation marks as boundaries.",
        "when": "Best for well-punctuated prose where sentences are natural semantic units.",
        "params": "`max_sentences` per chunk, `overlap_sentences` (sentences shared between chunks).",
    },
    "Paragraph": {
        "how": "Splits on double-newline boundaries (`\\n\\n`), grouping whole paragraphs.",
        "when": "Documents with clear paragraph structure (articles, reports, docs).",
        "params": "`max_paragraphs` per chunk.",
    },
    "Recursive": {
        "how": "Tries splitting on paragraphs first, then sentences, then characters — recursively finding the best natural boundary.",
        "when": "Best all-around strategy. Adapts to whatever structure the text has.",
        "params": "`chunk_size`, `overlap`.",
    },
    "Token": {
        "how": "Counts GPT tokens (via tiktoken) and splits at exact token boundaries.",
        "when": "When you need precise control over token budget per chunk (e.g., fitting within LLM context limits).",
        "params": "`max_tokens` per chunk, `overlap_tokens`.",
    },
    "Markdown / Headers": {
        "how": "Splits on Markdown headers (`#`, `##`, `###`), keeping each section as a chunk.",
        "when": "Structured documents with headers (README files, documentation, wikis).",
        "params": "`max_chunk_size` (max characters per section).",
    },
    "Sliding Window": {
        "how": "A fixed-size window slides across the text with a configurable step size, creating overlapping chunks.",
        "when": "When you want guaranteed overlap and uniform chunk sizes.",
        "params": "`window_size` (characters), `step_size` (how far the window moves each step).",
    },
    "Semantic": {
        "how": "Uses TF-IDF similarity between adjacent sentences to find topic boundaries. Groups sentences that are semantically related.",
        "when": "Documents where topic shifts matter more than fixed sizes.",
        "params": "`max_sentences` per group, `similarity_threshold` (lower = more splits).",
    },
}

for name, info in strategies.items():
    with st.expander(f"**{name}**"):
        st.markdown(f"**How it works:** {info['how']}")
        st.markdown(f"**Best used when:** {info['when']}")
        st.markdown(f"**Parameters:** {info['params']}")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🔢 Embedding Models")

st.markdown("""
Embeddings convert text into numerical vectors that capture meaning.
Similar texts produce similar vectors, enabling semantic search.

| Provider | Models | Notes |
|----------|--------|-------|
| **TF-IDF (Free)** | Local TF-IDF + SVD | No API key needed. Fits on your data. Good for demos and offline use. |
| **OpenAI** | `text-embedding-3-small`, `text-embedding-3-large`, `ada-002` | Best quality. Requires `OPENAI_API_KEY`. |
| **OpenRouter** | `openai/text-embedding-3-small`, `google/text-embedding-004`, etc. | Single API key accesses multiple providers. |
| **Ollama** | `nomic-embed-text`, `mxbai-embed-large`, `all-minilm` | 100% local. Requires [Ollama](https://ollama.com) running on your machine. |

**Tip:** TF-IDF works without any keys — great for learning. Switch to OpenAI or Ollama embeddings for production-quality results.
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📦 Vector Stores")

st.markdown("""
Vector stores hold your embedded chunks and perform similarity search.

| Store | Description |
|-------|-------------|
| **NumPy (In-Memory)** | Simple brute-force search using NumPy arrays. Excellent for learning and small datasets. Data is lost when the app restarts. |
| **ChromaDB** | Production-grade vector database with persistent storage, metadata filtering, and ANN (approximate nearest neighbor) indexing. |

Both stores support cosine similarity and MMR retrieval out of the box.
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🔍 Retrieval Strategies")

st.markdown("""
Retrieval determines *which* chunks the LLM sees when answering your question.
Choosing the right strategy can dramatically improve answer quality.
""")

retrieval = {
    "Cosine Similarity": {
        "how": "Embeds the query, computes cosine similarity against all stored chunks, returns the top K most similar.",
        "when": "Default choice. Fast and effective for most use cases.",
        "formula": r"\text{similarity}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}",
    },
    "MMR (Maximum Marginal Relevance)": {
        "how": "Iteratively selects chunks that are relevant to the query AND diverse from chunks already selected.",
        "when": "When top results are too similar to each other and you want broader coverage.",
        "formula": r"MMR = \lambda \cdot Sim(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} Sim(d_i, d_j)",
    },
    "Hybrid (BM25 + Semantic)": {
        "how": "Runs both keyword search (BM25/TF-IDF) and semantic search in parallel, then fuses results using Reciprocal Rank Fusion (RRF).",
        "when": "Queries that mix exact terms (names, codes) with semantic meaning. Generally the most robust strategy.",
        "formula": r"RRF(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}",
    },
    "Multi-Query": {
        "how": "The LLM generates 3 alternative phrasings of your question. Each is searched independently, and results are merged via RRF.",
        "when": "When the user's phrasing might not match the document's wording.",
        "formula": None,
    },
    "HyDE (Hypothetical Document Embeddings)": {
        "how": "The LLM writes a hypothetical passage that *would* answer the question. That passage is embedded and used as the search query.",
        "when": "When the question and answer live in very different semantic spaces (e.g., questions vs. factual descriptions).",
        "formula": None,
    },
}

for name, info in retrieval.items():
    with st.expander(f"**{name}**"):
        st.markdown(f"**How it works:** {info['how']}")
        st.markdown(f"**Best used when:** {info['when']}")
        if info.get("formula"):
            st.latex(info["formula"])

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🧠 Reasoning Modes")

st.markdown("""
Reasoning modes control *how* the LLM processes the retrieved context to generate an answer.
Each mode uses a different prompt template.
""")

reasoning = {
    "Standard": "The LLM receives context + question and answers directly. Fastest and simplest.",
    "Chain-of-Thought (CoT)": "The LLM reasons step-by-step before giving a final answer. Improves accuracy on complex questions.",
    "Analysis": "The LLM first evaluates each retrieved chunk's relevance, then synthesises an answer from only the useful chunks.",
    "CoT + Analysis": "Combines both — analyse chunks, then reason step-by-step. Most thorough but slowest and most expensive.",
    "Step-Back Prompting": "The LLM first answers a broader, more abstract version of the question, then uses that insight to answer the specific question.",
    "Self-Reflect": "The LLM generates an initial answer, critiques it for errors or gaps, then produces a refined final answer.",
}

for name, desc in reasoning.items():
    st.markdown(f"- **{name}** — {desc}")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🔄 LLM Reranking")

st.markdown("""
When enabled, the pipeline asks the LLM to re-score each retrieved chunk's relevance
to the question *after* initial retrieval. Chunks are then reordered by the LLM's score
and only the top N are passed to the generation step.

**Why rerank?**
- Embedding similarity is fast but imperfect — it processes query and document independently.
- Reranking processes query + document **together**, which is more accurate.
- Trade-off: one extra LLM call per query, but significantly better precision.

**Parameters:**
- **Enable LLM Reranking** — toggle on/off
- **Keep top N after rerank** — how many chunks survive reranking (default: 3)
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🛡️ Guardrails")

st.markdown("""
Guardrails are optional safety checks that run before retrieval (input) and after generation (output).
They protect against misuse and improve answer quality.
""")

col_in, col_out = st.columns(2)

with col_in:
    st.markdown("#### Input Guardrails")
    st.markdown("""
| Guardrail | Type | Description |
|-----------|------|-------------|
| **Input Length** | Rule | Rejects queries that are too short or too long |
| **PII Detection** | Regex | Detects emails, phone numbers, SSNs, credit cards |
| **Prompt Injection** | Regex | Catches jailbreak attempts ("ignore previous instructions…") |
| **Topic Filter** | LLM | Checks if the query is relevant to the knowledge base |
| **Toxicity** | LLM + Keywords | Blocks harmful or inappropriate queries |
    """)

with col_out:
    st.markdown("#### Output Guardrails")
    st.markdown("""
| Guardrail | Type | Description |
|-----------|------|-------------|
| **Hallucination Check** | LLM | Verifies every claim is grounded in the context |
| **Relevance Check** | LLM | Ensures the answer addresses the question |
| **PII in Output** | Regex | Flags personal data leaking into answers |
| **Toxicity (Output)** | LLM | Blocks harmful generated content |
    """)

st.markdown("""
**Guardrail modes:**
- **Off** — No guardrails run (fastest, no overhead)
- **Warn** — Guardrails run and results are displayed, but the query proceeds regardless
- **Block** — If any guardrail fails, the pipeline stops and returns an error

Items marked *LLM* use an extra LLM call for evaluation, which adds latency and token cost.
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("💬 Context Management")

st.markdown("""
Context management controls how conversation history is passed to the LLM
in multi-turn chat. The right strategy depends on your conversation length and budget.
""")

context = {
    "None (Stateless)": {
        "how": "Every query is independent. No previous conversation is sent to the LLM.",
        "when": "One-shot Q&A, search-style apps, or when privacy requires no history.",
        "cost": "Lowest — only current query + retrieved context.",
    },
    "Full History": {
        "how": "The entire conversation (all turns) is included in every prompt.",
        "when": "Short conversations where you need perfect recall of everything said.",
        "cost": "Grows linearly. Can exceed context window on long conversations.",
    },
    "Sliding Window": {
        "how": "Keeps only the last N user-assistant turn pairs. Older messages are dropped.",
        "when": "Medium-length conversations where recent context matters most.",
        "cost": "Bounded by window size. Typical: 3–10 turns.",
    },
    "Summary Buffer": {
        "how": "Recent turns are kept verbatim. When the buffer fills, older turns are summarised by the LLM into a rolling summary.",
        "when": "Long conversations in enterprise settings. Most popular production approach.",
        "cost": "Bounded, but requires one extra LLM call when summarising.",
    },
    "Token Budget": {
        "how": "Keeps as many recent turns as fit within a fixed token limit (counted via tiktoken).",
        "when": "When you need precise control over prompt size to manage costs.",
        "cost": "Exactly the budget you set. Typical: 1000–4000 tokens.",
    },
    "Relevant History": {
        "how": "Embeds every past turn. At query time, retrieves only the turns most similar to the current question.",
        "when": "Long conversations where users may refer back to topics discussed much earlier.",
        "cost": "Requires embedding all turns. Slightly higher latency, but very efficient context.",
    },
}

for name, info in context.items():
    with st.expander(f"**{name}**"):
        st.markdown(f"**How it works:** {info['how']}")
        st.markdown(f"**Best used when:** {info['when']}")
        st.markdown(f"**Token cost:** {info['cost']}")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📊 Token Tracking & Cost")

st.markdown("""
The Playground tracks token usage at every step so you can understand and optimise costs.

**What's tracked per query:**
- **Prompt tokens** — tokens sent to the LLM (context + question + history)
- **Completion tokens** — tokens generated by the LLM (the answer)
- **Embedding tokens** — tokens processed by the embedding model (query + any extra embeddings for multi-query/HyDE)

**Session dashboard:**
- Cumulative tokens across all chat turns
- Stacked bar chart showing prompt vs completion vs embedding tokens per turn
- Cost estimate based on model pricing (GPT-4o-mini, GPT-4o, Claude, etc.)

**Optimisation tips:**
- Use **Sliding Window** or **Token Budget** context to control history cost
- Choose **Standard** reasoning mode for simple queries (cheapest)
- Disable reranking when retrieval quality is already good
- Use smaller embedding models (`text-embedding-3-small` vs `large`) for cost savings
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📚 Resources & Further Reading")

resources = {
    "Core Concepts": [
        ("[What is RAG? — Hugging Face](https://huggingface.co/docs/transformers/model_doc/rag)", "Foundational explanation of Retrieval-Augmented Generation."),
        ("[Embedding Models — OpenAI](https://platform.openai.com/docs/guides/embeddings)", "How text embeddings work and how to use them."),
        ("[ChromaDB Documentation](https://docs.trychroma.com/)", "Getting started with ChromaDB for vector storage."),
    ],
    "Chunking & Retrieval": [
        ("[Chunking Strategies for LLM Applications — Pinecone](https://www.pinecone.io/learn/chunking-strategies/)", "Comprehensive guide to text chunking approaches."),
        ("[BM25 Algorithm Explained](https://en.wikipedia.org/wiki/Okapi_BM25)", "The keyword search algorithm used in hybrid retrieval."),
        ("[HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496)", "Original paper on the HyDE technique."),
        ("[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)", "The fusion method used in hybrid and multi-query search."),
    ],
    "Reasoning & Prompting": [
        ("[Chain-of-Thought Prompting — Google Research](https://arxiv.org/abs/2201.11903)", "Original paper showing step-by-step reasoning improves LLM accuracy."),
        ("[Step-Back Prompting](https://arxiv.org/abs/2310.06117)", "Paper on asking broader questions first for better answers."),
        ("[Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)", "Generate, critique, and refine pattern."),
    ],
    "Guardrails & Safety": [
        ("[NeMo Guardrails — NVIDIA](https://github.com/NVIDIA/NeMo-Guardrails)", "Production guardrails framework for LLM applications."),
        ("[Guardrails AI](https://www.guardrailsai.com/)", "Open-source library for validating LLM outputs."),
        ("[OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)", "Security risks in LLM applications including prompt injection."),
    ],
    "LLM Providers": [
        ("[OpenAI API Docs](https://platform.openai.com/docs/)", "Official OpenAI API documentation."),
        ("[OpenRouter](https://openrouter.ai/docs)", "One API key, many models. Free tier available."),
        ("[Ollama](https://ollama.com/)", "Run LLMs locally on your machine. Free and open source."),
    ],
    "Evaluation": [
        ("[RAGAS — RAG Evaluation Framework](https://docs.ragas.io/)", "Automated metrics for faithfulness, relevance, precision, and recall."),
        ("[LangSmith — LangChain](https://docs.smith.langchain.com/)", "Tracing, evaluation, and monitoring for LLM applications."),
    ],
}

for category, links in resources.items():
    with st.expander(f"**{category}**", expanded=True):
        for link, desc in links:
            st.markdown(f"- {link} — {desc}")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/8_🎮_Playground.py", label="← Playground", icon="🎮")
with col2:
    st.page_link("pages/10_🤖_Agent_Basics.py", label="Next: AI Agents →", icon="🤖")
