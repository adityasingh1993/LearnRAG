"""
Module 4: Retrieval
Explore different retrieval strategies and compare their results.
"""

import streamlit as st
import numpy as np
import os


from components.sidebar import render_provider_config, get_embedding_provider
from components.viz import plot_embeddings_2d, plot_retrieval_scores
from core.vector_store import NumpyVectorStore

render_provider_config()

st.title("🔍 Retrieval Strategies")
st.markdown("*Finding the right information is the most critical step in RAG.*")
st.markdown("---")

# ── Concept ──────────────────────────────────────────────────────────────
st.header("Why Retrieval Matters")

st.markdown("""
> **"Garbage in, garbage out"** — If the retriever gives the LLM bad context,
> the answer will be wrong no matter how smart the model is.

The retrieval step decides **which chunks** get passed to the LLM. Getting this right is often
more impactful than choosing a better LLM.
""")

# ── Strategies ───────────────────────────────────────────────────────────
st.header("Retrieval Strategies")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📏 Similarity", "🎯 MMR", "⚡ Hybrid", "🔄 Multi-Query", "💡 HyDE",
])

with tab1:
    st.markdown("""
    ### Cosine Similarity Search
    The simplest and most common approach.

    **How it works:**
    1. Compute cosine similarity between query and each document embedding
    2. Return the top K highest-scoring chunks

    **Pros:** Simple, fast, effective for most cases
    **Cons:** Results can be redundant (similar chunks repeated)
    """)

with tab2:
    st.markdown("""
    ### Maximum Marginal Relevance (MMR)
    Balances **relevance** (similar to query) with **diversity** (dissimilar to each other).

    **The MMR formula:**
    ```
    MMR = λ × Sim(chunk, query) - (1-λ) × max(Sim(chunk, already_selected))
    ```

    - **λ = 1.0** → Pure similarity (same as regular search)
    - **λ = 0.5** → Balance of relevance and diversity
    - **λ = 0.0** → Maximum diversity

    **Pros:** Avoids redundant results, covers more aspects
    **Cons:** May miss the most relevant chunk if it's similar to another
    """)

with tab3:
    st.markdown("""
    ### Hybrid Search (BM25 + Semantic)
    Combines **semantic search** (embeddings) with **keyword search** (BM25/TF-IDF)
    using **Reciprocal Rank Fusion (RRF)**.

    **Why combine?**
    - Semantic search understands meaning but can miss exact keywords
    - Keyword search finds exact matches but misses synonyms

    **Approach:**
    1. Run both semantic and keyword search in parallel
    2. Combine scores using RRF: `score(d) = Σ 1/(k + rank(d))`
    3. Return top K from combined, re-ranked results

    **Pros:** Best recall, handles both exact and semantic matches
    **Cons:** Slightly more compute
    """)

with tab4:
    st.markdown("""
    ### Multi-Query Retrieval
    Uses the LLM to **rewrite the query** 3 different ways, retrieves for each,
    and merges results with RRF.

    **Why?** A single query phrasing may miss relevant documents.
    Rephrasing covers synonyms, perspectives, and specificity levels.

    **Flow:**
    1. LLM generates 3 alternative phrasings of the question
    2. Each variant is searched independently
    3. Results merged via Reciprocal Rank Fusion

    **Pros:** Much better recall for ambiguous queries
    **Cons:** 1 extra LLM call + 3 extra vector searches
    """)

with tab5:
    st.markdown("""
    ### HyDE — Hypothetical Document Embeddings
    Instead of searching with the *question*, the LLM generates a **hypothetical
    answer** and that answer is embedded and used for search.

    **Why?** Questions and answers often live in different semantic spaces.
    A hypothetical answer is closer to actual document passages.

    **Flow:**
    1. LLM writes a short hypothetical passage answering the question
    2. That passage is embedded
    3. Nearest-neighbour search uses the hypothetical embedding

    **Pros:** Bridges the query–document semantic gap
    **Cons:** 1 extra LLM call; quality depends on the LLM's imagination
    """)

# ── Interactive Comparison ────────────────────────────────────────────────
st.markdown("---")
st.header("🔬 Interactive: Compare Retrieval Methods")

# Prepare data
default_docs = [
    "RAG combines retrieval with generation for accurate AI responses.",
    "Retrieval-Augmented Generation uses external knowledge to improve LLMs.",
    "Vector databases store embeddings for fast similarity search.",
    "Embeddings convert text into numerical vectors that capture meaning.",
    "Cosine similarity measures the angle between two vectors.",
    "LLMs like GPT-4 can hallucinate without grounding in real data.",
    "ChromaDB is a popular open-source vector database for AI applications.",
    "FAISS by Meta enables billion-scale similarity search efficiently.",
    "Chunking splits documents into smaller pieces for better retrieval.",
    "Prompt engineering helps LLMs produce better outputs from retrieved context.",
    "Fine-tuning adapts a model's weights to specific tasks or domains.",
    "Transformers use self-attention to process sequences in parallel.",
]

custom_docs = st.text_area(
    "Documents (one per line):",
    value="\n".join(default_docs),
    height=200,
    key="ret_docs",
)
docs = [d.strip() for d in custom_docs.strip().split("\n") if d.strip()]

query = st.text_input("Search query:", "How does RAG work with embeddings?", key="ret_query")

col1, col2 = st.columns(2)
with col1:
    k = st.slider("Top K results:", 1, min(10, len(docs)), 3, key="ret_k")
with col2:
    lambda_mult = st.slider("MMR Lambda (1.0=relevance, 0.0=diversity):", 0.0, 1.0, 0.5, key="ret_lambda")

if st.button("🔍 Compare Retrieval Methods", type="primary") and docs and query:
    with st.spinner("Embedding documents and searching..."):
        try:
            embed_provider = get_embedding_provider()
            doc_embeddings = embed_provider.embed(docs)
            query_emb = embed_provider.embed_query(query)

            store = NumpyVectorStore()
            store.add(docs, doc_embeddings)

            sim_results = store.search(query_emb, k=k)
            mmr_results = store.search_mmr(query_emb, k=k, lambda_mult=lambda_mult)

            st.session_state["ret_results"] = {
                "sim": sim_results,
                "mmr": mmr_results,
                "embeddings": doc_embeddings,
                "query_emb": query_emb,
                "docs": docs,
                "query": query,
            }
        except Exception as e:
            st.error(f"Error: {e}")

if "ret_results" in st.session_state:
    data = st.session_state["ret_results"]

    col_sim, col_mmr = st.columns(2)
    with col_sim:
        st.subheader("📏 Similarity Search")
        for i, r in enumerate(data["sim"]):
            score_color = "#00CC96" if r.score > 0.5 else "#FF6B6B"
            st.markdown(f"**#{i+1}** (score: <span style='color:{score_color}'>{r.score:.3f}</span>)",
                        unsafe_allow_html=True)
            st.markdown(f"> {r.text}")
        plot_retrieval_scores(data["sim"], data["query"])

    with col_mmr:
        st.subheader("🎯 MMR Search")
        for i, r in enumerate(data["mmr"]):
            score_color = "#00CC96" if r.score > 0.5 else "#FF6B6B"
            st.markdown(f"**#{i+1}** (score: <span style='color:{score_color}'>{r.score:.3f}</span>)",
                        unsafe_allow_html=True)
            st.markdown(f"> {r.text}")
        plot_retrieval_scores(data["mmr"], f"{data['query']} (MMR)")

    st.subheader("📊 Embedding Space")
    sim_indices = [r.index for r in data["sim"]]
    plot_embeddings_2d(
        data["embeddings"], data["docs"],
        query_embedding=data["query_emb"],
        query_label=data["query"][:30],
        highlight_indices=sim_indices,
    )

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/3_📦_Vector_Stores.py", label="← Vector Stores", icon="📦")
with col2:
    st.page_link("pages/5_🤖_Generation.py", label="Next: Generation →", icon="🤖")
