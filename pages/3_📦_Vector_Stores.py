"""
Module 3: Vector Stores
Interactive demo of storing embeddings and searching through them.
"""

import streamlit as st
import numpy as np
import os

st.set_page_config(page_title="Vector Stores | RAG Lab", page_icon="📦", layout="wide")

from components.sidebar import render_provider_config, get_embedding_provider, get_vector_store
from components.viz import plot_embeddings_2d, plot_retrieval_scores

render_provider_config()

st.title("📦 Vector Stores")
st.markdown("*Store embeddings and search through them at lightning speed.*")
st.markdown("---")

# ── Concept ──────────────────────────────────────────────────────────────
st.header("What is a Vector Store?")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
    A **vector store** (or vector database) is a specialized database designed to:

    1. **Store** high-dimensional vectors (embeddings)
    2. **Index** them for fast retrieval
    3. **Search** by similarity (not exact match)

    When you search a vector store, you give it a query vector and ask:
    *"Find me the K most similar vectors"*

    This is called **Approximate Nearest Neighbor (ANN)** search.
    """)

with col2:
    st.markdown("""
    **Popular Vector Stores:**

    | Name | Type | Best For |
    |------|------|----------|
    | **ChromaDB** | Embedded | Prototyping |
    | **FAISS** | Library | High performance |
    | **Pinecone** | Cloud | Production |
    | **Weaviate** | Self-hosted | Full-featured |
    | **Qdrant** | Self-hosted | Filtering |
    | **NumPy** | In-memory | Learning! |
    """)

# ── How Search Works ─────────────────────────────────────────────────────
with st.expander("🔧 How Similarity Search Works Under the Hood"):
    st.markdown("""
    **Brute Force (what our NumPy store does):**
    1. Compute cosine similarity between query and ALL stored vectors
    2. Sort by similarity
    3. Return top K results
    - Time: O(N × D) where N = num docs, D = dimension
    - Great for learning, slow for millions of docs

    **HNSW (what ChromaDB/production stores use):**
    1. Build a multi-layer graph connecting similar vectors
    2. Navigate the graph from coarse to fine layers
    3. Return approximate top K results
    - Time: O(log N)
    - Fast even for billions of vectors!
    """)

    st.code("""
# Simplified cosine similarity search (what happens inside)
def search(query_vector, stored_vectors, k=5):
    # Normalize vectors
    query_norm = query_vector / np.linalg.norm(query_vector)
    doc_norms = stored_vectors / np.linalg.norm(stored_vectors, axis=1, keepdims=True)

    # Compute similarities (dot product of normalized vectors = cosine similarity)
    similarities = doc_norms @ query_norm

    # Return top K
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return top_k_indices, similarities[top_k_indices]
""", language="python")

# ── Interactive Demo ─────────────────────────────────────────────────────
st.markdown("---")
st.header("🔬 Interactive: Build & Search a Vector Store")

from core.document_loader import SUPPORTED_EXTENSIONS, load_text as _load_file

sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples")

tab_upload, tab_sample, tab_custom = st.tabs(["📁 Upload File", "📄 Sample Data", "✏️ Custom Text"])

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a document (PDF, DOCX, VSDX, TXT)",
        type=SUPPORTED_EXTENSIONS,
        key="vs_upload",
    )
    if uploaded:
        try:
            doc_text = _load_file(uploaded.read(), uploaded.name)
            st.text_area("Preview:", doc_text[:500] + "...", height=150, disabled=True)
            st.caption(f"Extracted **{len(doc_text):,}** chars from `{uploaded.name}`")
        except Exception as e:
            st.error(f"Parse error: {e}")
            doc_text = ""

with tab_sample:
    sample_files = []
    if os.path.isdir(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".txt")]
    if sample_files:
        selected_sample = st.selectbox("Choose sample:", sample_files, key="vs_sample")
        with open(os.path.join(sample_dir, selected_sample), "r", encoding="utf-8") as f:
            doc_text_sample = f.read()
        st.text_area("Preview:", doc_text_sample[:500] + "...", height=150, disabled=True, key="vs_preview")
    else:
        st.info("No sample files found in data/samples/")

with tab_custom:
    doc_text_custom = st.text_area(
        "Enter your text:",
        value="RAG stands for Retrieval-Augmented Generation.\nIt combines search with language models.\nEmbeddings capture meaning as numbers.",
        height=150,
        key="vs_custom",
    )

# Determine which text to use
active_tab_text = None
if "vs_upload" in st.session_state and st.session_state.get("vs_upload"):
    active_tab_text = doc_text
elif sample_files:
    active_tab_text = doc_text_sample
else:
    active_tab_text = doc_text_custom if "doc_text_custom" not in dir() else doc_text_custom

if active_tab_text is None:
    active_tab_text = st.session_state.get("vs_custom", "")

# Chunking settings
with st.expander("⚙️ Chunking Settings"):
    chunk_strategy = st.selectbox("Strategy:", ["recursive", "sentence", "character", "paragraph"], key="vs_chunk_strat")
    chunk_size = st.slider("Chunk size:", 100, 2000, 500, key="vs_chunk_size")

if st.button("📦 Index Documents", type="primary"):
    if not active_tab_text.strip():
        st.warning("Please provide some text to index.")
    else:
        with st.spinner("Chunking → Embedding → Storing..."):
            try:
                from core.chunking import chunk_text
                chunks = chunk_text(active_tab_text, strategy=chunk_strategy, chunk_size=chunk_size)
                chunk_texts = [c.text for c in chunks]

                embed_provider = get_embedding_provider()
                embeddings = embed_provider.embed(chunk_texts)

                vs = get_vector_store()
                vs.clear()
                vs.add(chunk_texts, embeddings)

                st.session_state["vs_instance"] = vs
                st.session_state["vs_chunks"] = chunk_texts
                st.session_state["vs_embeddings"] = embeddings

                st.success(f"Indexed **{len(chunks)} chunks** into vector store! "
                           f"(Embedding dim: {embeddings.shape[1]})")
            except Exception as e:
                st.error(f"Error: {e}")

# Search
if "vs_instance" in st.session_state:
    st.markdown("---")
    st.subheader("🔍 Search the Vector Store")

    query = st.text_input("Enter your search query:", placeholder="What is RAG?", key="vs_query")
    k = st.slider("Number of results (k):", 1, 10, 3, key="vs_k")

    if query:
        try:
            embed_provider = get_embedding_provider()
            query_emb = embed_provider.embed_query(query)
            vs = st.session_state["vs_instance"]
            results = vs.search(query_emb, k=k)

            st.markdown(f"**Found {len(results)} results:**")
            for i, r in enumerate(results):
                with st.container():
                    st.markdown(f"**#{i+1}** — Score: `{r.score:.4f}`")
                    st.markdown(f"> {r.text}")
                    st.markdown("")

            col1, col2 = st.columns(2)
            with col1:
                plot_retrieval_scores(results, query)
            with col2:
                highlight = [r.index for r in results]
                plot_embeddings_2d(
                    st.session_state["vs_embeddings"],
                    st.session_state["vs_chunks"],
                    query_embedding=query_emb,
                    query_label=query[:30],
                    highlight_indices=highlight,
                )
        except Exception as e:
            st.error(f"Search error: {e}")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/2_🧩_Embeddings.py", label="← Embeddings", icon="🧩")
with col2:
    st.page_link("pages/4_🔍_Retrieval.py", label="Next: Retrieval →", icon="🔍")
