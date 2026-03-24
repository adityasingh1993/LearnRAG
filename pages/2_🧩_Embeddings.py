"""
Module 2: Embeddings
Interactive exploration of how text becomes vectors.
"""

import streamlit as st
import numpy as np

st.set_page_config(page_title="Embeddings | RAG Lab", page_icon="🧩", layout="wide")

from components.sidebar import render_provider_config, get_embedding_provider
from components.viz import plot_embeddings_2d, plot_similarity_heatmap

render_provider_config()

st.title("🧩 Embeddings")
st.markdown("*Turn text into numbers that capture meaning. The foundation of semantic search.*")
st.markdown("---")

# ── Concept ──────────────────────────────────────────────────────────────
st.header("What Are Embeddings?")
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    An **embedding** is a list of numbers (a vector) that represents the *meaning* of a piece of text.

    **Key properties:**
    - Similar meanings → vectors that are **close together**
    - Different meanings → vectors that are **far apart**
    - This enables **semantic search** (search by meaning, not keywords)

    **Example:** The sentences "The cat sat on the mat" and "A feline rested on the rug"
    would have similar embeddings, even though they share few words.
    """)

with col2:
    st.markdown("""
    ```
    "I love dogs" → [0.12, -0.45, 0.78, ...]
    "I adore puppies" → [0.11, -0.43, 0.79, ...]
    "Quantum physics" → [-0.89, 0.23, -0.12, ...]
    ```

    Notice how the first two vectors are similar (close topic)
    while the third is very different.
    """)

# ── Interactive Demo ─────────────────────────────────────────────────────
st.markdown("---")
st.header("🔬 Interactive: Explore Embeddings")

st.markdown("Enter some sentences below and watch how they map to vector space.")

if "embed_sentences" not in st.session_state:
    st.session_state.embed_sentences = [
        "The cat sat on the mat",
        "A dog played in the park",
        "Machine learning is fascinating",
        "Deep learning uses neural networks",
        "I love eating pizza",
        "The weather is sunny today",
    ]

sentences_text = st.text_area(
    "Enter sentences (one per line):",
    value="\n".join(st.session_state.embed_sentences),
    height=180,
)

sentences = [s.strip() for s in sentences_text.strip().split("\n") if s.strip()]

if sentences and len(sentences) >= 2:
    if st.button("🔢 Generate Embeddings", type="primary"):
        with st.spinner("Computing embeddings..."):
            try:
                embed_provider = get_embedding_provider()
                embeddings = embed_provider.embed(sentences)

                st.session_state.embed_sentences = sentences
                st.session_state.computed_embeddings = embeddings
                st.session_state.embed_provider_name = embed_provider.name()

                st.success(f"Generated {len(sentences)} embeddings using **{embed_provider.name()}** "
                           f"(dimension: {embeddings.shape[1]})")
            except Exception as e:
                st.error(f"Error computing embeddings: {e}")

    if "computed_embeddings" in st.session_state:
        embeddings = st.session_state.computed_embeddings
        sents = st.session_state.embed_sentences

        tab1, tab2, tab3 = st.tabs(["📊 2D Visualization", "🌡️ Similarity Heatmap", "🔢 Raw Vectors"])

        with tab1:
            st.markdown("Embeddings projected to 2D using PCA. Nearby points = similar meaning.")
            plot_embeddings_2d(embeddings, sents)

        with tab2:
            st.markdown("Each cell shows cosine similarity between two sentences (1.0 = identical, 0.0 = unrelated).")
            plot_similarity_heatmap(sents, embeddings)

        with tab3:
            st.markdown(f"Each sentence is a vector of {embeddings.shape[1]} dimensions. "
                        "Showing first 10 values:")
            for i, (s, emb) in enumerate(zip(sents, embeddings)):
                vec_str = ", ".join(f"{v:.4f}" for v in emb[:10])
                st.code(f'"{s[:50]}..." → [{vec_str}, ...]', language=None)

        # Similarity calculator
        st.markdown("---")
        st.subheader("🔍 Similarity Calculator")
        col1, col2 = st.columns(2)
        with col1:
            idx1 = st.selectbox("Sentence A:", range(len(sents)),
                                format_func=lambda i: sents[i][:60], key="sim_a")
        with col2:
            idx2 = st.selectbox("Sentence B:", range(len(sents)),
                                format_func=lambda i: sents[i][:60], index=min(1, len(sents)-1), key="sim_b")

        if idx1 is not None and idx2 is not None:
            a = embeddings[idx1]
            b = embeddings[idx2]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            euclidean = np.linalg.norm(a - b)

            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("Cosine Similarity", f"{cos_sim:.4f}")
            with mc2:
                st.metric("Euclidean Distance", f"{euclidean:.4f}")

            if cos_sim > 0.8:
                st.success("These sentences are **very similar** in meaning!")
            elif cos_sim > 0.5:
                st.info("These sentences are **somewhat related**.")
            else:
                st.warning("These sentences have **different meanings**.")

else:
    st.warning("Enter at least 2 sentences to generate embeddings.")

# ── How Embeddings Work ──────────────────────────────────────────────────
st.markdown("---")
st.header("How Do Embedding Models Work?")

with st.expander("📚 The Technical Details", expanded=False):
    st.markdown("""
    **Training Process:**
    1. The model sees millions of text pairs (similar/dissimilar)
    2. It learns to map similar texts to nearby points in vector space
    3. The model uses a transformer architecture (like BERT) to understand context

    **Types of Embedding Models:**

    | Model | Dimensions | Speed | Quality | Cost |
    |-------|-----------|-------|---------|------|
    | OpenAI text-embedding-3-small | 1536 | Fast | Excellent | Paid |
    | OpenAI text-embedding-3-large | 3072 | Medium | Best | Paid |
    | nomic-embed-text (Ollama) | 768 | Medium | Very Good | Free/Local |
    | all-MiniLM-L6 | 384 | Very Fast | Good | Free/Local |
    | TF-IDF + SVD | Configurable | Instant | Basic | Free/Local |

    **Cosine Similarity:** The standard way to compare embeddings.
    It measures the angle between two vectors, ignoring magnitude.
    - **1.0** = identical direction (same meaning)
    - **0.0** = perpendicular (unrelated)
    - **-1.0** = opposite (opposite meaning)
    """)

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/1_📖_Learn_Basics.py", label="← Learn Basics", icon="📖")
with col2:
    st.page_link("pages/3_📦_Vector_Stores.py", label="Next: Vector Stores →", icon="📦")
