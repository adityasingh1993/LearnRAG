"""
📊 Bag of Words — The simplest text representation.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header, show_pros_cons, SAMPLE_SENTENCES
from embedding_utils.embeddings import get_bow_details
from embedding_utils.visualization import (
    plot_document_term_matrix,
    plot_embedding_vector,
    plot_similarity_heatmap,
    plot_embeddings_2d,
    reduce_dimensions,
)

inject_custom_css()

page_header(
    "Bag of Words (BoW)",
    "📊",
    "~1954",
    "The simplest text embedding — count how many times each word appears. "
    "Each document becomes a sparse vector where each dimension is a word in the vocabulary.",
)

# ─── Theory Section ───
with st.expander("📖 How It Works", expanded=True):
    st.markdown("""
    ### The Bag of Words Model

    **Concept:** Represent each document as a vector of word counts, ignoring grammar and word order.

    **Steps:**
    1. **Build vocabulary** — Collect all unique words from all documents
    2. **Vectorize** — For each document, count how often each vocabulary word appears
    3. **Result** — Each document = a vector of length `|vocabulary|`

    **Example:**
    ```
    Doc 1: "the cat sat"     → [1, 1, 1, 0]  (the, cat, sat, dog)
    Doc 2: "the dog sat"     → [1, 0, 1, 1]  (the, cat, sat, dog)
    ```

    > 💡 **Key insight:** BoW completely ignores word order — "dog bites man" and "man bites dog" have the **same** embedding!
    """)

st.markdown("---")

# ─── Live Demo ───
st.markdown("### 🎮 Live Demo")
st.markdown("*Type your own sentences below and watch the BoW vectors form:*")

default_text = "The cat sat on the mat\nThe dog sat on the log\nThe cat chased the dog"
user_input = st.text_area(
    "Enter sentences (one per line):",
    value=default_text,
    height=120,
    key="bow_input",
)

sentences = [s.strip() for s in user_input.strip().split("\n") if s.strip()]

if len(sentences) >= 2:
    details = get_bow_details(sentences)

    # ─── Metrics ───
    cols = st.columns(3)
    with cols[0]:
        st.metric("📝 Documents", len(sentences))
    with cols[1]:
        st.metric("📚 Vocabulary Size", details["vocab_size"])
    with cols[2]:
        st.metric("🕳️ Sparsity", f"{details['sparsity']:.1%}")

    st.markdown("---")

    # ─── Tabs for different views ───
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Document-Term Matrix",
        "📊 Vector View",
        "🔥 Similarity Heatmap",
        "🗺️ 2D Embedding Space",
    ])

    with tab1:
        st.markdown("#### Document-Term Matrix")
        st.markdown("Each row is a document, each column is a word. Values = word counts.")
        st.plotly_chart(
            plot_document_term_matrix(details["matrix"], details["features"], sentences),
            use_container_width=True,
        )

        # Also show as DataFrame
        df = pd.DataFrame(details["matrix"], columns=details["features"],
                          index=[f"Doc {i+1}" for i in range(len(sentences))])
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.markdown("#### Individual Document Vectors")
        selected_doc = st.selectbox(
            "Select a document to inspect:",
            [f"Doc {i+1}: {s[:50]}" for i, s in enumerate(sentences)],
        )
        doc_idx = int(selected_doc.split(":")[0].replace("Doc ", "")) - 1
        st.plotly_chart(
            plot_embedding_vector(
                details["matrix"][doc_idx],
                title=f"BoW Vector — Doc {doc_idx + 1}",
                feature_names=list(details["features"]),
            ),
            use_container_width=True,
        )

    with tab3:
        st.markdown("#### Cosine Similarity Between Documents")
        st.markdown("How similar are the documents in BoW space?")
        st.plotly_chart(
            plot_similarity_heatmap(details["matrix"], sentences),
            use_container_width=True,
        )

    with tab4:
        st.markdown("#### 2D Projection (PCA)")
        st.markdown("Documents projected to 2D using PCA:")
        if details["matrix"].shape[1] >= 2:
            reduced = reduce_dimensions(details["matrix"].astype(float), method="pca", n_components=2)
            short_labels = [f"D{i+1}" for i in range(len(sentences))]
            st.plotly_chart(
                plot_embeddings_2d(reduced, short_labels, title="BoW Embeddings — 2D PCA"),
                use_container_width=True,
            )
        else:
            st.info("Need at least 2 vocabulary words for 2D projection.")

elif len(sentences) == 1:
    st.warning("Please enter at least 2 sentences to compare.")
else:
    st.info("Enter some sentences above to see the demo.")

# ─── Vocabulary Growth Demo ───
st.markdown("---")
st.markdown("### 📈 Vocabulary Growth")
st.markdown("*See how vocabulary size explodes as you add more documents:*")

all_sents = []
for cat, sents in SAMPLE_SENTENCES.items():
    all_sents.extend(sents)

num_docs = st.slider("Number of documents:", 2, len(all_sents), 5)
subset = all_sents[:num_docs]
sub_details = get_bow_details(subset)

col1, col2 = st.columns(2)
with col1:
    st.metric("Documents", num_docs)
with col2:
    st.metric("Vocabulary Size", sub_details["vocab_size"])

st.markdown(f"**Vector dimensionality:** Each document is now a `{sub_details['vocab_size']}`-dimensional vector "
            f"with **{sub_details['sparsity']:.1%}** zeros!")

with st.expander("👀 View the active documents driving this vocabulary", expanded=False):
    for i, s in enumerate(subset):
        st.markdown(f"**Doc {i+1}:** {s}")

# ─── Strengths & Weaknesses ───
st.markdown("---")
show_pros_cons(
    pros=[
        "Simple and intuitive",
        "Fast to compute",
        "Works well for basic text classification",
        "No training required",
    ],
    cons=[
        "Ignores word order completely",
        "Very high dimensionality (= vocabulary size)",
        "Extremely sparse vectors",
        "No semantic understanding (\"good\" ≠ \"great\")",
        "Common words dominate",
    ],
)
