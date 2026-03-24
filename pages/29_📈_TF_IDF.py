"""
📈 TF-IDF — Term Frequency × Inverse Document Frequency.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header, show_pros_cons
from embedding_utils.embeddings import get_bow_details, get_tfidf_details
from embedding_utils.visualization import (
    plot_document_term_matrix,
    plot_comparison_bars,
    plot_similarity_heatmap,
    plot_embeddings_2d,
    reduce_dimensions,
)

inject_custom_css()

page_header(
    "TF-IDF",
    "📈",
    "~1972",
    "An improvement over Bag of Words — weighs terms by how important they are. "
    "Common words get downweighted, rare words get boosted.",
)

# ─── Theory ───
with st.expander("📖 How It Works", expanded=True):
    st.markdown("""
    ### Term Frequency × Inverse Document Frequency

    **Problem with BoW:** Words like "the", "is", "a" appear everywhere and dominate the vectors.

    **Solution:** Weight each word by how *discriminative* it is across documents.

    **Formula:**

    $$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$$

    Where:
    - **TF(t, d)** = How often term `t` appears in document `d` (normalized)
    - **IDF(t)** = log(Total documents / Documents containing `t`)

    **Intuition:**
    - Words appearing in **many** documents → low IDF → less important
    - Words appearing in **few** documents → high IDF → more important

    > 💡 **Key insight:** "the" appears everywhere → IDF ≈ 0 → near-zero weight.
    > "quantum" appears rarely → high IDF → high weight!
    """)

st.markdown("---")

# ─── Live Demo ───
st.markdown("### 🎮 Live Demo")
st.markdown("*Compare BoW and TF-IDF on the same sentences:*")

default_text = (
    "The cat sat on the mat\n"
    "The dog sat on the log\n"
    "The cat chased the dog around the park\n"
    "A bird flew over the park"
)
user_input = st.text_area("Enter sentences (one per line):", value=default_text, height=120, key="tfidf_input")
sentences = [s.strip() for s in user_input.strip().split("\n") if s.strip()]

if len(sentences) >= 2:
    bow_details = get_bow_details(sentences)
    tfidf_details = get_tfidf_details(sentences)

    # ─── Metrics ───
    cols = st.columns(4)
    with cols[0]:
        st.metric("📝 Documents", len(sentences))
    with cols[1]:
        st.metric("📚 Vocabulary Size", tfidf_details["vocab_size"])
    with cols[2]:
        bow_max = bow_details["matrix"].max()
        tfidf_max = tfidf_details["matrix"].max()
        st.metric("BoW Max Weight", f"{bow_max:.0f}")
    with cols[3]:
        st.metric("TF-IDF Max Weight", f"{tfidf_max:.3f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚖️ BoW vs TF-IDF Comparison",
        "📊 IDF Values",
        "🔥 Similarity Comparison",
        "🗺️ 2D Embedding Space",
    ])

    with tab1:
        st.markdown("#### Side-by-Side: BoW vs TF-IDF Weights")
        selected_doc = st.selectbox(
            "Select a document:",
            [f"Doc {i+1}: {s[:50]}" for i, s in enumerate(sentences)],
            key="tfidf_doc_select",
        )
        doc_idx = int(selected_doc.split(":")[0].replace("Doc ", "")) - 1

        # Normalize BoW for fair comparison
        bow_norm = bow_details["matrix"][doc_idx].astype(float)
        if bow_norm.max() > 0:
            bow_norm = bow_norm / bow_norm.max()
        tfidf_norm = tfidf_details["matrix"][doc_idx]
        if tfidf_norm.max() > 0:
            tfidf_norm = tfidf_norm / tfidf_norm.max()

        st.plotly_chart(
            plot_comparison_bars(bow_norm, tfidf_norm, tfidf_details["features"]),
            use_container_width=True,
        )

        st.info("🔍 Notice how common words like 'the' get **downweighted** in TF-IDF, "
                "while distinctive words get **boosted**!")

    with tab2:
        st.markdown("#### IDF Values — How Discriminative Is Each Word?")
        st.markdown("Higher IDF = word appears in fewer documents = more discriminative.")

        idf_df = pd.DataFrame(
            sorted(tfidf_details["idf_values"].items(), key=lambda x: x[1], reverse=True),
            columns=["Word", "IDF Score"],
        )
        st.dataframe(idf_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### TF-IDF Document-Term Matrix")
        st.plotly_chart(
            plot_document_term_matrix(
                tfidf_details["matrix"],
                tfidf_details["features"],
                sentences,
                title="TF-IDF Weighted Document-Term Matrix",
            ),
            use_container_width=True,
        )

    with tab3:
        st.markdown("#### Similarity: BoW vs TF-IDF")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_similarity_heatmap(
                    bow_details["matrix"].astype(float), sentences, title="BoW Similarity"
                ),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                plot_similarity_heatmap(
                    tfidf_details["matrix"], sentences, title="TF-IDF Similarity"
                ),
                use_container_width=True,
            )
        st.info("📊 TF-IDF often gives **more nuanced** similarity scores because "
                "common words don't inflate the similarity.")

    with tab4:
        st.markdown("#### 2D Projection Comparison")
        col1, col2 = st.columns(2)
        if bow_details["matrix"].shape[1] >= 2:
            bow_reduced = reduce_dimensions(bow_details["matrix"].astype(float), method="pca", n_components=2)
            tfidf_reduced = reduce_dimensions(tfidf_details["matrix"], method="pca", n_components=2)
            short_labels = [f"D{i+1}" for i in range(len(sentences))]

            with col1:
                st.plotly_chart(
                    plot_embeddings_2d(bow_reduced, short_labels, title="BoW — 2D PCA"),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    plot_embeddings_2d(tfidf_reduced, short_labels, title="TF-IDF — 2D PCA"),
                    use_container_width=True,
                )

else:
    st.info("Enter at least 2 sentences above to see the demo.")

# ─── Pros/Cons ───
st.markdown("---")
show_pros_cons(
    pros=[
        "Downweights common/stop words automatically",
        "Better than BoW for information retrieval",
        "Still fast to compute",
        "No training required — unsupervised",
    ],
    cons=[
        "Still ignores word order",
        "Still high dimensionality (= vocabulary size)",
        "No semantic understanding",
        "Vectors are still fairly sparse",
    ],
)
