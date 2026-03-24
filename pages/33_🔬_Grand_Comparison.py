"""
🔬 Grand Comparison — All embedding methods side by side.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header, TIMELINE
from embedding_utils.embeddings import (
    get_bow_details,
    get_tfidf_details,
    get_transformer_embedding,
    load_transformer_model,
)
from embedding_utils.visualization import (
    plot_similarity_heatmap,
    plot_embeddings_2d,
    reduce_dimensions,
    plot_sparsity_comparison,
    PLOTLY_LAYOUT,
)
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import pandas as pd
import numpy as np
inject_custom_css()

page_header(
    "Grand Comparison",
    "🔬",
    "1954 → 2018+",
    "Compare ALL embedding techniques side by side on the same input. "
    "See how each method represents text differently and which captures meaning best.",
)

# ─── Evolution Summary Timeline ───
st.markdown("### 📅 The Evolution at a Glance")
html_str = '<div style="background: linear-gradient(135deg, #1a1d23, #252830); border: 1px solid rgba(108, 99, 255, 0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem;"><div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">'
for item in TIMELINE:
    html_str += f"""
<div style="text-align: center; flex: 1; min-width: 120px; padding: 0.8rem; background: rgba(108, 99, 255, 0.1); border-radius: 12px; border: 1px solid rgba(108, 99, 255, 0.15);">
    <div style="font-size: 1.5rem;">{item['icon']}</div>
    <div style="color: #B794F6; font-size: 0.75rem; font-weight: 600;">{item['year']}</div>
    <div style="color: #FAFAFA; font-size: 0.85rem; font-weight: 500;">{item['name']}</div>
</div>
"""
html_str += '</div></div>'
st.markdown(html_str, unsafe_allow_html=True)

# ─── Comparison Table ───
st.markdown("### 📋 Feature Comparison")
comparison_data = {
    "Feature": [
        "Vector Type", "Dimensionality", "Context Aware", "Handles OOV",
        "Training Required", "Semantic Understanding", "Speed", "Memory Usage",
    ],
    "Bag of Words": [
        "Sparse", "|Vocabulary|", "❌", "❌",
        "None", "❌ None", "⚡ Very Fast", "📈 High (sparse)",
    ],
    "TF-IDF": [
        "Sparse", "|Vocabulary|", "❌", "❌",
        "None", "❌ None", "⚡ Very Fast", "📈 High (sparse)",
    ],
    "Word2Vec": [
        "Dense", "300", "❌", "❌",
        "Large corpus", "✅ Good", "🏃 Fast", "📊 Medium",
    ],
    "GloVe": [
        "Dense", "100-300", "❌", "❌",
        "Large corpus", "✅ Good", "🏃 Fast", "📊 Medium",
    ],
    "Transformer": [
        "Dense", "384-768+", "✅", "✅",
        "Massive corpus", "✅✅ Excellent", "🐢 Slower", "📈 High (model)",
    ],
}
df = pd.DataFrame(comparison_data)
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

# ─── Live Comparison ───
st.markdown("### 🎮 Live Side-by-Side Comparison")
st.markdown("*Enter sentences and see how each method computes similarity:*")

default_sents = (
    "Dogs are loyal and friendly animals\n"
    "Cats are independent creatures\n"
    "Machine learning is transforming technology\n"
    "AI algorithms are reshaping the tech industry\n"
    "I enjoy eating pizza for dinner"
)
user_input = st.text_area(
    "Enter sentences (one per line, minimum 2):",
    value=default_sents,
    height=130,
    key="gc_input",
)
sentences = [s.strip() for s in user_input.strip().split("\n") if s.strip()]

if len(sentences) >= 2:
    if st.button("🚀 Compare All Methods", key="gc_compare"):
        results = {}

        # ─── BoW ───
        with st.spinner("Computing Bag of Words..."):
            bow = get_bow_details(sentences)
            results["Bag of Words"] = {
                "embeddings": bow["matrix"].astype(float),
                "dim": bow["vocab_size"],
                "sparsity": bow["sparsity"],
            }

        # ─── TF-IDF ───
        with st.spinner("Computing TF-IDF..."):
            tfidf = get_tfidf_details(sentences)
            results["TF-IDF"] = {
                "embeddings": tfidf["matrix"],
                "dim": tfidf["vocab_size"],
                "sparsity": 1.0 - (np.count_nonzero(tfidf["matrix"]) / tfidf["matrix"].size),
            }

        # ─── Transformer ───
        with st.spinner("Computing Transformer embeddings..."):
            try:
                t_model = load_transformer_model()
                t_emb = get_transformer_embedding(sentences, model=t_model)
                results["Transformer"] = {
                    "embeddings": t_emb,
                    "dim": t_emb.shape[1],
                    "sparsity": 0.0,
                }
            except Exception as e:
                st.warning(f"Transformer model unavailable: {e}")

        st.markdown("---")

        # ─── Side-by-side Similarity Heatmaps ───
        st.markdown("### 🔥 Similarity Matrices — Side by Side")
        heatmap_cols = st.columns(len(results))

        for i, (method_name, data) in enumerate(results.items()):
            with heatmap_cols[i]:
                sim = cosine_similarity(data["embeddings"])
                st.plotly_chart(
                    plot_similarity_heatmap(
                        data["embeddings"], sentences, title=method_name
                    ),
                    use_container_width=True,
                )

        st.markdown("---")

        # ─── 2D Embedding Space Comparison ───
        st.markdown("### 🗺️ 2D Embedding Space — Side by Side")
        space_cols = st.columns(len(results))
        short_labels = [f"S{i+1}" for i in range(len(sentences))]

        for i, (method_name, data) in enumerate(results.items()):
            with space_cols[i]:
                emb = data["embeddings"]
                if emb.shape[1] >= 2:
                    reduced = reduce_dimensions(emb, method="pca", n_components=2)
                    st.plotly_chart(
                        plot_embeddings_2d(reduced, short_labels, title=method_name),
                        use_container_width=True,
                    )

        st.markdown("---")

        # ─── Vector Properties Comparison ───
        st.markdown("### 📊 Vector Properties")
        prop_data = {}
        for method_name, data in results.items():
            prop_data[method_name] = {
                "Dimensionality": min(data["dim"], 1000),  # Cap for visualization
                "Sparsity (%)": round(data["sparsity"] * 100, 1),
            }
        st.plotly_chart(
            plot_sparsity_comparison(prop_data, title="Embedding Properties Comparison"),
            use_container_width=True,
        )

        # ─── Key Takeaways ───
        st.markdown("---")
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1a1d23, #2d1f4e);
                border: 1px solid rgba(108, 99, 255, 0.3);
                border-radius: 16px;
                padding: 2rem;
            ">
                <h3 style="color: #B794F6; margin-top: 0;">🎯 Key Takeaways</h3>
                <div style="color: #a0aec0; line-height: 1.8;">
                    <p>1️⃣ <b style="color: #FAFAFA;">BoW & TF-IDF</b> create sparse, high-dimensional vectors. Simple but effective for basic tasks.</p>
                    <p>2️⃣ <b style="color: #FAFAFA;">Word2Vec & GloVe</b> learn dense vectors that capture semantic relationships, but one vector per word.</p>
                    <p>3️⃣ <b style="color: #FAFAFA;">Transformers</b> produce context-aware embeddings — the same word gets different vectors in different sentences.</p>
                    <p style="margin-top: 1rem;">
                        📈 <b style="color: #6C63FF;">The trend:</b> From counting words → learning fixed representations → understanding context.
                        Each generation solves a limitation of the previous one.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Enter at least 2 sentences to run the comparison.")

# ─── Evolution Visualization ───
st.markdown("---")
st.markdown("### 🧬 The Evolution Story")
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 16px;
        padding: 2rem;
    ">
        <div style="color: #a0aec0; line-height: 2;">
            <p>📊 <b style="color: #FF6B6B;">1954 — Bag of Words:</b> "Let's just count words!" → Sparse vectors, no semantics</p>
            <p>📈 <b style="color: #FFB347;">1972 — TF-IDF:</b> "Common words shouldn't dominate!" → Weighted counts, still sparse</p>
            <p>🧠 <b style="color: #6C63FF;">2013 — Word2Vec:</b> "Words should be dense vectors learned from context!" → king - man + woman = queen</p>
            <p>🌐 <b style="color: #4ECB71;">2014 — GloVe:</b> "Let's combine global statistics with local context!" → Better training efficiency</p>
            <p>🤖 <b style="color: #00CED1;">2018 — Transformers:</b> "Context is everything!" → Same word, different embedding based on meaning</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
