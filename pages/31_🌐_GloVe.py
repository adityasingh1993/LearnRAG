"""
🌐 GloVe — Global Vectors for Word Representation.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header, show_pros_cons
from embedding_utils.embeddings import (
    load_glove_model,
    get_glove_embeddings,
    glove_most_similar,
)
from embedding_utils.visualization import (
    plot_embeddings_2d,
    plot_similarity_heatmap,
    plot_analogy_result,
    plot_word_clusters,
    reduce_dimensions,
)

inject_custom_css()

page_header(
    "GloVe",
    "🌐",
    "2014 (Stanford)",
    "Global Vectors — combines the best of count-based methods (like TF-IDF) and prediction-based methods "
    "(like Word2Vec). Uses global co-occurrence statistics to learn word embeddings.",
)

# ─── Theory ───
with st.expander("📖 How It Works", expanded=True):
    st.markdown("""
    ### GloVe: Global Vectors for Word Representation

    **Key Insight:** Word2Vec only uses local context windows, but the full corpus has
    global co-occurrence statistics we should exploit.

    **Approach:**
    1. **Build a co-occurrence matrix** — count how often word pairs appear together
    2. **Factorize** — learn vectors such that their dot product approximates the log of the co-occurrence count
    3. **Objective:** $w_i \\cdot w_j + b_i + b_j = \\log(X_{ij})$

    **How it differs from Word2Vec:**

    | | Word2Vec | GloVe |
    |---|---|---|
    | **Method** | Prediction (neural net) | Matrix factorization |
    | **Context** | Local window | Global co-occurrence |
    | **Training** | Online (SGD on windows) | Batch (full matrix) |
    | **Key insight** | Context prediction | Co-occurrence ratios |

    > 💡 **GloVe uses global statistics**, so it can capture broader relationships.
    > In practice, GloVe and Word2Vec perform similarly on most tasks.
    """)

st.markdown("---")
st.markdown("### 🎮 Interactive Demos")
st.info("⏳ The GloVe model will download on first use. Subsequent loads are instant.")

try:
    model = load_glove_model()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load GloVe model: {e}")
    model_loaded = False

if model_loaded:
    tab1, tab2, tab3 = st.tabs([
        "🔍 Similar Words",
        "🗺️ Word Clusters",
        "📊 Sentence Similarity",
    ])

    with tab1:
        st.markdown("#### 🔍 Find Similar Words with GloVe")
        query_word = st.text_input("Enter a word:", value="science", key="glove_similar")

        if st.button("🔎 Find Similar", key="glove_find"):
            results = glove_most_similar(query_word, model=model, topn=15)
            if results:
                words, scores = zip(*results)
                st.plotly_chart(
                    plot_analogy_result(list(words), list(scores), title=f'GloVe: Words similar to "{query_word}"'),
                    use_container_width=True,
                )
            else:
                st.warning(f'"{query_word}" not found in GloVe vocabulary.')

    with tab2:
        st.markdown("#### 🗺️ GloVe Word Clusters")
        default_words = "happy, sad, joyful, angry, excited, depressed, love, hate, peace, war"
        word_input = st.text_input("Enter words (comma-separated):", value=default_words, key="glove_clusters")
        words = [w.strip() for w in word_input.split(",") if w.strip()]

        method = st.radio("Reduction method:", ["PCA", "t-SNE"], horizontal=True, key="glove_method")

        if words and st.button("📊 Visualize", key="glove_viz"):
            valid_words = []
            word_vecs = []
            for w in words:
                try:
                    vec = model[w]
                    valid_words.append(w)
                    word_vecs.append(vec)
                except KeyError:
                    st.warning(f'"{w}" not in GloVe vocabulary, skipping.')

            if len(valid_words) >= 3:
                embeddings = np.array(word_vecs)
                st.plotly_chart(
                    plot_word_clusters(valid_words, embeddings, title="GloVe Word Clusters", method=method.lower()),
                    use_container_width=True,
                )
                st.markdown("*Observe how semantically related words group together — "
                            "emotions cluster, and antonyms form their own patterns!*")
            else:
                st.warning("Need at least 3 valid words.")

    with tab3:
        st.markdown("#### 📊 Sentence Similarity (GloVe)")
        default_sents = (
            "Physics and chemistry are natural sciences\n"
            "Biology studies living organisms\n"
            "Football is a popular team sport\n"
            "Basketball players are tall"
        )
        sent_input = st.text_area("Enter sentences:", value=default_sents, height=100, key="glove_sents")
        sents = [s.strip() for s in sent_input.strip().split("\n") if s.strip()]

        if len(sents) >= 2 and st.button("🔢 Generate Embeddings", key="glove_gen"):
            embeddings = get_glove_embeddings(sents, model=model)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Embedding Dimension", model.vector_size)
            with col2:
                st.metric("Model", "GloVe-Wiki-100d")

            reduced = reduce_dimensions(embeddings, method="pca", n_components=2)
            short_labels = [f"S{i+1}" for i in range(len(sents))]
            st.plotly_chart(
                plot_embeddings_2d(reduced, short_labels, title="GloVe Sentence Embeddings — 2D PCA"),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_similarity_heatmap(embeddings, sents, title="GloVe Sentence Similarity"),
                use_container_width=True,
            )

st.markdown("---")
show_pros_cons(
    pros=[
        "Leverages global co-occurrence statistics",
        "Efficient training on large corpora",
        "Competitive with Word2Vec on most benchmarks",
        "Pre-trained models readily available (6B, 42B, 840B tokens)",
    ],
    cons=[
        "Still one vector per word (no context awareness)",
        "Requires large memory for co-occurrence matrix",
        "Cannot handle OOV words",
        "Fixed vocabulary after training",
    ],
)
