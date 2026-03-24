"""
🧠 Word2Vec — Dense learned word embeddings.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header, show_pros_cons, FLAT_SAMPLES, SAMPLE_LABELS
from embedding_utils.embeddings import (
    load_word2vec_model,
    get_word2vec_embeddings,
    word2vec_analogy,
    word2vec_most_similar,
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
    "Word2Vec",
    "🧠",
    "2013 (Google)",
    "The breakthrough that revolutionized NLP — learn dense, low-dimensional word vectors from context. "
    'Words used in similar contexts get similar vectors. Famous for "King - Man + Woman = Queen".',
)

# ─── Theory ───
with st.expander("📖 How It Works", expanded=True):
    st.markdown("""
    ### Word2Vec: Learning Word Representations

    **Key idea:** "A word is characterized by the company it keeps" (Firth, 1957)

    **Two architectures:**

    | | CBOW | Skip-gram |
    |---|---|---|
    | **Input** | Context words | Target word |
    | **Output** | Target word | Context words |
    | **Best for** | Frequent words | Rare words |
    | **Speed** | Faster | Slower |

    **How it learns:**
    1. Slide a window over the text
    2. Predict target from context (CBOW) or context from target (Skip-gram)
    3. Adjust weights via backpropagation
    4. After training, the hidden layer weights ARE the word embeddings

    **Result:** 300-dimensional dense vectors where:
    - Similar words are close (cosine similarity)
    - Directions encode meaning (king → queen ≈ man → woman)

    > 💡 **Revolution:** Unlike BoW/TF-IDF, Word2Vec understands that "good" and "great" are similar,
    > even if they never appear in the same document!
    """)

st.markdown("---")

# ─── Load Model ───
st.markdown("### 🎮 Interactive Demos")
st.info("⏳ The Word2Vec model (~1.7GB) will download on first use. Subsequent loads are instant.")

try:
    model = load_word2vec_model()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load Word2Vec model: {e}")
    model_loaded = False

if model_loaded:
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧮 Word Analogies",
        "🔍 Similar Words",
        "🗺️ Word Clusters",
        "📊 Sentence Embeddings",
    ])

    # ─── Tab 1: Analogies ───
    with tab1:
        st.markdown("#### ✨ Word Analogy Explorer")
        st.markdown('*Solve equations like: **King - Man + Woman = ?***')

        col1, col2, col3 = st.columns(3)
        with col1:
            pos1 = st.text_input("Positive word 1:", value="king", key="w2v_pos1")
        with col2:
            neg1 = st.text_input("Negative word:", value="man", key="w2v_neg")
        with col3:
            pos2 = st.text_input("Positive word 2:", value="woman", key="w2v_pos2")

        st.markdown(f"**Equation:** *{pos1}* − *{neg1}* + *{pos2}* = **?**")

        if st.button("🚀 Solve Analogy", key="w2v_solve"):
            results = word2vec_analogy([pos1, pos2], [neg1], model=model)
            if results:
                words, scores = zip(*results)
                st.plotly_chart(
                    plot_analogy_result(list(words), list(scores), title=f"{pos1} − {neg1} + {pos2} = ?"),
                    use_container_width=True,
                )
                st.success(f"🎯 Top answer: **{results[0][0]}** (score: {results[0][1]:.4f})")
            else:
                st.warning("One or more words not found in vocabulary.")

        st.markdown("---")
        st.markdown("**Try these classic analogies:**")
        examples = st.columns(3)
        with examples[0]:
            st.markdown("- `paris - france + italy` = rome")
        with examples[1]:
            st.markdown("- `king - man + woman` = queen")
        with examples[2]:
            st.markdown("- `walking - walk + swim` = swimming")

    # ─── Tab 2: Similar Words ───
    with tab2:
        st.markdown("#### 🔍 Find Similar Words")
        query_word = st.text_input("Enter a word:", value="computer", key="w2v_similar")

        if st.button("🔎 Find Similar", key="w2v_find"):
            results = word2vec_most_similar(query_word, model=model, topn=15)
            if results:
                words, scores = zip(*results)
                st.plotly_chart(
                    plot_analogy_result(list(words), list(scores), title=f'Words similar to "{query_word}"'),
                    use_container_width=True,
                )
            else:
                st.warning(f'"{query_word}" not found in vocabulary.')

    # ─── Tab 3: Word Clusters ───
    with tab3:
        st.markdown("#### 🗺️ Visualize Word Clusters")
        st.markdown("*See how semantically related words group together in embedding space:*")

        default_words = "king, queen, prince, princess, man, woman, boy, girl, cat, dog, puppy, kitten"
        word_input = st.text_input("Enter words (comma-separated):", value=default_words, key="w2v_clusters")
        words = [w.strip() for w in word_input.split(",") if w.strip()]

        method = st.radio("Reduction method:", ["PCA", "t-SNE"], horizontal=True, key="w2v_method")

        if words and st.button("📊 Visualize", key="w2v_viz"):
            valid_words = []
            word_vecs = []
            for w in words:
                try:
                    vec = model[w]
                    valid_words.append(w)
                    word_vecs.append(vec)
                except KeyError:
                    st.warning(f'"{w}" not in vocabulary, skipping.')

            if len(valid_words) >= 3:
                embeddings = np.array(word_vecs)
                st.plotly_chart(
                    plot_word_clusters(valid_words, embeddings, title="Word2Vec Clusters", method=method.lower()),
                    use_container_width=True,
                )
            else:
                st.warning("Need at least 3 valid words.")

    # ─── Tab 4: Sentence Embeddings ───
    with tab4:
        st.markdown("#### 📊 Sentence Embeddings (Average Word Vectors)")
        st.markdown("*Word2Vec gives word vectors — for sentences, we average all word vectors:*")

        default_sents = "I love programming\nCoding is fun\nThe weather is nice today\nIt is sunny outside"
        sent_input = st.text_area("Enter sentences:", value=default_sents, height=100, key="w2v_sents")
        sents = [s.strip() for s in sent_input.strip().split("\n") if s.strip()]

        if len(sents) >= 2 and st.button("🔢 Generate Embeddings", key="w2v_gen"):
            embeddings = get_word2vec_embeddings(sents, model=model)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Embedding Dimension", model.vector_size)
            with col2:
                st.metric("Dense Values", "100%", help="Unlike BoW, no zeros!")

            reduced = reduce_dimensions(embeddings, method="pca", n_components=2)
            short_labels = [f"S{i+1}" for i in range(len(sents))]

            st.plotly_chart(
                plot_embeddings_2d(reduced, short_labels, title="Sentence Embeddings — 2D PCA"),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_similarity_heatmap(embeddings, sents, title="Sentence Similarity (Word2Vec Avg)"),
                use_container_width=True,
            )

# ─── Pros/Cons ───
st.markdown("---")
show_pros_cons(
    pros=[
        "Captures semantic meaning — similar words cluster",
        "Dense, low-dimensional vectors (300 dims vs 10k+ for BoW)",
        "Supports word analogies and arithmetic",
        "Transfer learning — pre-trained models work out of the box",
    ],
    cons=[
        "One vector per word — ignores context (polysemy)",
        '"bank" gets the same vector for finance and river',
        "Can't handle out-of-vocabulary (OOV) words",
        "Requires large training corpus",
    ],
)
