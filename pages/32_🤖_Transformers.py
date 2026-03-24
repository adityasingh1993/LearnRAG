"""
🤖 Transformers — Contextual embeddings that understand meaning in context.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header, show_pros_cons, POLYSEMY_EXAMPLES
from embedding_utils.embeddings import load_transformer_model, get_transformer_embedding
from embedding_utils.visualization import (
    plot_embeddings_2d,
    plot_similarity_heatmap,
    plot_embedding_vector,
    reduce_dimensions,
)
from sklearn.metrics.pairwise import cosine_similarity

inject_custom_css()

page_header(
    "Transformer Embeddings",
    "🤖",
    "2018+ (Google, OpenAI)",
    "The current state of the art — contextual embeddings where the SAME word gets DIFFERENT vectors "
    "depending on its surrounding context. Powered by the attention mechanism.",
)

# ─── Theory ───
with st.expander("📖 How It Works", expanded=True):
    st.markdown("""
    ### Transformer-Based Contextual Embeddings

    **The fundamental problem with all previous methods:**
    > "I went to the **bank** to deposit money" vs "I sat on the river **bank**"
    >
    > Word2Vec, GloVe, FastText → **same vector** for "bank" in both sentences! ❌

    **Transformer solution:** Process the **entire sentence** and generate a different
    embedding for each word based on its context.

    **How Transformers work:**

    1. **Tokenization** — Break text into subword tokens
    2. **Self-Attention** — Each token attends to every other token
       - "bank" attends to "deposit", "money" → financial meaning
       - "bank" attends to "river", "sat" → nature meaning
    3. **Multiple layers** — Stack attention layers for deeper understanding
    4. **Output** — Context-aware embedding for each token

    **Key models:**
    | Model | Year | Key Innovation |
    |---|---|---|
    | BERT | 2018 | Bidirectional context |
    | GPT-2 | 2019 | Unidirectional, generative |
    | RoBERTa | 2019 | Optimized BERT training |
    | Sentence-BERT | 2019 | Efficient sentence embeddings |

    > 💡 **We use Sentence-BERT (all-MiniLM-L6-v2)** — a lightweight 384-dim model
    > optimized for sentence-level similarity.
    """)

st.markdown("---")
st.markdown("### 🎮 Interactive Demos")

try:
    model = load_transformer_model()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load Transformer model: {e}")
    model_loaded = False

if model_loaded:
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔀 Context Matters!",
        "📊 Sentence Similarity",
        "🔢 Vector Inspector",
        "🗺️ Embedding Space",
    ])

    # ─── Tab 1: Polysemy Demo ───
    with tab1:
        st.markdown("#### 🔀 Same Word, Different Meaning, Different Embedding!")
        st.markdown("*The killer feature of transformer embeddings — context changes everything:*")

        # Pre-built examples
        example_choice = st.selectbox(
            "Choose an example or type your own:",
            ["Custom"] + [f'"{ex["word"]}" — {ex["context_a"]} vs {ex["context_b"]}' for ex in POLYSEMY_EXAMPLES],
            key="transformer_example",
        )

        if example_choice == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                sent_a = st.text_input("Sentence A:", value="I deposited money at the bank", key="t_sent_a")
            with col2:
                sent_b = st.text_input("Sentence B:", value="We sat by the river bank", key="t_sent_b")
        else:
            idx = [f'"{ex["word"]}" — {ex["context_a"]} vs {ex["context_b"]}' for ex in POLYSEMY_EXAMPLES].index(example_choice)
            ex = POLYSEMY_EXAMPLES[idx]
            sent_a = ex["sentence_a"]
            sent_b = ex["sentence_b"]
            st.markdown(f"**Sentence A ({ex['context_a']}):** {sent_a}")
            st.markdown(f"**Sentence B ({ex['context_b']}):** {sent_b}")

        if st.button("🔍 Compare Context", key="t_compare"):
            emb_a = get_transformer_embedding([sent_a], model=model)[0]
            emb_b = get_transformer_embedding([sent_b], model=model)[0]
            sim = cosine_similarity([emb_a], [emb_b])[0][0]

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1a1d23, #2d1f4e);
                    border: 1px solid rgba(108, 99, 255, 0.3);
                    border-radius: 16px;
                    padding: 2rem;
                    text-align: center;
                    margin: 1rem 0;
                ">
                    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px; padding: 1rem;">
                            <div style="color: #B794F6; font-size: 0.9rem; margin-bottom: 0.5rem;">SENTENCE A</div>
                            <div style="color: #FAFAFA; font-size: 1.1rem;">"{sent_a}"</div>
                        </div>
                        <div style="padding: 1rem;">
                            <div style="
                                font-size: 3rem;
                                font-weight: 700;
                                color: {'#4ECB71' if sim > 0.7 else '#FFB347' if sim > 0.4 else '#FF6B6B'};
                            ">{sim:.4f}</div>
                            <div style="color: #a0aec0;">Cosine Similarity</div>
                        </div>
                        <div style="flex: 1; min-width: 200px; padding: 1rem;">
                            <div style="color: #B794F6; font-size: 0.9rem; margin-bottom: 0.5rem;">SENTENCE B</div>
                            <div style="color: #FAFAFA; font-size: 1.1rem;">"{sent_b}"</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if sim < 0.5:
                st.success("🎯 The transformer correctly distinguishes these different contexts! "
                           "The same word gets very different embeddings.")
            elif sim < 0.7:
                st.info("📊 The sentences are somewhat similar in the transformer's view.")
            else:
                st.info("📊 The transformer sees these as quite similar in meaning.")

    # ─── Tab 2: Sentence Similarity ───
    with tab2:
        st.markdown("#### 📊 Sentence Similarity")
        st.markdown("*Transformer embeddings excel at capturing semantic similarity:*")

        default_sents = (
            "I love dogs, they are great pets\n"
            "Canines make wonderful companions\n"
            "The stock market crashed today\n"
            "Financial markets experienced a downturn\n"
            "It is raining heavily outside\n"
            "The weather is quite bad with heavy rain"
        )
        sent_input = st.text_area("Enter sentences:", value=default_sents, height=140, key="t_sents")
        sents = [s.strip() for s in sent_input.strip().split("\n") if s.strip()]

        if len(sents) >= 2 and st.button("🔢 Generate & Compare", key="t_gen"):
            embeddings = get_transformer_embedding(sents, model=model)

            cols = st.columns(3)
            with cols[0]:
                st.metric("Embedding Dimension", embeddings.shape[1])
            with cols[1]:
                st.metric("Model", "all-MiniLM-L6-v2")
            with cols[2]:
                st.metric("Context-Aware", "✅ Yes")

            st.plotly_chart(
                plot_similarity_heatmap(embeddings, sents, title="Transformer Sentence Similarity"),
                use_container_width=True,
            )

            st.info("🔍 Notice how semantically similar sentences (like paraphrases) have HIGH similarity, "
                    "even when they use completely different words!")

    # ─── Tab 3: Vector Inspector ───
    with tab3:
        st.markdown("#### 🔢 Inspect Raw Embedding Values")
        inspect_text = st.text_input("Enter a sentence:", value="The transformer model is amazing", key="t_inspect")

        if st.button("🔍 Inspect", key="t_inspect_btn"):
            emb = get_transformer_embedding([inspect_text], model=model)[0]

            cols = st.columns(3)
            with cols[0]:
                st.metric("Dimensions", len(emb))
            with cols[1]:
                st.metric("Min Value", f"{emb.min():.4f}")
            with cols[2]:
                st.metric("Max Value", f"{emb.max():.4f}")

            st.plotly_chart(
                plot_embedding_vector(emb, title=f'Embedding Vector (first 50 dims)', max_features=50),
                use_container_width=True,
            )
            st.markdown("*Each dimension captures some aspect of the sentence's meaning. "
                        "Unlike BoW, ALL dimensions have non-zero values (dense vectors).*")

    # ─── Tab 4: Embedding Space ───
    with tab4:
        st.markdown("#### 🗺️ Sentence Embedding Space")
        st.markdown("*See how sentences cluster by meaning:*")

        default_space = (
            "Dogs are loyal animals\n"
            "Cats are independent pets\n"
            "Python is a programming language\n"
            "JavaScript is used for web development\n"
            "Pizza is my favorite food\n"
            "I love eating sushi\n"
            "The sun is shining brightly\n"
            "It is a beautiful sunny day"
        )
        space_input = st.text_area("Enter sentences:", value=default_space, height=140, key="t_space")
        space_sents = [s.strip() for s in space_input.strip().split("\n") if s.strip()]

        if len(space_sents) >= 3 and st.button("🗺️ Visualize Space", key="t_space_btn"):
            embeddings = get_transformer_embedding(space_sents, model=model)

            method = st.radio("Reduction:", ["PCA", "t-SNE"], horizontal=True, key="t_space_method")
            reduced = reduce_dimensions(embeddings, method=method.lower(), n_components=2)
            short_labels = [f"S{i+1}: {s[:20]}..." if len(s) > 20 else f"S{i+1}: {s}" for i, s in enumerate(space_sents)]

            st.plotly_chart(
                plot_embeddings_2d(reduced, short_labels, title="Transformer Embedding Space"),
                use_container_width=True,
            )
            st.markdown("*Sentences about similar topics cluster together, even with different wording!*")

st.markdown("---")
show_pros_cons(
    pros=[
        "Context-aware — same word, different embedding based on context",
        "State-of-the-art performance on nearly all NLP tasks",
        "Handles polysemy, idioms, and nuanced language",
        "Pre-trained on massive corpora, easy to fine-tune",
    ],
    cons=[
        "Computationally expensive (needs GPU for large models)",
        "Large model sizes (100MB to 100GB+)",
        "Slower inference than static embeddings",
        "Can be overkill for simple tasks",
    ],
)
