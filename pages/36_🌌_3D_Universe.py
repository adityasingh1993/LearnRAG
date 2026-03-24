"""
🌌 3D Universe of Embeddings
Interactive 3D scatter plot of word embeddings from user-provided texts.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header
from embedding_utils.embeddings import (
    get_word2vec_word_embedding, get_word2vec_sentence_embedding,
    get_glove_word_embedding, get_glove_sentence_embedding,
    get_fasttext_word_embedding, get_fasttext_sentence_embedding,
    get_transformer_embedding, load_transformer_model,
    get_tfidf_embedding
)
from embedding_utils.visualization import reduce_dimensions, plot_embeddings_3d
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
inject_custom_css()

page_header(
    "3D Universe of Embeddings", 
    "🌌", 
    "Interactive WebGL", 
    "Type your own texts and see how different models map each word in 3D space!"
)

st.sidebar.markdown("### ⚙️ Universe Settings")
model_choice = st.sidebar.radio("Select Model:", ["TF-IDF", "Word2Vec", "GloVe", "Transformers"])
dim_reduction = st.sidebar.radio("Dimensionality Reduction:", ["PCA", "t-SNE (Slower, finer clusters)"])

st.markdown("### 📝 Enter Texts")
st.markdown("Enter at least two texts. We will graph the individual words from each text. Notice how **Transformers** place the *same* word (e.g., 'bank') in *different* places depending on its context, while older models place it in the exact same spot!")

col1, col2 = st.columns(2)
with col1:
    text1 = st.text_area("Text 1:", value="The river bank is covered in green grass.", height=80)
with col2:
    text2 = st.text_area("Text 2:", value="The bank of India issued a new currency note.", height=80)

texts = [t.strip() for t in [text1, text2] if t.strip()]

# Function to compute session state data
def compute_data():
    if len(texts) < 2:
        st.warning("Please enter at least two texts to compare.")
        return
        
    import re
    def tokenize(text):
        return [w.lower() for w in re.findall(r'\b\w+\b', text)]
        
    words1 = tokenize(texts[0])
    words2 = tokenize(texts[1])
    all_unique_words = list(set(words1) | set(words2))
    
    # Store sentence similarity
    sim_score = 0.0
    try:
        if model_choice == "TF-IDF":
            matrix, _, _ = get_tfidf_embedding(texts)
            sim_score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        elif model_choice == "Word2Vec":
            vec1 = get_word2vec_sentence_embedding(texts[0])
            vec2 = get_word2vec_sentence_embedding(texts[1])
            sim_score = cosine_similarity([vec1], [vec2])[0][0]
        elif model_choice == "GloVe":
            vec1 = get_glove_sentence_embedding(texts[0])
            vec2 = get_glove_sentence_embedding(texts[1])
            sim_score = cosine_similarity([vec1], [vec2])[0][0]
        elif model_choice == "Transformers":
            matrix = get_transformer_embedding(texts)
            sim_score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    except Exception as e:
        sim_score = None
        
    st.session_state.sim_score = sim_score
    st.session_state.processed_model = model_choice
    st.session_state.processed_texts = texts
    
    embeddings = []
    valid_words = []
    valid_colors = []
    
    if model_choice == "Transformers":
        import torch
        from transformers import AutoTokenizer, AutoModel
        hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        hf_model = AutoModel.from_pretrained(hf_model_name)
        
        def get_contextual_tokens(text, label):
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = hf_model(**inputs)
            token_embeddings = outputs.last_hidden_state[0].cpu().numpy()
            input_ids = inputs["input_ids"][0].cpu().numpy()
            
            # Simple dedup strategy for exact same string at same position so UI list doesn't get messed up if duplicate tokens occur.
            for i, tid in enumerate(input_ids):
                token_str = tokenizer.decode([tid]).strip()
                if token_str and token_str not in ["[CLS]", "[SEP]", "<s>", "</s>", "<pad>"]:
                    word_label = f"{token_str} ({label} | Pos:{i})"
                    valid_words.append(word_label)
                    embeddings.append(token_embeddings[i])
                    valid_colors.append(label)
                    
        get_contextual_tokens(texts[0], "Text 1 Only")
        get_contextual_tokens(texts[1], "Text 2 Only")
        
    else:
        # non-contextual
        def process_static_words(words_list, label):
            if model_choice == "TF-IDF":
                matrix, features, _ = get_tfidf_embedding(texts)
                word_to_vec = {f: matrix[:, i] for i, f in enumerate(features)}
                for w in words_list:
                    if w in word_to_vec:
                        valid_words.append(f"{w} [{label}]")
                        embeddings.append(word_to_vec[w])
                        valid_colors.append(label)
            else:
                for w in words_list:
                    vec = get_word2vec_word_embedding(w) if model_choice == "Word2Vec" else get_glove_word_embedding(w)
                    if vec is not None:
                        valid_words.append(f"{w} [{label}]")
                        embeddings.append(vec)
                        valid_colors.append(label)
                        
        words1_set = list(set(words1))
        words2_set = list(set(words2))

        process_static_words(words1_set, "Text 1 Only")
        process_static_words(words2_set, "Text 2 Only")

    st.session_state.embeddings = np.array(embeddings) if embeddings else np.empty((0,))
    st.session_state.valid_words = valid_words
    st.session_state.valid_colors = valid_colors


if st.button("🚀 Process & Generate 3D Map", type="primary"):
    with st.spinner(f"Generating embeddings using {model_choice}..."):
        compute_data()

# Render State
if "embeddings" in st.session_state and st.session_state.embeddings.shape[0] > 0:
    
    st.markdown("---")
    sim_score = st.session_state.sim_score
    if sim_score is not None:
        st.markdown(f"### 📊 Sentence Similarity ({st.session_state.processed_model})")
        st.info(f"The cosine similarity between Text 1 and Text 2 is: **{sim_score:.4f}**")
        
    st.markdown("---")
    st.markdown("### 🌌 Interactive 3D Word Universe")
    
    valid_words = st.session_state.valid_words
    valid_colors = st.session_state.valid_colors
    embeddings = st.session_state.embeddings
    
    # 1. Multiselect for words
    selected_words = st.multiselect(
        "Select specific words to view on the map and their raw vectors:",
        options=valid_words,
        default=valid_words
    )
    
    if not selected_words:
        st.warning("Please select at least one word to display.")
    else:
        # Filter data based on selection
        selected_indices = [valid_words.index(w) for w in selected_words]
        
        filtered_words = [valid_words[i] for i in selected_indices]
        filtered_colors = [valid_colors[i] for i in selected_indices]
        filtered_embeddings = embeddings[selected_indices]
        
        # 2. Raw vectors table
        with st.expander("👀 View Raw Vectors", expanded=False):
            # Limit vector preview to first 10 dimensions for readability
            max_dims = min(10, filtered_embeddings.shape[1])
            df_data = {
                "Word": filtered_words,
                "Origin": filtered_colors,
                **{f"Dim {i}": filtered_embeddings[:, i] for i in range(max_dims)}
            }
            if filtered_embeddings.shape[1] > 10:
                df_data["..."] = ["..." for _ in range(len(filtered_words))]
            
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)
            
        # 3. 3D Map rendering
        method_str = "tsne" if "t-SNE" in dim_reduction else "pca"
        reduced = reduce_dimensions(filtered_embeddings, method=method_str, n_components=3)
        
        if reduced.shape[1] < 3:
            padding = np.zeros((reduced.shape[0], 3 - reduced.shape[1]))
            reduced = np.hstack((reduced, padding))
            
        fig = plot_embeddings_3d(
            reduced=reduced,
            labels=filtered_words,
            title=f"3D Word Map ({st.session_state.processed_model})",
            color_labels=filtered_colors
        )
        
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color="#FAFAFA")))
        st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.processed_model == "Transformers":
            st.success("✅ **Notice:** In Transformers, if you look at the same word from Text 1 and Text 2, they appear in **different locations** because Transformers embed the word based on its unique sentence context!")
        else:
            st.warning(f"⚠️ **Notice:** In {st.session_state.processed_model}, if a word appears in both texts, both points will overlap perfectly at the **exact same location** because older models give words a single static meaning regardless of context.")
