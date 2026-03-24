"""
🎮 Guess the Embedding
An interactive mini-game to test user's understanding of different embeddings.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header
from embedding_utils.embeddings import (
    get_bow_embedding, get_tfidf_embedding, get_word2vec_embeddings,
    get_glove_embeddings, get_fasttext_embeddings, get_transformer_embedding
)
from embedding_utils.visualization import plot_similarity_heatmap

inject_custom_css()

page_header(
    "Guess the Embedding",
    "🎮",
    "Mini-Game",
    "Test your knowledge! Can you identify which embedding technique produced these similarities?"
)

# Test vocabulary/sentences carefully chosen to highlight differences
test_items = [
    "Apple releases new Macbook.",         # 0
    "I ate a juicy red apple.",            # 1 (Same word 'apple', totally different meaning)
    "Microsoft announces new Surface.",    # 2 (Similar meaning to 0, no vocabulary overlap)
    "The dog chased the cat.",             # 3
    "Felines are independent animals."     # 4 (Semantic match to 3, no vocabulary overlap)
]

techniques = ["Bag of Words", "TF-IDF", "Word2Vec", "GloVe", "FastText", "Transformers"]

if "mystery_technique" not in st.session_state:
    st.session_state.mystery_technique = random.choice(techniques)
    st.session_state.game_won = False
    
def reset_game():
    st.session_state.mystery_technique = random.choice(techniques)
    st.session_state.game_won = False

# Compute similarities for the mystery technique
def get_similarities(technique, items):
    if technique == "Bag of Words":
        matrix, _, _ = get_bow_embedding(items)
    elif technique == "TF-IDF":
        matrix, _, _ = get_tfidf_embedding(items)
    elif technique == "Word2Vec":
        matrix = get_word2vec_embeddings(items)
    elif technique == "GloVe":
        matrix = get_glove_embeddings(items)
    elif technique == "FastText":
        matrix = get_fasttext_embeddings(items)
    elif technique == "Transformers":
        matrix = get_transformer_embedding(items)
    else:
        matrix = np.eye(len(items))
        
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(matrix)

with st.spinner("Generating mystery matrix (might take a moment to load models...)"):
    sim_matrix = get_similarities(st.session_state.mystery_technique, test_items)

st.markdown("### 🕵️‍♂️ The Mystery Matrix")
st.markdown("Below is the cosine similarity matrix for 5 key sentences using a **mystery embedding technique**. Look closely at the similarities between sentences that share words vs. sentences that share meaning.")

st.plotly_chart(
    plot_similarity_heatmap(sim_matrix, test_items, title="Mystery Similarity Matrix"),
    use_container_width=True
)

st.markdown("### 🤔 Your Guess")
cols = st.columns(3)
with cols[0]:
    guess = st.selectbox("Which technique generated this matrix?", ["Select..."] + techniques, key="guess_select")

if guess != "Select...":
    if guess == st.session_state.mystery_technique:
        st.success(f"🎉 Correct! It was **{st.session_state.mystery_technique}**.")
        st.session_state.game_won = True
        st.balloons()
    else:
        st.error("❌ Not quite! Take another look.")

if st.session_state.game_won:
    st.button("Play Again", on_click=reset_game, type="primary")
else:
    st.button("Give up & Reveal", on_click=lambda: st.info(f"The answer was **{st.session_state.mystery_technique}**."))

st.markdown("---")
with st.expander("💡 Tips on Identifying Techniques"):
    st.markdown("""
    - **Bag of Words / TF-IDF**: Look for **0.00 similarity** between sentences with no overlapping words (e.g., Sentence 0 and 2, Sentence 3 and 4). Also, look for artificial high similarity simply because of the word "apple" (Sentence 0 and 1).
    - **Word2Vec / GloVe / FastText**: Uses average of word vectors. Sentence 0 and 2 might have moderate similarity now. 
    - **Transformers**: True semantic understanding. Sentence 3 and 4 should have very high similarity. Sentence 0 and 1 should have much lower similarity because the *context* of "apple" is differentiated!
    """)
