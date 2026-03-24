"""
Embedding generation module.
Provides functions for all embedding techniques: BoW, TF-IDF, Word2Vec, GloVe, FastText, Transformers.
"""
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ═══════════════════════════════════════════════════════════
# 1. BAG OF WORDS
# ═══════════════════════════════════════════════════════════

def get_bow_embedding(texts: list[str]):
    """
    Generate Bag of Words embeddings.
    Returns: (matrix, feature_names, vectorizer)
    """
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer.get_feature_names_out(), vectorizer


def get_bow_details(texts: list[str]):
    """Get detailed BoW info including vocabulary and document vectors."""
    matrix, features, vectorizer = get_bow_embedding(texts)
    return {
        "matrix": matrix,
        "features": features,
        "vocab_size": len(features),
        "sparsity": 1.0 - (np.count_nonzero(matrix) / matrix.size),
        "similarity": cosine_similarity(matrix),
    }


# ═══════════════════════════════════════════════════════════
# 2. TF-IDF
# ═══════════════════════════════════════════════════════════

def get_tfidf_embedding(texts: list[str]):
    """
    Generate TF-IDF embeddings.
    Returns: (matrix, feature_names, vectorizer)
    """
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer.get_feature_names_out(), vectorizer


def get_tfidf_details(texts: list[str]):
    """Get detailed TF-IDF info with IDF values."""
    matrix, features, vectorizer = get_tfidf_embedding(texts)
    return {
        "matrix": matrix,
        "features": features,
        "idf_values": dict(zip(features, vectorizer.idf_)),
        "vocab_size": len(features),
        "similarity": cosine_similarity(matrix),
    }


# ═══════════════════════════════════════════════════════════
# 3. WORD2VEC
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Word2Vec model (this may take a minute)...")
def load_word2vec_model():
    """Load a pre-trained Word2Vec model via Gensim."""
    import gensim.downloader as api
    return api.load("word2vec-google-news-300")


def get_word2vec_word_embedding(word: str, model=None):
    """Get Word2Vec embedding for a single word."""
    if model is None:
        model = load_word2vec_model()
    try:
        return model[word]
    except KeyError:
        return None


def get_word2vec_sentence_embedding(text: str, model=None):
    """Average Word2Vec vectors for all words in a sentence."""
    if model is None:
        model = load_word2vec_model()
    words = text.lower().split()
    vectors = []
    for w in words:
        try:
            vectors.append(model[w])
        except KeyError:
            continue
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def get_word2vec_embeddings(texts: list[str], model=None):
    """Get sentence embeddings for multiple texts."""
    if model is None:
        model = load_word2vec_model()
    embeddings = np.array([get_word2vec_sentence_embedding(t, model) for t in texts])
    return embeddings


def word2vec_analogy(positive: list[str], negative: list[str], model=None, topn=5):
    """Solve word analogies (e.g., king - man + woman = queen)."""
    if model is None:
        model = load_word2vec_model()
    try:
        results = model.most_similar(positive=positive, negative=negative, topn=topn)
        return results
    except KeyError as e:
        return None


def word2vec_most_similar(word: str, model=None, topn=10):
    """Find most similar words."""
    if model is None:
        model = load_word2vec_model()
    try:
        return model.most_similar(word, topn=topn)
    except KeyError:
        return None


# ═══════════════════════════════════════════════════════════
# 4. GLOVE
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading GloVe model (this may take a minute)...")
def load_glove_model():
    """Load a pre-trained GloVe model via Gensim."""
    import gensim.downloader as api
    return api.load("glove-wiki-gigaword-100")


def get_glove_word_embedding(word: str, model=None):
    """Get GloVe embedding for a single word."""
    if model is None:
        model = load_glove_model()
    try:
        return model[word]
    except KeyError:
        return None


def get_glove_sentence_embedding(text: str, model=None):
    """Average GloVe vectors for all words in a sentence."""
    if model is None:
        model = load_glove_model()
    words = text.lower().split()
    vectors = []
    for w in words:
        try:
            vectors.append(model[w])
        except KeyError:
            continue
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def get_glove_embeddings(texts: list[str], model=None):
    """Get sentence embeddings for multiple texts."""
    if model is None:
        model = load_glove_model()
    return np.array([get_glove_sentence_embedding(t, model) for t in texts])


def glove_most_similar(word: str, model=None, topn=10):
    """Find most similar words using GloVe."""
    if model is None:
        model = load_glove_model()
    try:
        return model.most_similar(word, topn=topn)
    except KeyError:
        return None


# ═══════════════════════════════════════════════════════════
# 5. FASTTEXT
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading FastText model (this may take a minute)...")
def load_fasttext_model():
    """Load a pre-trained FastText model via Gensim."""
    import gensim.downloader as api
    return api.load("fasttext-wiki-news-subwords-300")


def get_fasttext_word_embedding(word: str, model=None):
    """Get FastText embedding for a word (works with OOV too)."""
    if model is None:
        model = load_fasttext_model()
    try:
        return model[word]
    except KeyError:
        return None


def get_fasttext_sentence_embedding(text: str, model=None):
    """Average FastText vectors for all words in a sentence."""
    if model is None:
        model = load_fasttext_model()
    words = text.lower().split()
    vectors = []
    for w in words:
        try:
            vectors.append(model[w])
        except KeyError:
            continue
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def get_fasttext_embeddings(texts: list[str], model=None):
    """Get sentence embeddings for multiple texts."""
    if model is None:
        model = load_fasttext_model()
    return np.array([get_fasttext_sentence_embedding(t, model) for t in texts])


# ═══════════════════════════════════════════════════════════
# 6. TRANSFORMER (BERT / Sentence-Transformers)
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Transformer model...")
def load_transformer_model():
    """Load a sentence-transformers model (small & fast)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_transformer_embedding(texts: list[str], model=None):
    """Get transformer-based sentence embeddings."""
    if model is None:
        model = load_transformer_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings)


def get_transformer_word_in_context(sentences: list[str], model=None):
    """
    Get embeddings for sentences to demonstrate contextual differences.
    Returns embeddings for each sentence.
    """
    if model is None:
        model = load_transformer_model()
    return model.encode(sentences, show_progress_bar=False)
