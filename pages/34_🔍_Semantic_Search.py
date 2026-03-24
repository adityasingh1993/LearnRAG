"""
🔍 Semantic Search vs. Keyword Search
Demonstrates how embeddings power modern search engines and RAG applications.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header
from embedding_utils.embeddings import get_tfidf_embedding, get_transformer_embedding
from sklearn.metrics.pairwise import cosine_similarity
inject_custom_css()

page_header(
    "Semantic Search",
    "🔍",
    "Real-World Usage",
    "Why do we need embeddings? Discover how modern Semantic Search (Transformers) "
    "beats traditional Keyword Search (TF-IDF) by understanding context and synonyms."
)

st.markdown("""
### 🏗️ The Problem with Keyword Search
Traditional search engines (like early Google) relied heavily on **TF-IDF** or BM25. 
If you search for **"laptop"**, it looks for the exact word "laptop". 
If an article says **"Macbook Pro"** but never uses the word "laptop", keyword search will completely miss it!

### 🧠 The Semantic Search Solution
Transformers map both the query and the documents into the *same* dense embedding space. 
Since "laptop" and "Macbook" appear in similar contexts during training, their vectors end up very close to each other. 
By calculating **Cosine Similarity** between the query vector and document vectors, we can find documents that mean the same thing, even if they use different words.

---
""")

# Sample Documents
st.markdown("### 📚 Document Database")
st.markdown("Here is our tiny database of facts:")

documents = [
    "Apple releases the new Macbook Pro with M3 chip.",
    "A golden retriever puppy learns to fetch.",
    "The Federal Reserve announced an interest rate hike today.",
    "Python is a versatile programming language used for data science.",
    "The stock market fell significantly after the inflation report.",
    "Felines are known for their agility and independent nature.",
    "A cozy coffee shop opened downtown serving artisanal espresso.",
    "Dell XPS 15 is a powerful notebook computer for professionals."
]

for i, doc in enumerate(documents):
    st.markdown(f"**Doc {i+1}:** {doc}")

st.markdown("---")

st.markdown("### 🔎 Interactive Search Demo")
query = st.text_input("Enter a search query:", value="Which laptop should I buy?")

if query:
    # 1. TF-IDF (Keyword Search)
    st.markdown(f"#### Searching for: *\"{query}\"*")
    
    col1, col2 = st.columns(2)
    
    # TF-IDF Implementation
    # We must fit the vectorizer on BOTH the documents and the query to have a shared vocabulary
    all_texts = documents + [query]
    tfidf_matrix, features, tfidf_vectorizer = get_tfidf_embedding(all_texts)
    
    doc_tfidf = tfidf_matrix[:-1]
    query_tfidf = tfidf_matrix[-1:]
    
    tfidf_similarities = cosine_similarity(query_tfidf, doc_tfidf).flatten()
    top_tfidf_indices = tfidf_similarities.argsort()[-3:][::-1]
    
    with col1:
        st.markdown("#### ❌ Keyword Search (TF-IDF)")
        st.info("Relies on exact word matches.")
        for idx in top_tfidf_indices:
            score = tfidf_similarities[idx]
            if score > 0:
                st.success(f"**Score: {score:.3f}**  \\n{documents[idx]}")
            else:
                st.warning(f"**Score: {score:.3f}**  \\n{documents[idx]}")
                
    # 2. Semantic Search (Transformers)
    with col2:
        st.markdown("#### ✅ Semantic Search (Transformers)")
        st.info("Understands meaning and context.")
        with st.spinner("Generating Embeddings..."):
            transformer_embeddings = get_transformer_embedding(documents)
            query_embedding = get_transformer_embedding([query])
            
            transformer_similarities = cosine_similarity(query_embedding, transformer_embeddings).flatten()
            top_transformer_indices = transformer_similarities.argsort()[-3:][::-1]
            
            for idx in top_transformer_indices:
                score = transformer_similarities[idx]
                st.success(f"**Score: {score:.3f}**  \\n{documents[idx]}")

st.markdown("---")
st.markdown("""
### 💡 Try these queries:
- `"cat"` (Notice how TF-IDF fails, but Semantic Search finds "Felines")
- `"economic news"` (Finds the Federal Reserve and stock market docs)
- `"programming in python"` (Both might find it, but compare the scores!)
""")
