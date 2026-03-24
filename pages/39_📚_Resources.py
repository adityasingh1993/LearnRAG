"""
📚 Resources
A directory of modern SOTA embeddings used for RAG and recommended readings.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header

inject_custom_css()

page_header(
    "Modern RAG Resources", 
    "📚", 
    "State of the Art (SOTA)", 
    "What models are actually powering real-world AI applications today? Explore the top embedding models and further reading."
)

st.markdown("### 🥇 The MTEB Leaderboard")
st.write("Before diving into specific models, you must know about **MTEB (Massive Text Embedding Benchmark)**.")
st.write("MTEB is the gold standard leaderboard hosted on HuggingFace where every new embedding model is aggressively tested on Classification, Clustering, Retrieval (RAG), and Summarization tasks.")
st.markdown("👉 [**View the Live MTEB Leaderboard Here**](https://huggingface.co/spaces/mteb/leaderboard)")

st.markdown("---")

st.markdown("### 🚀 Top Embedding Models for RAG Today")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background: rgba(108, 99, 255, 0.05); border: 1px solid rgba(108, 99, 255, 0.2); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; height: 95%;">
        <h4 style="margin-top:0; color: #FAFAFA;">1. OpenAI Text-Embedding 3</h4>
        <p style="font-size: 0.9rem; color: #a0aec0;">Released in early 2024, <code>text-embedding-3-small</code> and <code>large</code> are the absolute industry standard for enterprise developers due to extreme ease of use, low cost, and native integration into most frameworks.</p>
        <p style="font-size: 0.85rem;"><a href="https://openai.com/blog/new-embedding-models-and-api-updates" style="color: #6C63FF; text-decoration: none;" target="_blank">📘 Official Announcement</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(108, 99, 255, 0.05); border: 1px solid rgba(108, 99, 255, 0.2); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; height: 95%;">
        <h4 style="margin-top:0; color: #FAFAFA;">2. Cohere Embed (v3)</h4>
        <p style="font-size: 0.9rem; color: #a0aec0;">Cohere's models are purpose-built for enterprise RAG and search. Their multilingual model is routinely considered the best on the market for handling over 100 languages natively in a single vector space.</p>
        <p style="font-size: 0.85rem;"><a href="https://cohere.com/blog/introducing-embed-v3" style="color: #6C63FF; text-decoration: none;" target="_blank">📘 Read the V3 Paper</a></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(108, 99, 255, 0.05); border: 1px solid rgba(108, 99, 255, 0.2); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; height: 95%;">
        <h4 style="margin-top:0; color: #FAFAFA;">3. Voyage AI</h4>
        <p style="font-size: 0.9rem; color: #a0aec0;">A rising star founded by Stanford professors, Voyage focuses explicitly on retrieval quality for RAG. They produce domain-specific embeddings (e.g., Voyage-Law, Voyage-Finance) with massive context windows.</p>
        <p style="font-size: 0.85rem;"><a href="https://blog.voyageai.com/2023/10/29/voyage-embeddings/" style="color: #6C63FF; text-decoration: none;" target="_blank">📘 Exploring Voyage</a></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: rgba(78, 203, 113, 0.05); border: 1px solid rgba(78, 203, 113, 0.3); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; height: 95%;">
        <h4 style="margin-top:0; color: #4ECB71;">4. BAAI BGE (Open Source)</h4>
        <p style="font-size: 0.9rem; color: #a0aec0;"><b>BGE (BAAI General Embedding)</b> models consistently dominate the top of the open-source MTEB leaderboard. They are highly efficient, free to run locally, and rival closed APIs.</p>
        <p style="font-size: 0.85rem;"><a href="https://huggingface.co/BAAI/bge-large-en-v1.5" style="color: #4ECB71; text-decoration: none;" target="_blank">🟢 BGE on HuggingFace</a></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(78, 203, 113, 0.05); border: 1px solid rgba(78, 203, 113, 0.3); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; height: 95%;">
        <h4 style="margin-top:0; color: #4ECB71;">5. Nomic Embed (Open Weights)</h4>
        <p style="font-size: 0.9rem; color: #a0aec0;">Nomic is uniquely notable because its weights, training data, and architecture are entirely open-source. It boasts a completely massive <b>8192 token context window</b> while maintaining top-tier accuracy.</p>
        <p style="font-size: 0.85rem;"><a href="https://blog.nomic.ai/posts/nomic-embed-text-v1" style="color: #4ECB71; text-decoration: none;" target="_blank">🟢 Nomic V1 Release</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(78, 203, 113, 0.05); border: 1px solid rgba(78, 203, 113, 0.3); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; height: 95%;">
        <h4 style="margin-top:0; color: #4ECB71;">6. E5 Models (Microsoft OSS)</h4>
        <p style="font-size: 0.9rem; color: #a0aec0;">The <b>E5 (EmbEdder for Everyone)</b> family by Microsoft offers fantastic zero-shot capabilities and comes in highly optimized multi-lingual variants.</p>
        <p style="font-size: 0.85rem;"><a href="https://huggingface.co/intfloat/multilingual-e5-large" style="color: #4ECB71; text-decoration: none;" target="_blank">🟢 E5 on HuggingFace</a></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 📖 Excellent Further Reading & Learning Hubs")
st.markdown("""
If you want to dive deeper into RAG, Vector Databases, and chunking strategies, these are some of the best learning centers on the internet:

* **[Pinecone Learning Center](https://www.pinecone.io/learn/)**: Features some of the best, most beautifully illustrated interactive articles on Vector Databases and Semantic Search.
* **[Milvus Vector DB Glossary](https://milvus.io/docs)**: Deep dive into how vector indexing algorithms like HNSW (Hierarchical Navigable Small World) actually work for split-second retrieval.
* **[HuggingFace NLP Course](https://huggingface.co/learn/nlp-course)**: The definitive free code-focused course for learning Transformer architectures and embeddings directly in Python.
* **[LangChain Docs & Tutorials](https://python.langchain.com/docs/modules/data_connection/)**: A practical guide to plugging these exact embeddings into standard RAG (Retrieval-Augmented Generation) pipelines to build chatbots!
""")
