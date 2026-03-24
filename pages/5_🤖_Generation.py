"""
Module 5: Generation
See how LLMs use retrieved context to generate grounded answers.
"""

import streamlit as st


from components.sidebar import render_provider_config, get_llm_provider, get_embedding_provider
from core.vector_store import NumpyVectorStore

render_provider_config()

st.title("🤖 Generation")
st.markdown("*The final step: LLMs generate answers grounded in retrieved context.*")
st.markdown("---")

# ── Concept ──────────────────────────────────────────────────────────────
st.header("How Generation Works in RAG")

st.markdown("""
After retrieval gives us relevant chunks, we build a **prompt** that includes both
the retrieved context and the user's question. The LLM then generates an answer
based on this enriched prompt.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    #### Without RAG (Plain LLM)
    ```
    User: What is HNSW?

    LLM: I think HNSW might be... [may hallucinate]
    ```
    """)
with col2:
    st.markdown("""
    #### With RAG (Context-Augmented)
    ```
    System: Answer based on this context:
    "HNSW (Hierarchical Navigable Small World)
     is an algorithm for approximate nearest
     neighbor search in high dimensions..."

    User: What is HNSW?

    LLM: HNSW is an algorithm for approximate
         nearest neighbor search... [grounded!]
    ```
    """)

# ── Prompt Templates ─────────────────────────────────────────────────────
st.markdown("---")
st.header("Prompt Engineering for RAG")

st.markdown("The prompt template is crucial. It tells the LLM how to use the context.")

from core.rag_pipeline import PROMPT_TEMPLATES, REASONING_MODES

templates = {
    "Standard": PROMPT_TEMPLATES["standard"],
    "Chain-of-Thought (CoT)": PROMPT_TEMPLATES["cot"],
    "Analysis": PROMPT_TEMPLATES["analysis"],
    "CoT + Analysis": PROMPT_TEMPLATES["cot_analysis"],
    "Step-Back Prompting": PROMPT_TEMPLATES["step_back"],
    "Self-Reflect": PROMPT_TEMPLATES["self_reflect"],
    "Detailed with Citations": (
        "You are a precise research assistant. Answer the question using ONLY the provided "
        "context. For each claim, cite which chunk it came from (e.g., [Chunk 1]). "
        "If the context doesn't contain the answer, clearly state that.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a detailed answer with citations:"
    ),
    "Concise": (
        "Based on the context below, give a brief, direct answer. "
        "If unsure, say so.\n\n"
        "Context: {context}\n\n"
        "Q: {question}\nA:"
    ),
}

selected_template = st.selectbox("Choose a prompt template:", list(templates.keys()), key="gen_template")

if st.session_state.get("_prev_gen_template") != selected_template:
    st.session_state["_prev_gen_template"] = selected_template
    st.session_state["gen_prompt"] = templates[selected_template]

template_text = st.text_area("Prompt template (editable):", height=200, key="gen_prompt")

st.info("**CoT** = the LLM reasons step-by-step before answering. "
        "**Analysis** = the LLM evaluates each chunk's relevance first. "
        "**CoT + Analysis** = both combined for maximum depth.")

# ── Interactive Demo ─────────────────────────────────────────────────────
st.markdown("---")
st.header("🔬 Interactive: See Generation in Action")

st.markdown("We'll retrieve context and then generate an answer, showing you the full prompt.")

context_docs = st.text_area(
    "Knowledge base (one fact per line):",
    value=(
        "RAG stands for Retrieval-Augmented Generation, a technique to enhance LLMs.\n"
        "RAG works by retrieving relevant documents and including them in the LLM prompt.\n"
        "Vector stores like ChromaDB enable fast similarity search over embeddings.\n"
        "Embeddings are numerical representations of text that capture semantic meaning.\n"
        "HNSW is an algorithm used by vector databases for approximate nearest neighbor search.\n"
        "Cosine similarity measures the angle between two vectors, commonly used for embedding comparison.\n"
        "Chunking splits documents into smaller pieces, typically 200-1000 characters each.\n"
        "Fine-tuning changes model weights, while RAG adds external knowledge at inference time."
    ),
    height=180,
    key="gen_kb",
)

question = st.text_input("Your question:", "Explain how RAG works and why it's better than fine-tuning", key="gen_q")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature:", 0.0, 2.0, 0.7, 0.1, key="gen_temp")
with col2:
    max_tokens = st.slider("Max tokens:", 64, 2048, 512, key="gen_max_tokens")

k_results = st.slider("Number of chunks to retrieve:", 1, 8, 3, key="gen_k")

if st.button("🤖 Retrieve & Generate", type="primary") and question:
    docs = [d.strip() for d in context_docs.strip().split("\n") if d.strip()]

    with st.spinner("Step 1: Embedding & Retrieval..."):
        try:
            embed_provider = get_embedding_provider()
            doc_embeddings = embed_provider.embed(docs)
            query_emb = embed_provider.embed_query(question)

            store = NumpyVectorStore()
            store.add(docs, doc_embeddings)
            results = store.search(query_emb, k=k_results)

            context = "\n\n".join(
                f"[Chunk {i+1} | Score: {r.score:.3f}]\n{r.text}"
                for i, r in enumerate(results)
            )

            st.subheader("📋 Retrieved Context")
            for i, r in enumerate(results):
                st.markdown(f"**Chunk {i+1}** (score: {r.score:.3f}): {r.text}")

        except Exception as e:
            st.error(f"Retrieval error: {e}")
            st.stop()

    full_prompt = template_text.format(context=context, question=question)

    with st.expander("📝 Full Prompt Sent to LLM", expanded=True):
        st.code(full_prompt, language=None)

    with st.spinner("Step 2: Generating answer..."):
        try:
            llm = get_llm_provider()
            st.subheader("🤖 Generated Answer")
            response_container = st.empty()
            full_response = ""
            try:
                for chunk in llm.generate_stream(full_prompt, temperature=temperature, max_tokens=max_tokens):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
            except Exception:
                response = llm.generate(full_prompt, temperature=temperature, max_tokens=max_tokens)
                st.markdown(response.text)
                full_response = response.text

            if full_response:
                st.success(f"Generated {len(full_response.split())} words using **{llm.name()}**")
        except Exception as e:
            st.error(f"Generation error: {e}")
            st.info("Make sure you've configured an LLM provider in the sidebar. "
                    "You need either an API key or a running Ollama instance.")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/4_🔍_Retrieval.py", label="← Retrieval", icon="🔍")
with col2:
    st.page_link("pages/6_🔬_Full_Pipeline.py", label="Next: Full Pipeline →", icon="🔬")
