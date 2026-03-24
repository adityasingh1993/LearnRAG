"""
Module 6: Full Pipeline
End-to-end RAG walkthrough showing every step with timing and details.
"""

import streamlit as st
import os
import time

st.set_page_config(page_title="Full Pipeline | RAG Lab", page_icon="🔬", layout="wide")

from components.sidebar import render_provider_config, get_embedding_provider, get_llm_provider, get_vector_store
from components.viz import render_pipeline_flow, render_step_metrics, plot_embeddings_2d, plot_retrieval_scores
from core.rag_pipeline import RAGPipeline

render_provider_config()

st.title("🔬 Full RAG Pipeline")
st.markdown("*Watch every component work together, step by step.*")
st.markdown("---")

render_pipeline_flow([], active_step=None)

# ── Step 1: Document Input ────────────────────────────────────────────────
st.header("Step 1: Document Input")

sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples")

from core.document_loader import SUPPORTED_EXTENSIONS, load_text as _load_file

input_method = st.radio("Choose input:", ["Sample Document", "Upload File", "Paste Text"], horizontal=True, key="fp_input")

doc_text = ""
if input_method == "Sample Document":
    sample_files = []
    if os.path.isdir(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".txt")]
    if sample_files:
        selected = st.selectbox("Select sample:", sample_files, key="fp_sample")
        with open(os.path.join(sample_dir, selected), "r", encoding="utf-8") as f:
            doc_text = f.read()
        st.text_area("Document preview:", doc_text[:800] + "...", height=200, disabled=True, key="fp_preview")
elif input_method == "Upload File":
    uploaded = st.file_uploader(
        "Upload document (PDF, DOCX, VSDX, TXT):",
        type=SUPPORTED_EXTENSIONS,
        key="fp_upload",
    )
    if uploaded:
        try:
            doc_text = _load_file(uploaded.read(), uploaded.name)
            st.text_area("Document preview:", doc_text[:800] + "...", height=200, disabled=True, key="fp_upload_preview")
            st.caption(f"Extracted **{len(doc_text):,}** chars from `{uploaded.name}`")
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
else:
    doc_text = st.text_area("Paste your text:", height=200, key="fp_paste",
                             value="Enter your document text here. The pipeline will chunk, embed, store, and query it.")

st.metric("Document Length", f"{len(doc_text)} characters")

# ── Pipeline Configuration ────────────────────────────────────────────────
st.markdown("---")
st.header("Step 2: Configure Pipeline")

from core.rag_pipeline import REASONING_MODES
from core.chunking import STRATEGY_INFO as CHUNK_INFO
from core.retrieval import RETRIEVAL_STRATEGIES

col1, col2, col3, col4 = st.columns(4)
with col1:
    chunk_strategy = st.selectbox(
        "Chunking Strategy:", list(CHUNK_INFO.keys()),
        format_func=lambda k: f"{k} — {CHUNK_INFO[k]}", key="fp_chunk",
    )
    chunk_size = st.slider("Chunk Size:", 100, 2000, 500, key="fp_chunk_size")
with col2:
    retrieval_strategy = st.selectbox(
        "Retrieval Strategy:", list(RETRIEVAL_STRATEGIES.keys()),
        format_func=lambda k: f"{k} — {RETRIEVAL_STRATEGIES[k]}", key="fp_retrieval",
    )
    retrieval_k = st.slider("Retrieval K:", 1, 10, 3, key="fp_k")
    use_reranking = st.checkbox("LLM Reranking", key="fp_rerank")
with col3:
    reasoning_mode = st.selectbox(
        "Reasoning Mode:", list(REASONING_MODES.keys()),
        format_func=lambda k: f"{k} — {REASONING_MODES[k]}", key="fp_reasoning",
    )
with col4:
    temperature = st.slider("LLM Temperature:", 0.0, 2.0, 0.7, 0.1, key="fp_temp")
    guardrail_mode = st.selectbox("Guardrails:", ["off", "warn", "block"], key="fp_guard_mode")

# Guardrails config
fp_input_guards, fp_output_guards = [], []
if guardrail_mode != "off":
    from core.guardrails import AVAILABLE_INPUT_GUARDRAILS, AVAILABLE_OUTPUT_GUARDRAILS
    with st.expander("🛡️ Configure Guardrails"):
        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("**Input**")
            for key, (label, desc, needs_llm) in AVAILABLE_INPUT_GUARDRAILS.items():
                if st.checkbox(f"{label}{' *(LLM)*' if needs_llm else ''}", key=f"fp_ig_{key}", help=desc):
                    fp_input_guards.append(key)
        with gc2:
            st.markdown("**Output**")
            for key, (label, desc, needs_llm) in AVAILABLE_OUTPUT_GUARDRAILS.items():
                if st.checkbox(f"{label}{' *(LLM)*' if needs_llm else ''}", key=f"fp_og_{key}", help=desc):
                    fp_output_guards.append(key)

# Context management
from core.context import CONTEXT_STRATEGIES
ctx_col1, ctx_col2 = st.columns([2, 1])
with ctx_col1:
    fp_ctx_strategy = st.selectbox(
        "Context Strategy:",
        list(CONTEXT_STRATEGIES.keys()),
        format_func=lambda k: f"{k} — {CONTEXT_STRATEGIES[k]}",
        key="fp_ctx_strategy",
    )
with ctx_col2:
    fp_ctx_kwargs: dict = {}
    if fp_ctx_strategy == "sliding_window":
        fp_ctx_kwargs["window_size"] = st.slider("Window (turns):", 1, 20, 5, key="fp_ctx_window")
    elif fp_ctx_strategy == "summary_buffer":
        fp_ctx_kwargs["buffer_size"] = st.slider("Buffer (recent turns):", 1, 10, 3, key="fp_ctx_buffer")
    elif fp_ctx_strategy == "token_budget":
        fp_ctx_kwargs["max_tokens"] = st.slider("Token budget:", 256, 8000, 2000, 256, key="fp_ctx_tokens")
    elif fp_ctx_strategy == "relevant_history":
        fp_ctx_kwargs["top_k"] = st.slider("Relevant turns:", 1, 10, 3, key="fp_ctx_topk")

# ── Run Pipeline ──────────────────────────────────────────────────────────
st.markdown("---")
st.header("Step 3: Ingest & Query")

if st.button("📥 Ingest Document", type="primary") and doc_text.strip():
    with st.spinner("Running ingestion pipeline..."):
        try:
            embed_provider = get_embedding_provider()
            vs = get_vector_store()
            llm = get_llm_provider()

            ig_list, og_list = [], []
            if guardrail_mode != "off":
                from core.guardrails import create_input_guardrails, create_output_guardrails
                ig_list = create_input_guardrails(fp_input_guards, llm_provider=llm)
                og_list = create_output_guardrails(fp_output_guards, llm_provider=llm)

            chunk_kw_map = {
                "character":      {"chunk_size": chunk_size, "overlap": 50},
                "recursive":      {"chunk_size": chunk_size, "overlap": 50},
                "sentence":       {"max_sentences": max(1, chunk_size // 100), "overlap_sentences": 1},
                "paragraph":      {"max_paragraphs": max(1, chunk_size // 300)},
                "token":          {"max_tokens": max(1, chunk_size // 4), "overlap_tokens": 8},
                "markdown":       {"max_chunk_size": chunk_size},
                "sliding_window": {"window_size": chunk_size, "step_size": max(1, chunk_size // 2)},
                "semantic":       {"max_sentences": max(1, chunk_size // 100), "similarity_threshold": 0.5},
            }
            ck = chunk_kw_map.get(chunk_strategy, {"chunk_size": chunk_size})

            from core.context import create_context_manager
            ctx_mgr = create_context_manager(
                fp_ctx_strategy, llm_provider=llm,
                embedding_provider=embed_provider, **fp_ctx_kwargs,
            )

            pipeline = RAGPipeline(
                embedding_provider=embed_provider,
                vector_store=vs,
                llm_provider=llm,
                chunk_strategy=chunk_strategy,
                chunk_kwargs=ck,
                retrieval_k=retrieval_k,
                retrieval_strategy=retrieval_strategy,
                reasoning_mode=reasoning_mode,
                use_reranking=use_reranking,
                context_manager=ctx_mgr,
                input_guardrails=ig_list,
                output_guardrails=og_list,
                guardrail_mode=guardrail_mode,
            )

            chunks = pipeline.ingest(doc_text)
            steps = pipeline.last_steps

            st.session_state["fp_pipeline"] = pipeline
            st.session_state["fp_ingest_steps"] = steps
            st.session_state["fp_chunks"] = chunks

            st.success(f"Ingested **{len(chunks)} chunks** in {sum(s.duration_ms for s in steps):.0f}ms")

            render_pipeline_flow(steps, active_step=3)

            st.subheader("Ingestion Steps")
            render_step_metrics(steps)

            with st.expander("📋 View All Chunks"):
                for c in chunks:
                    st.markdown(f"**Chunk {c.index}** ({c.char_count} chars)")
                    st.text(c.text[:200] + ("..." if c.char_count > 200 else ""))
                    st.markdown("---")

        except Exception as e:
            st.error(f"Ingestion error: {e}")

if "fp_pipeline" in st.session_state:
    st.markdown("---")
    st.subheader("🔍 Query the Pipeline")

    question = st.text_input("Ask a question about the document:", key="fp_question",
                              placeholder="What are the main topics covered?")

    if st.button("🤖 Run RAG Query", type="secondary") and question:
        pipeline = st.session_state["fp_pipeline"]
        pipeline.retrieval_k = retrieval_k

        with st.spinner("Querying pipeline..."):
            try:
                result = pipeline.query(question)

                render_pipeline_flow(result.steps, active_step=5)

                st.subheader("Pipeline Execution")
                render_step_metrics(result.steps)

                col_ret, col_gen = st.columns([1, 2])

                with col_ret:
                    st.subheader("📋 Retrieved Chunks")
                    for i, chunk in enumerate(result.retrieved_chunks):
                        st.markdown(f"**Chunk {i+1}** — Score: `{chunk.score:.3f}`")
                        st.markdown(f"> {chunk.text[:200]}{'...' if len(chunk.text) > 200 else ''}")
                        st.markdown("")
                    plot_retrieval_scores(result.retrieved_chunks, question)

                with col_gen:
                    st.subheader("🤖 Generated Answer")
                    st.markdown(result.answer)

                    st.markdown("---")
                    tu = result.token_usage
                    tok_info = ""
                    if tu and tu.total_tokens > 0:
                        tok_info = (f" | Tokens: **{tu.total_tokens:,}** "
                                    f"(prompt={tu.prompt_tokens:,}, "
                                    f"completion={tu.completion_tokens:,}, "
                                    f"embed={tu.embedding_tokens:,})")
                    st.caption(f"Total pipeline time: **{result.total_duration_ms:.0f}ms** | "
                               f"Model: **{pipeline.llm_provider.name()}**{tok_info}")

                # Token breakdown
                if tu and tu.total_tokens > 0:
                    with st.expander("📊 Token Usage Breakdown"):
                        from components.viz import render_token_usage
                        render_token_usage(tu, label="this query")
                        st.markdown("**Per step:**")
                        for s in tu.steps:
                            st.markdown(f"- **{s.step_name}**: {s.total_tokens:,} tokens "
                                        f"(prompt={s.prompt_tokens:,}, completion={s.completion_tokens:,}, embed={s.embedding_tokens:,})")

                for step in result.steps:
                    with st.expander(f"🔍 {step.name} — {step.duration_ms:.0f}ms"):
                        st.json(step.details)

            except Exception as e:
                st.error(f"Query error: {e}")
                st.info("Ensure your LLM provider is configured correctly in the sidebar.")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/1_📚_RAG/05_🤖_Generation.py", label="← Generation", icon="🤖")
with col2:
    st.page_link("pages/1_📚_RAG/07_📊_Evaluation.py", label="Next: Evaluation →", icon="📊")
