"""
Module 7: RAG Playground
Visual RAG pipeline builder with deployment export.
Build, configure, run, and deploy your own RAG pipeline.
"""

import streamlit as st
import os
import json
import textwrap

st.set_page_config(page_title="Playground | RAG Lab", page_icon="🎮", layout="wide")

from components.sidebar import render_provider_config
from core.document_loader import SUPPORTED_EXTENSIONS
from core.rag_pipeline import REASONING_MODES

render_provider_config()

st.title("🎮 RAG Playground")
st.markdown("*Build your own RAG pipeline visually — configure, run, and deploy.*")
st.markdown("---")

# ── Pipeline State ────────────────────────────────────────────────────────
if "pg_pipeline_steps" not in st.session_state:
    st.session_state.pg_pipeline_steps = {
        "document_source": None,
        "chunking": None,
        "embeddings": None,
        "vector_store": None,
        "retrieval": None,
        "reasoning": None,
        "llm": None,
    }
if "pg_pipeline_built" not in st.session_state:
    st.session_state.pg_pipeline_built = False

# ── Component Definitions ─────────────────────────────────────────────────
COMPONENTS = {
    "document_source": {
        "icon": "📄",
        "title": "Document Source",
        "color": "#4ECDC4",
        "options": {
            "upload": {"label": "File Upload", "desc": "Upload PDF, DOCX, VSDX, or TXT files"},
            "paste": {"label": "Paste Text", "desc": "Type or paste text directly"},
            "sample": {"label": "Sample Data", "desc": "Use included sample documents"},
            "url": {"label": "URL (text)", "desc": "Fetch text from a URL"},
        },
    },
    "chunking": {
        "icon": "✂️",
        "title": "Chunking Strategy",
        "color": "#45B7D1",
        "options": {
            "recursive": {"label": "Recursive", "desc": "Smart split on natural boundaries"},
            "character": {"label": "Fixed Characters", "desc": "Split at exact character count"},
            "sentence": {"label": "Sentence-based", "desc": "Groups of N sentences"},
            "paragraph": {"label": "Paragraph-based", "desc": "Split on double-newline"},
            "token": {"label": "Token-based", "desc": "Split by GPT token count"},
            "markdown": {"label": "Markdown Headers", "desc": "Header-aware structured split"},
            "sliding_window": {"label": "Sliding Window", "desc": "Overlapping windows with stride"},
            "semantic": {"label": "Semantic", "desc": "Groups by topic similarity (TF-IDF)"},
        },
    },
    "embeddings": {
        "icon": "🔢",
        "title": "Embedding Model",
        "color": "#96CEB4",
        "options": {
            "tfidf": {"label": "TF-IDF (Free)", "desc": "Local, no API key needed"},
            "openai": {"label": "OpenAI", "desc": "Best quality, needs API key"},
            "openrouter": {"label": "OpenRouter", "desc": "Many embedding models, one API key"},
            "ollama": {"label": "Ollama", "desc": "Local, needs Ollama running"},
        },
    },
    "vector_store": {
        "icon": "📦",
        "title": "Vector Store",
        "color": "#FFEAA7",
        "options": {
            "numpy": {"label": "NumPy (In-Memory)", "desc": "Simple, educational, fast"},
            "chroma": {"label": "ChromaDB", "desc": "Production-grade, persistent"},
        },
    },
    "retrieval": {
        "icon": "🔍",
        "title": "Retrieval Strategy",
        "color": "#DDA0DD",
        "options": {
            "similarity": {"label": "Cosine Similarity", "desc": "Standard nearest-neighbour search"},
            "mmr": {"label": "MMR (Diverse)", "desc": "Relevance + diversity balance"},
            "hybrid": {"label": "Hybrid (BM25+Semantic)", "desc": "Keyword + semantic fused via RRF"},
            "multi_query": {"label": "Multi-Query", "desc": "LLM rewrites query 3 ways, merges results"},
            "hyde": {"label": "HyDE", "desc": "Search with a hypothetical answer embedding"},
        },
    },
    "reasoning": {
        "icon": "🧠",
        "title": "Reasoning Mode",
        "color": "#FF8C94",
        "options": {
            "standard": {"label": "Standard", "desc": "Direct answer from context"},
            "cot": {"label": "Chain-of-Thought", "desc": "Step-by-step reasoning before answering"},
            "analysis": {"label": "Analysis", "desc": "Analyse each chunk, then synthesise"},
            "cot_analysis": {"label": "CoT + Analysis", "desc": "Full analysis + step-by-step"},
            "step_back": {"label": "Step-Back", "desc": "Answer a broader question first"},
            "self_reflect": {"label": "Self-Reflect", "desc": "Generate, critique, then refine"},
        },
    },
    "llm": {
        "icon": "🤖",
        "title": "Language Model",
        "color": "#98D8C8",
        "options": {
            "openai": {"label": "OpenAI GPT", "desc": "GPT-4o-mini / GPT-4o"},
            "openrouter": {"label": "OpenRouter", "desc": "Many models, one API"},
            "ollama": {"label": "Ollama", "desc": "100% local inference"},
        },
    },
}


def render_flow_diagram():
    """Render the current pipeline as a visual flow."""
    steps = st.session_state.pg_pipeline_steps
    html_parts = ['<div style="display:flex;align-items:center;justify-content:center;gap:8px;padding:20px 0;flex-wrap:wrap;">']

    for key, comp in COMPONENTS.items():
        selected = steps.get(key)
        is_set = selected is not None
        border_color = comp["color"] if is_set else "#444"
        bg = f"{comp['color']}15" if is_set else "#1a1d29"
        shadow = f"0 0 12px {comp['color']}44" if is_set else "none"
        label = comp["options"][selected]["label"] if is_set and selected in comp["options"] else "Not set"
        opacity = "1" if is_set else "0.5"

        html_parts.append(f'''
            <div style="text-align:center;padding:12px 14px;border-radius:14px;border:2px solid {border_color};
                        background:{bg};min-width:95px;box-shadow:{shadow};opacity:{opacity};">
                <div style="font-size:24px;">{comp["icon"]}</div>
                <div style="font-size:10px;color:{comp["color"]};font-weight:600;margin-top:3px;">{comp["title"]}</div>
                <div style="font-size:9px;color:#aaa;margin-top:2px;">{"✅ " + label if is_set else "⬜ " + label}</div>
            </div>
        ''')
        if key != "llm":
            arrow_color = comp["color"] if is_set else "#444"
            html_parts.append(f'<div style="font-size:18px;color:{arrow_color};">→</div>')

    html_parts.append('</div>')
    st.html("".join(html_parts))


# ── Pipeline Flow Visualization ───────────────────────────────────────────
st.subheader("Your Pipeline")
render_flow_diagram()

# ── Component Selection ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧱 Build Your Pipeline")
st.markdown("Select a component for each step.")

cols = st.columns(4)
step_keys = list(COMPONENTS.keys())

for i, key in enumerate(step_keys):
    comp = COMPONENTS[key]
    with cols[i % 4]:
        st.markdown(f"#### {comp['icon']} {comp['title']}")

        current = st.session_state.pg_pipeline_steps.get(key)
        option_keys = list(comp["options"].keys())
        option_labels = [comp["options"][k]["label"] for k in option_keys]

        current_idx = option_keys.index(current) if current in option_keys else 0
        selected_label = st.radio(
            f"Choose {comp['title']}:",
            option_labels,
            index=current_idx,
            key=f"pg_select_{key}",
        )

        selected_key = option_keys[option_labels.index(selected_label)]
        st.session_state.pg_pipeline_steps[key] = selected_key

        desc = comp["options"][selected_key]["desc"]
        st.caption(desc)
        st.markdown("")

# ── Configuration Panel ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚙️ Fine-tune Configuration")

pg_config = {}
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Chunking**")
    pg_config["chunk_size"] = st.slider("Chunk size:", 100, 2000, 500, key="pg_chunk_size")
    pg_config["chunk_overlap"] = st.slider("Overlap:", 0, 200, 50, key="pg_chunk_overlap")

with col2:
    st.markdown("**Retrieval**")
    pg_config["retrieval_k"] = st.slider("Top K results:", 1, 10, 3, key="pg_k")
    pg_config["use_reranking"] = st.checkbox("Enable LLM Reranking", key="pg_rerank")
    if pg_config["use_reranking"]:
        pg_config["rerank_top_n"] = st.slider("Keep top N after rerank:", 1, 10, 3, key="pg_rerank_n")

with col3:
    st.markdown("**Generation**")
    pg_config["temperature"] = st.slider("Temperature:", 0.0, 2.0, 0.7, 0.1, key="pg_temp")
    pg_config["max_tokens"] = st.slider("Max tokens:", 64, 2048, 512, key="pg_max_tokens")

with col4:
    st.markdown("**Guardrails** *(optional)*")
    pg_config["guardrail_mode"] = st.selectbox("Mode:", ["off", "warn", "block"], key="pg_guard_mode")

if pg_config["guardrail_mode"] != "off":
    from core.guardrails import AVAILABLE_INPUT_GUARDRAILS, AVAILABLE_OUTPUT_GUARDRAILS

    with st.expander("🛡️ Configure Guardrails", expanded=True):
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("**Input Guardrails**")
            pg_config["input_guardrails"] = []
            for key, (label, desc, needs_llm) in AVAILABLE_INPUT_GUARDRAILS.items():
                tag = " *(LLM)*" if needs_llm else ""
                if st.checkbox(f"{label}{tag}", key=f"pg_ig_{key}", help=desc):
                    pg_config["input_guardrails"].append(key)
        with gcol2:
            st.markdown("**Output Guardrails**")
            pg_config["output_guardrails"] = []
            for key, (label, desc, needs_llm) in AVAILABLE_OUTPUT_GUARDRAILS.items():
                tag = " *(LLM)*" if needs_llm else ""
                if st.checkbox(f"{label}{tag}", key=f"pg_og_{key}", help=desc):
                    pg_config["output_guardrails"].append(key)
        st.caption("Items marked *(LLM)* use an extra LLM call for evaluation.")

# ── Context Management ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💬 Conversation Context")

from core.context import CONTEXT_STRATEGIES

ctx_col1, ctx_col2 = st.columns([2, 1])
with ctx_col1:
    pg_config["context_strategy"] = st.selectbox(
        "Context Strategy:",
        list(CONTEXT_STRATEGIES.keys()),
        format_func=lambda k: f"{k} — {CONTEXT_STRATEGIES[k]}",
        key="pg_ctx_strategy",
    )
with ctx_col2:
    strat = pg_config["context_strategy"]
    if strat == "sliding_window":
        pg_config["ctx_window_size"] = st.slider("Window (turns):", 1, 20, 5, key="pg_ctx_window")
    elif strat == "summary_buffer":
        pg_config["ctx_buffer_size"] = st.slider("Buffer (recent turns):", 1, 10, 3, key="pg_ctx_buffer")
    elif strat == "token_budget":
        pg_config["ctx_max_tokens"] = st.slider("Token budget:", 256, 8000, 2000, 256, key="pg_ctx_tokens")
    elif strat == "relevant_history":
        pg_config["ctx_top_k"] = st.slider("Relevant turns to retrieve:", 1, 10, 3, key="pg_ctx_topk")

# ── Document Input ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📄 Provide Documents")

doc_source = st.session_state.pg_pipeline_steps.get("document_source", "paste")
doc_text = ""

if doc_source == "upload":
    uploaded = st.file_uploader(
        "Upload document:",
        type=SUPPORTED_EXTENSIONS,
        key="pg_file",
        help="Supported: PDF, DOCX, VSDX, TXT",
    )
    if uploaded:
        try:
            from core.document_loader import load_text
            raw_bytes = uploaded.read()
            doc_text = load_text(raw_bytes, uploaded.name)
            st.text_area("Extracted text preview:", doc_text[:1000] + ("..." if len(doc_text) > 1000 else ""),
                         height=200, disabled=True, key="pg_file_preview")
            st.caption(f"Extracted **{len(doc_text):,}** characters from `{uploaded.name}`")
        except Exception as e:
            st.error(f"Failed to parse file: {e}")

elif doc_source == "paste":
    doc_text = st.text_area("Enter text:", height=200, key="pg_text",
                             value="Paste your document text here for the RAG pipeline to process.")

elif doc_source == "sample":
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples")
    samples = []
    if os.path.isdir(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.endswith(".txt")]
    if samples:
        selected_file = st.selectbox("Choose sample:", samples, key="pg_sample_file")
        with open(os.path.join(sample_dir, selected_file), "r", encoding="utf-8") as f:
            doc_text = f.read()
        st.text_area("Preview:", doc_text[:500] + "...", disabled=True, key="pg_sample_preview")
    else:
        st.info("No samples found in data/samples/")

elif doc_source == "url":
    url = st.text_input("Enter URL:", key="pg_url")
    if url:
        try:
            import requests
            resp = requests.get(url, timeout=10)
            doc_text = resp.text[:10000]
            st.text_area("Fetched:", doc_text[:500] + "...", disabled=True, key="pg_url_preview")
        except Exception as e:
            st.error(f"Failed to fetch URL: {e}")

# ── Build & Run ───────────────────────────────────────────────────────────
st.markdown("---")

all_set = all(v is not None for v in st.session_state.pg_pipeline_steps.values())
has_text = bool(doc_text and doc_text.strip() and doc_text != "Paste your document text here for the RAG pipeline to process.")

col1, col2 = st.columns([2, 1])
with col1:
    if not all_set:
        st.warning("Configure all pipeline steps above to proceed.")
    elif not has_text:
        st.warning("Provide document text above.")
    else:
        st.success("Pipeline configured! Click below to build and run.")
with col2:
    pipeline_ready = all_set and has_text

if pipeline_ready:
    if st.button("🚀 Build & Ingest Pipeline", type="primary", use_container_width=True):
        steps = st.session_state.pg_pipeline_steps

        with st.spinner("Building pipeline..."):
            try:
                from core.embeddings import create_embeddings
                from core.vector_store import create_vector_store
                from core.llm_providers import create_llm
                from core.rag_pipeline import RAGPipeline

                cfg = st.session_state.get("provider_config", {})

                embed_kwargs = {}
                if steps["embeddings"] == "openai":
                    embed_kwargs = {"api_key": cfg.get("llm_api_key", ""), "model": cfg.get("embed_model", "text-embedding-3-small")}
                elif steps["embeddings"] == "openrouter":
                    embed_kwargs = {
                        "api_key": st.session_state.get("openrouter_key", "") or cfg.get("llm_api_key", ""),
                        "model": cfg.get("embed_model", "openai/text-embedding-3-small"),
                    }
                elif steps["embeddings"] == "ollama":
                    embed_kwargs = {"model": cfg.get("embed_model", "nomic-embed-text"), "base_url": st.session_state.get("ollama_url", "http://localhost:11434")}
                embed_provider = create_embeddings(steps["embeddings"], **embed_kwargs)

                vs = create_vector_store(steps["vector_store"])

                llm_kwargs = {"model": cfg.get("llm_model", "gpt-4o-mini")}
                if steps["llm"] == "openai":
                    llm_kwargs["api_key"] = cfg.get("llm_api_key", "")
                elif steps["llm"] == "openrouter":
                    llm_kwargs["api_key"] = cfg.get("llm_api_key", "")
                elif steps["llm"] == "ollama":
                    llm_kwargs["base_url"] = st.session_state.get("ollama_url", "http://localhost:11434")
                llm = create_llm(steps["llm"], **llm_kwargs)

                reasoning = steps.get("reasoning", "standard")

                # Build guardrails if enabled
                ig_list, og_list = [], []
                if pg_config.get("guardrail_mode", "off") != "off":
                    from core.guardrails import create_input_guardrails, create_output_guardrails
                    ig_list = create_input_guardrails(pg_config.get("input_guardrails", []), llm_provider=llm)
                    og_list = create_output_guardrails(pg_config.get("output_guardrails", []), llm_provider=llm)

                size_val = pg_config["chunk_size"]
                overlap_val = pg_config["chunk_overlap"]
                chunk_kw_map = {
                    "character":      {"chunk_size": size_val, "overlap": overlap_val},
                    "recursive":      {"chunk_size": size_val, "overlap": overlap_val},
                    "sentence":       {"max_sentences": max(1, size_val // 100), "overlap_sentences": max(0, overlap_val // 100)},
                    "paragraph":      {"max_paragraphs": max(1, size_val // 300)},
                    "token":          {"max_tokens": max(1, size_val // 4), "overlap_tokens": max(0, overlap_val // 4)},
                    "markdown":       {"max_chunk_size": size_val},
                    "sliding_window": {"window_size": size_val, "step_size": max(1, size_val - overlap_val)},
                    "semantic":       {"max_sentences": max(1, size_val // 100), "similarity_threshold": 0.5},
                }
                ck = chunk_kw_map.get(steps["chunking"], {"chunk_size": size_val})

                # Build context manager
                from core.context import create_context_manager
                ctx_strat = pg_config.get("context_strategy", "none")
                ctx_kwargs = {}
                if ctx_strat == "sliding_window":
                    ctx_kwargs["window_size"] = pg_config.get("ctx_window_size", 5)
                elif ctx_strat == "summary_buffer":
                    ctx_kwargs["buffer_size"] = pg_config.get("ctx_buffer_size", 3)
                elif ctx_strat == "token_budget":
                    ctx_kwargs["max_tokens"] = pg_config.get("ctx_max_tokens", 2000)
                elif ctx_strat == "relevant_history":
                    ctx_kwargs["top_k"] = pg_config.get("ctx_top_k", 3)
                ctx_mgr = create_context_manager(
                    ctx_strat, llm_provider=llm, embedding_provider=embed_provider, **ctx_kwargs,
                )

                pipeline = RAGPipeline(
                    embedding_provider=embed_provider,
                    vector_store=vs,
                    llm_provider=llm,
                    chunk_strategy=steps["chunking"],
                    chunk_kwargs=ck,
                    retrieval_k=pg_config["retrieval_k"],
                    retrieval_strategy=steps["retrieval"],
                    reasoning_mode=reasoning,
                    use_reranking=pg_config.get("use_reranking", False),
                    rerank_top_n=pg_config.get("rerank_top_n"),
                    input_guardrails=ig_list,
                    output_guardrails=og_list,
                    guardrail_mode=pg_config.get("guardrail_mode", "warn"),
                    context_manager=ctx_mgr,
                )

                chunks = pipeline.ingest(doc_text)
                ingest_steps = pipeline.last_steps

                st.session_state["pg_pipeline"] = pipeline
                st.session_state["pg_built"] = True
                st.session_state["pg_doc_text"] = doc_text

                st.success(f"Pipeline built! Ingested **{len(chunks)} chunks** "
                           f"| Reasoning: **{reasoning}** | Reranking: **{'ON' if pg_config.get('use_reranking') else 'OFF'}**")

                render_flow_diagram()

                st.subheader("⏱️ Ingestion Timing")
                from components.viz import render_step_metrics
                render_step_metrics(ingest_steps)

                with st.expander(f"📋 View {len(chunks)} Chunks"):
                    for c in chunks:
                        st.markdown(f"**Chunk {c.index}** ({c.char_count} chars)")
                        st.text(c.text[:300])
                        st.divider()

            except Exception as e:
                st.error(f"Pipeline build error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ── Chat Interface ────────────────────────────────────────────────────────
if st.session_state.get("pg_built"):
    st.markdown("---")
    st.subheader("💬 Chat with Your Pipeline")

    pipeline = st.session_state["pg_pipeline"]
    ctx_mgr = pipeline.context_manager

    if "pg_chat_history" not in st.session_state:
        st.session_state["pg_chat_history"] = []

    from core.token_tracker import SessionTokenTracker
    if "pg_token_tracker" not in st.session_state:
        st.session_state["pg_token_tracker"] = SessionTokenTracker()
    token_tracker = st.session_state["pg_token_tracker"]

    # ── Header bar: stats + controls ──
    hdr1, hdr2, hdr3 = st.columns([3, 1, 1])
    with hdr1:
        info_parts = []
        if ctx_mgr is not None:
            info_parts.append(f"Context: **{ctx_mgr.strategy_name}** ({ctx_mgr.turn_count} turns)")
        if token_tracker.turn_count > 0:
            info_parts.append(f"**{token_tracker.total_tokens:,}** tokens / **{token_tracker.turn_count}** queries")
        if info_parts:
            st.caption(" | ".join(info_parts))
        else:
            st.caption("Ask a question below to start chatting.")
    with hdr2:
        if st.button("🗑️ Clear Chat", key="pg_clear_chat", use_container_width=True):
            st.session_state["pg_chat_history"] = []
            st.session_state["pg_token_tracker"] = SessionTokenTracker()
            if ctx_mgr is not None:
                ctx_mgr.clear()
            st.rerun()
    with hdr3:
        show_details = st.toggle("Show details", value=False, key="pg_show_details")

    # ── Scrollable chat window ──
    chat_container = st.container(height=480)

    with chat_container:
        if not st.session_state["pg_chat_history"]:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:100%;opacity:0.4;padding:60px 0;text-align:center;">'
                '<div><div style="font-size:48px;">💬</div>'
                '<p>Your conversation will appear here.<br>'
                'Type a question below to get started.</p></div></div>',
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state["pg_chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("details"):
                        tok = msg["details"].get("token_summary", "")
                        if tok:
                            st.caption(tok)
                        if show_details:
                            if msg["details"].get("guardrails"):
                                for gr_item in msg["details"]["guardrails"]:
                                    icon = "✅" if gr_item["passed"] else "🚫"
                                    st.caption(f"{icon} {gr_item['name']}: {gr_item['reason']}")
                            if msg["details"].get("chunks"):
                                with st.expander("Retrieved chunks"):
                                    for ci, ch in enumerate(msg["details"]["chunks"]):
                                        st.markdown(f"**#{ci+1}** `{ch['score']}`  \n> {ch['text']}")
                            if msg["details"].get("timing"):
                                st.caption(msg["details"]["timing"])
                            if msg["details"].get("token_steps"):
                                with st.expander("Token breakdown"):
                                    for ts in msg["details"]["token_steps"]:
                                        st.markdown(f"- {ts}")

    # ── Chat input (pinned below the container) ──
    if question := st.chat_input("Ask about your documents..."):
        st.session_state["pg_chat_history"].append({"role": "user", "content": question})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = pipeline.query(question)

                        details: dict = {}

                        gr_items = []
                        for report, label in [(result.input_guardrails, "Input"), (result.output_guardrails, "Output")]:
                            if report:
                                for gr in report.results:
                                    gr_items.append({"name": f"{label}: {gr.name}",
                                                     "passed": gr.passed, "reason": gr.reason})
                        if gr_items:
                            details["guardrails"] = gr_items

                        st.markdown(result.answer)

                        tu = result.token_usage
                        if tu and tu.total_tokens > 0:
                            token_tracker.add_turn(tu)
                            tok_summary = (
                                f"Tokens — prompt: **{tu.prompt_tokens:,}** | "
                                f"completion: **{tu.completion_tokens:,}** | "
                                f"embedding: **{tu.embedding_tokens:,}** | "
                                f"total: **{tu.total_tokens:,}**"
                            )
                            st.caption(tok_summary)
                            details["token_summary"] = tok_summary
                            details["token_steps"] = [
                                f"**{s.step_name}**: {s.total_tokens:,} tok "
                                f"(prompt={s.prompt_tokens:,}, completion={s.completion_tokens:,}, embed={s.embedding_tokens:,})"
                                for s in tu.steps
                            ]

                        chunk_details = []
                        for i, chunk in enumerate(result.retrieved_chunks):
                            chunk_details.append({"score": f"{chunk.score:.3f}",
                                                  "text": chunk.text[:200] + "..."})
                        details["chunks"] = chunk_details
                        details["timing"] = (
                            f"Total: {result.total_duration_ms:.0f}ms | "
                            f"Steps: {' → '.join(s.name for s in result.steps)}"
                        )

                        st.session_state["pg_chat_history"].append({
                            "role": "assistant", "content": result.answer, "details": details,
                        })

                    except Exception as e:
                        err_msg = f"Error: {e}"
                        st.error(err_msg)
                        st.session_state["pg_chat_history"].append({"role": "assistant", "content": err_msg})

    # ── Collapsible panels below the chat ──
    panel1, panel2 = st.columns(2)
    with panel1:
        if token_tracker.turn_count > 0:
            with st.expander("📊 Session Token Usage & Cost"):
                cfg = st.session_state.get("provider_config", {})
                model_name = cfg.get("llm_model", "gpt-4o-mini")
                from components.viz import render_session_token_summary
                render_session_token_summary(token_tracker, model_name=model_name)
    with panel2:
        if ctx_mgr is not None and ctx_mgr.turn_count > 0:
            with st.expander("📜 Raw Context Window"):
                from core.context import RelevantHistoryContext
                if isinstance(ctx_mgr, RelevantHistoryContext):
                    raw = ctx_mgr.get_context_string(query="(latest)")
                else:
                    raw = ctx_mgr.get_context_string()
                if raw:
                    st.code(raw, language="text")
                else:
                    st.caption("Context is empty.")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/6_🔬_Full_Pipeline.py", label="← Full Pipeline", icon="🔬")
with col2:
    st.page_link("pages/8_🚀_Advanced_Topics.py", label="Next: Advanced Topics →", icon="🚀")
