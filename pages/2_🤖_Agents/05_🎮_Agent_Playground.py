"""
Module 13: Agent Playground
Build, configure, and chat with your own AI agent — the culminating experience.
"""

import time
import streamlit as st

st.set_page_config(page_title="Agent Playground | RAG Lab", page_icon="🎮", layout="wide")

from components.sidebar import render_provider_config, get_llm_provider
from core.tools import (
    BUILTIN_TOOLS,
    Tool,
    ToolParameter,
    create_tool_registry,
    create_custom_tool,
)
from core.agent_loop import AgentExecutor, AgentStep, AgentResult, AGENT_PATTERNS

render_provider_config()

st.title("🎮 Agent Playground")
st.markdown("*Build your own agent — choose tools, pick a pattern, and start chatting.*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 1 — Visual Flow Diagram
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Agent Architecture")

selected_tools_for_diagram = st.session_state.get("ap_selected_tools", [])
selected_pattern_for_diagram = st.session_state.get("ap_pattern", "react")
is_built = st.session_state.get("ap_agent_built", False)

tool_color = "#4ECDC4" if selected_tools_for_diagram else "#444"
pattern_color = "#FF6B6B" if selected_pattern_for_diagram else "#444"
llm_color = "#96CEB4"
agent_color = "#FFEAA7" if is_built else "#444"

st.html(f"""
<div style="display:flex;align-items:center;justify-content:center;gap:10px;padding:20px 0;flex-wrap:wrap;">
  <div style="text-align:center;padding:14px 16px;border-radius:14px;border:2px solid {tool_color};
              background:{tool_color}15;min-width:100px;">
    <div style="font-size:26px;">🔧</div>
    <div style="font-size:11px;color:{tool_color};font-weight:600;margin-top:4px;">Tools</div>
    <div style="font-size:9px;color:#aaa;margin-top:2px;">{len(selected_tools_for_diagram)} selected</div>
  </div>
  <div style="font-size:20px;color:{tool_color};">→</div>
  <div style="text-align:center;padding:14px 16px;border-radius:14px;border:2px solid {pattern_color};
              background:{pattern_color}15;min-width:100px;">
    <div style="font-size:26px;">🔄</div>
    <div style="font-size:11px;color:{pattern_color};font-weight:600;margin-top:4px;">Pattern</div>
    <div style="font-size:9px;color:#aaa;margin-top:2px;">{selected_pattern_for_diagram}</div>
  </div>
  <div style="font-size:20px;color:{pattern_color};">→</div>
  <div style="text-align:center;padding:14px 16px;border-radius:14px;border:2px solid {llm_color};
              background:{llm_color}15;min-width:100px;">
    <div style="font-size:26px;">🤖</div>
    <div style="font-size:11px;color:{llm_color};font-weight:600;margin-top:4px;">LLM</div>
    <div style="font-size:9px;color:#aaa;margin-top:2px;">from sidebar</div>
  </div>
  <div style="font-size:20px;color:{llm_color};">→</div>
  <div style="text-align:center;padding:14px 16px;border-radius:14px;border:2px solid {agent_color};
              background:{agent_color}15;min-width:100px;">
    <div style="font-size:26px;">{"✅" if is_built else "⬜"}</div>
    <div style="font-size:11px;color:{agent_color};font-weight:600;margin-top:4px;">Agent</div>
    <div style="font-size:9px;color:#aaa;margin-top:2px;">{"Ready" if is_built else "Not built"}</div>
  </div>
</div>
""")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 — Configuration Panel
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("⚙️ Configure Your Agent")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("**🔧 Tool Selection**")
    tool_options = list(BUILTIN_TOOLS.keys())
    selected_tools = st.multiselect(
        "Choose tools for your agent:",
        options=tool_options,
        default=["calculator", "web_search"],
        key="ap_tool_select",
        help="Pick which tools your agent can use.",
    )
    st.session_state["ap_selected_tools"] = selected_tools

    if selected_tools:
        for tname in selected_tools:
            tool = BUILTIN_TOOLS[tname]
            st.caption(f"**{tname}** ({tool.category}) — {tool.description}")

with col2:
    st.markdown("**🔄 Agent Pattern**")
    pattern_keys = list(AGENT_PATTERNS.keys())
    pattern_labels = [AGENT_PATTERNS[k].split("—")[0].strip() for k in pattern_keys]
    selected_pattern_label = st.selectbox(
        "How the agent reasons:",
        options=pattern_labels,
        key="ap_pattern_select",
    )
    selected_pattern = pattern_keys[pattern_labels.index(selected_pattern_label)]
    st.session_state["ap_pattern"] = selected_pattern
    st.caption(AGENT_PATTERNS[selected_pattern])

with col3:
    st.markdown("**⚙️ Settings**")
    max_steps = st.slider("Max steps:", 1, 12, 6, key="ap_max_steps")
    temperature = st.slider("Temperature:", 0.0, 2.0, 0.1, 0.1, key="ap_temperature")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 3 — Optional RAG Integration
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("📚 Optional RAG Integration")

enable_rag = st.checkbox(
    "Give agent RAG capabilities",
    key="ap_enable_rag",
    help="Add a rag_search tool backed by a real RAG pipeline.",
)

rag_knowledge = ""
if enable_rag:
    rag_knowledge = st.text_area(
        "Knowledge base text:",
        height=180,
        key="ap_rag_text",
        placeholder="Paste the text your agent should be able to search through...",
    )
    if rag_knowledge:
        st.caption(f"{len(rag_knowledge):,} characters of knowledge loaded.")
    else:
        st.warning("Enter some knowledge base text for the RAG tool to search.")


def _build_rag_tool(knowledge_text: str) -> Tool:
    """Build a rag_search tool backed by a lightweight RAG pipeline."""
    from core.embeddings import create_embeddings
    from core.vector_store import NumpyVectorStore
    from core.rag_pipeline import RAGPipeline

    embed = create_embeddings("tfidf")
    vs = NumpyVectorStore()
    llm = get_llm_provider()

    pipeline = RAGPipeline(
        embedding_provider=embed,
        vector_store=vs,
        llm_provider=llm,
        chunk_strategy="recursive",
        chunk_kwargs={"chunk_size": 400, "overlap": 50},
        retrieval_k=3,
    )
    pipeline.ingest(knowledge_text)

    def _search(query: str) -> str:
        result = pipeline.query(query)
        return result.answer

    return Tool(
        name="rag_search",
        description="Search a knowledge base using RAG. Retrieves relevant context and generates an answer.",
        parameters=[ToolParameter("query", "string", "The question to search the knowledge base for")],
        function=_search,
        category="knowledge",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Build Agent
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")

can_build = bool(selected_tools or (enable_rag and rag_knowledge))

if st.button("🚀 Build Agent", type="primary", use_container_width=True, disabled=not can_build):
    with st.spinner("Building agent..."):
        try:
            llm = get_llm_provider()
            registry = create_tool_registry(selected_tools)

            if enable_rag and rag_knowledge:
                rag_tool = _build_rag_tool(rag_knowledge)
                registry.register(rag_tool)

            agent = AgentExecutor(
                llm_provider=llm,
                tool_registry=registry,
                pattern=selected_pattern,
                max_steps=max_steps,
            )

            st.session_state["agent_executor"] = agent
            st.session_state["ap_agent_built"] = True
            st.session_state["agent_chat_history"] = []

            tool_list = registry.tool_names()
            st.success(
                f"Agent built! Pattern: **{selected_pattern}** | "
                f"Tools: **{', '.join(tool_list)}** | "
                f"Max steps: **{max_steps}**"
            )
            st.rerun()

        except Exception as e:
            st.error(f"Failed to build agent: {e}")
            import traceback
            st.code(traceback.format_exc())

if not can_build:
    st.warning("Select at least one tool (or enable RAG with knowledge text) to build an agent.")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 4 — Chat Interface
# ═══════════════════════════════════════════════════════════════════════════

def _render_step(step: AgentStep):
    """Render a single agent reasoning step as a visual timeline row."""
    step_cols = st.columns([1, 1, 1])
    with step_cols[0]:
        st.markdown(
            f'<div style="border-left:3px solid #45B7D1;padding:8px 12px;margin:4px 0;'
            f'background:#45B7D110;border-radius:0 8px 8px 0;">'
            f'<b style="color:#45B7D1;">Step {step.step_number} — Thought</b><br>'
            f'<span style="font-size:13px;">{step.thought or "—"}</span></div>',
            unsafe_allow_html=True,
        )
    with step_cols[1]:
        action_color = "#4CAF50" if not step.is_final else "#FFEAA7"
        st.markdown(
            f'<div style="border-left:3px solid {action_color};padding:8px 12px;margin:4px 0;'
            f'background:{action_color}10;border-radius:0 8px 8px 0;">'
            f'<b style="color:{action_color};">Action</b><br>'
            f'<span style="font-size:13px;">{step.action or "—"}</span></div>',
            unsafe_allow_html=True,
        )
    with step_cols[2]:
        obs = step.observation
        if not obs and isinstance(step.action_input, dict):
            obs = step.action_input.get("answer", "—")
        obs = obs or "—"
        box_color = "#4CAF50" if step.is_final else "#FFC107"
        label = "✅ Final Answer" if step.is_final else "Observation"
        st.markdown(
            f'<div style="border-left:3px solid {box_color};padding:8px 12px;margin:4px 0;'
            f'background:{box_color}10;border-radius:0 8px 8px 0;">'
            f'<b style="color:{box_color};">{label}</b><br>'
            f'<span style="font-size:13px;">{str(obs)[:300]}</span></div>',
            unsafe_allow_html=True,
        )


if st.session_state.get("ap_agent_built") and "agent_executor" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Chat with Your Agent")

    agent: AgentExecutor = st.session_state["agent_executor"]

    if "agent_chat_history" not in st.session_state:
        st.session_state["agent_chat_history"] = []

    hdr1, hdr2 = st.columns([3, 1])
    with hdr1:
        info = (
            f"Pattern: **{agent.pattern}** | "
            f"Tools: **{', '.join(agent.tools.tool_names())}** | "
            f"Messages: **{len(st.session_state['agent_chat_history'])}**"
        )
        st.caption(info)
    with hdr2:
        if st.button("🗑️ Clear Chat", key="ap_clear_chat", use_container_width=True):
            st.session_state["agent_chat_history"] = []
            st.rerun()

    chat_container = st.container(height=500)

    with chat_container:
        if not st.session_state["agent_chat_history"]:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:100%;opacity:0.4;padding:60px 0;text-align:center;">'
                '<div><div style="font-size:48px;">🤖</div>'
                '<p>Your agent is ready!<br>'
                'Type a question below to start chatting.</p></div></div>',
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state["agent_chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

                    if msg.get("agent_result"):
                        res: AgentResult = msg["agent_result"]

                        st.caption(
                            f"Steps: **{len(res.steps)}** | "
                            f"Tools used: **{', '.join(res.tools_used) or 'none'}** | "
                            f"Duration: **{res.total_duration_ms:.0f} ms**"
                        )

                        with st.expander("🧠 Agent Reasoning"):
                            for step in res.steps:
                                _render_step(step)

    if question := st.chat_input("Ask your agent anything..."):
        st.session_state["agent_chat_history"].append({"role": "user", "content": question})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Agent is thinking..."):
                    try:
                        t0 = time.time()
                        result: AgentResult = agent.run(question)
                        duration = (time.time() - t0) * 1000

                        st.markdown(result.answer)

                        st.caption(
                            f"Steps: **{len(result.steps)}** | "
                            f"Tools used: **{', '.join(result.tools_used) or 'none'}** | "
                            f"Duration: **{result.total_duration_ms:.0f} ms**"
                        )

                        with st.expander("🧠 Agent Reasoning"):
                            for step in result.steps:
                                _render_step(step)

                        st.session_state["agent_chat_history"].append({
                            "role": "assistant",
                            "content": result.answer,
                            "agent_result": result,
                        })

                    except Exception as e:
                        err_msg = f"Error: {e}"
                        st.error(err_msg)
                        st.session_state["agent_chat_history"].append({
                            "role": "assistant",
                            "content": err_msg,
                        })

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/2_🤖_Agents/04_🌐_Multi_Agent.py", label="← Multi-Agent Systems", icon="🌐")
with col2:
    st.page_link("pages/2_🤖_Agents/06_❓_Agent_Help.py", label="Next: Agent Help →", icon="❓")
