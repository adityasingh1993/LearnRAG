"""
Module 12: Agent Patterns
Explore ReAct, Plan-and-Execute, Reflection, and Tool Choice patterns with live demos.
"""

import time

import streamlit as st

st.set_page_config(page_title="Agent Patterns | RAG Lab", page_icon="🔄", layout="wide")

from components.sidebar import render_provider_config, get_llm_provider
from core.agent_loop import AGENT_PATTERNS, AgentExecutor, AgentResult, AgentStep
from core.tools import BUILTIN_TOOLS, create_tool_registry

render_provider_config()

# ── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .concept-box {
        background: linear-gradient(145deg, #1a1d29, #22263a);
        border-left: 4px solid #6C63FF;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .concept-box h4 { color: #6C63FF; margin-top: 0; }
    .step-thought {
        background: #1a2744;
        border-left: 4px solid #45B7D1;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        color: #c8dff7;
    }
    .step-action {
        background: #1a3328;
        border-left: 4px solid #00CC96;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        color: #b8f0d8;
    }
    .step-observation {
        background: #3d3520;
        border-left: 4px solid #FFEAA7;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        color: #f5ecc8;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔄 Agent Patterns")
st.markdown("*Learn four distinct patterns for building AI agents, and try them live.*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════
st.header("Overview")
st.markdown("The agent framework supports four patterns, each suited for different situations:")

for key, description in AGENT_PATTERNS.items():
    pattern_name, pattern_desc = description.split(" — ", 1)
    st.markdown(f"- **{pattern_name}** (`{key}`) — {pattern_desc}")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 — Interactive Pattern Explorer
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Interactive Pattern Explorer")

selected_pattern = st.selectbox(
    "Choose a pattern to explore",
    options=list(AGENT_PATTERNS.keys()),
    format_func=lambda x: AGENT_PATTERNS[x].split(" — ")[0],
)

# ── Pattern Diagrams ─────────────────────────────────────────────────────

_PATTERN_DIAGRAMS = {
    "react": """
    <div style="display:flex;align-items:center;justify-content:center;gap:10px;padding:24px 0;flex-wrap:wrap;">
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#6C63FF22,#6C63FF11);
                    border:2px solid #6C63FF;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">❓</div>
            <div style="color:#6C63FF;font-weight:600;font-size:0.9rem;">Question</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#45B7D122,#45B7D111);
                    border:2px solid #45B7D1;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">🧠</div>
            <div style="color:#45B7D1;font-weight:600;font-size:0.9rem;">Thought</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#00CC9622,#00CC9611);
                    border:2px solid #00CC96;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">🔧</div>
            <div style="color:#00CC96;font-weight:600;font-size:0.9rem;">Action</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#FFEAA722,#FFEAA711);
                    border:2px solid #FFEAA7;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">👁️</div>
            <div style="color:#FFEAA7;font-weight:600;font-size:0.9rem;">Observe</div>
        </div>
        <div style="font-size:24px;color:#555;">↻</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#4ECDC422,#4ECDC411);
                    border:2px solid #4ECDC4;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">✅</div>
            <div style="color:#4ECDC4;font-weight:600;font-size:0.9rem;">Finish</div>
        </div>
    </div>
    """,
    "plan_execute": """
    <div style="display:flex;align-items:center;justify-content:center;gap:10px;padding:24px 0;flex-wrap:wrap;">
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#6C63FF22,#6C63FF11);
                    border:2px solid #6C63FF;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">❓</div>
            <div style="color:#6C63FF;font-weight:600;font-size:0.9rem;">Question</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#DDA0DD22,#DDA0DD11);
                    border:2px solid #DDA0DD;border-radius:14px;min-width:120px;">
            <div style="font-size:26px;">📋</div>
            <div style="color:#DDA0DD;font-weight:600;font-size:0.9rem;">Create Plan</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#00CC9622,#00CC9611);
                    border:2px solid #00CC96;border-radius:14px;min-width:130px;">
            <div style="font-size:26px;">⚡</div>
            <div style="color:#00CC96;font-weight:600;font-size:0.9rem;">Execute Steps</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#45B7D122,#45B7D111);
                    border:2px solid #45B7D1;border-radius:14px;min-width:120px;">
            <div style="font-size:26px;">🧩</div>
            <div style="color:#45B7D1;font-weight:600;font-size:0.9rem;">Synthesise</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#4ECDC422,#4ECDC411);
                    border:2px solid #4ECDC4;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">✅</div>
            <div style="color:#4ECDC4;font-weight:600;font-size:0.9rem;">Answer</div>
        </div>
    </div>
    """,
    "reflection": """
    <div style="display:flex;align-items:center;justify-content:center;gap:10px;padding:24px 0;flex-wrap:wrap;">
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#6C63FF22,#6C63FF11);
                    border:2px solid #6C63FF;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">❓</div>
            <div style="color:#6C63FF;font-weight:600;font-size:0.9rem;">Question</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#00CC9622,#00CC9611);
                    border:2px solid #00CC96;border-radius:14px;min-width:130px;">
            <div style="font-size:26px;">💡</div>
            <div style="color:#00CC96;font-weight:600;font-size:0.9rem;">Initial Answer</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#FF555522,#FF555511);
                    border:2px solid #FF5555;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">🔍</div>
            <div style="color:#FF5555;font-weight:600;font-size:0.9rem;">Critique</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#4ECDC422,#4ECDC411);
                    border:2px solid #4ECDC4;border-radius:14px;min-width:130px;">
            <div style="font-size:26px;">✨</div>
            <div style="color:#4ECDC4;font-weight:600;font-size:0.9rem;">Refined Answer</div>
        </div>
    </div>
    """,
    "tool_choice": """
    <div style="display:flex;align-items:center;justify-content:center;gap:10px;padding:24px 0;flex-wrap:wrap;">
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#6C63FF22,#6C63FF11);
                    border:2px solid #6C63FF;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">❓</div>
            <div style="color:#6C63FF;font-weight:600;font-size:0.9rem;">Question</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#45B7D122,#45B7D111);
                    border:2px solid #45B7D1;border-radius:14px;min-width:130px;">
            <div style="font-size:26px;">🎯</div>
            <div style="color:#45B7D1;font-weight:600;font-size:0.9rem;">Pick Best Tool</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#00CC9622,#00CC9611);
                    border:2px solid #00CC96;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">⚡</div>
            <div style="color:#00CC96;font-weight:600;font-size:0.9rem;">Execute</div>
        </div>
        <div style="font-size:24px;color:#555;">→</div>
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#4ECDC422,#4ECDC411);
                    border:2px solid #4ECDC4;border-radius:14px;min-width:110px;">
            <div style="font-size:26px;">✅</div>
            <div style="color:#4ECDC4;font-weight:600;font-size:0.9rem;">Answer</div>
        </div>
    </div>
    """,
}

st.html(_PATTERN_DIAGRAMS[selected_pattern])

# ── Comparison Table ─────────────────────────────────────────────────────

_PATTERN_DETAILS = {
    "react": {
        "when": "General-purpose tasks needing flexible multi-step reasoning",
        "pros": "Flexible, handles unexpected situations, easy to debug step-by-step",
        "cons": "Can loop inefficiently, may use extra LLM calls, order of tools not planned",
        "use_cases": "Research questions, complex calculations, information gathering",
    },
    "plan_execute": {
        "when": "Complex tasks that benefit from upfront planning",
        "pros": "Structured approach, clear progress tracking, good for multi-step tasks",
        "cons": "Rigid plan may not adapt well to surprises, extra LLM call for planning phase",
        "use_cases": "Multi-part analysis, report generation, step-by-step problem solving",
    },
    "reflection": {
        "when": "Tasks where answer quality matters more than speed",
        "pros": "Self-improving answers, catches errors, higher quality final output",
        "cons": "Slower (multiple LLM calls), higher cost, may over-critique simple answers",
        "use_cases": "Writing tasks, factual accuracy, nuanced analysis, code review",
    },
    "tool_choice": {
        "when": "Simple tasks where one tool call is sufficient",
        "pros": "Fast, minimal LLM calls, predictable cost, simple to understand",
        "cons": "Cannot chain tools, limited to single-step tasks",
        "use_cases": "Quick lookups, simple calculations, single conversions, one-shot queries",
    },
}

details = _PATTERN_DETAILS[selected_pattern]
detail_table = {
    "Aspect": ["When to use", "Pros", "Cons", "Typical use cases"],
    "Details": [details["when"], details["pros"], details["cons"], details["use_cases"]],
}
st.table(detail_table)

# ═══════════════════════════════════════════════════════════════════════════
#  Section 3 — Live Demo
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Live Demo")
st.markdown("Enter a question, pick a pattern and tools, and watch the agent work step by step.")

col_q, col_opts = st.columns([2, 1])

with col_q:
    question = st.text_input(
        "Your question",
        value="What is 25 * 17 plus the square root of 144?",
        key="agent_question",
    )

with col_opts:
    demo_pattern = st.selectbox(
        "Pattern",
        options=list(AGENT_PATTERNS.keys()),
        format_func=lambda x: AGENT_PATTERNS[x].split(" — ")[0],
        key="demo_pattern",
    )

selected_tools = st.multiselect(
    "Tools to give the agent",
    options=list(BUILTIN_TOOLS.keys()),
    default=["calculator", "web_search", "datetime"],
    format_func=lambda x: f"{x} — {BUILTIN_TOOLS[x].description[:50]}",
    key="demo_tools",
)

if st.button("▶ Run Agent", key="run_agent", type="primary"):
    if not selected_tools:
        st.warning("Please select at least one tool.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        registry = create_tool_registry(selected_tools)

        try:
            llm = get_llm_provider()
        except Exception as e:
            st.error(f"Could not initialise LLM provider. Check sidebar settings.\n\n{e}")
            st.stop()

        executor = AgentExecutor(
            llm_provider=llm,
            tool_registry=registry,
            pattern=demo_pattern,
            max_steps=8,
        )

        with st.spinner("Agent is thinking..."):
            t_start = time.time()
            result: AgentResult = executor.run(question)
            wall_time = (time.time() - t_start) * 1000

        # ── Display Steps Timeline ─────────────────────────────────────
        st.markdown("### Agent Execution Timeline")

        for step in result.steps:
            st.markdown(f"**Step {step.step_number}** {'✅ Final' if step.is_final else ''}")

            if step.thought:
                st.markdown(
                    f'<div class="step-thought"><strong>🧠 Thought:</strong> {step.thought}</div>',
                    unsafe_allow_html=True,
                )
            if step.action:
                action_str = f"<strong>🔧 Action:</strong> {step.action}"
                if step.action_input:
                    action_str += f" &nbsp;|&nbsp; <strong>Input:</strong> <code>{step.action_input}</code>"
                st.markdown(
                    f'<div class="step-action">{action_str}</div>',
                    unsafe_allow_html=True,
                )
            if step.observation:
                obs_display = step.observation[:500] + ("..." if len(step.observation) > 500 else "")
                st.markdown(
                    f'<div class="step-observation"><strong>👁️ Observation:</strong> {obs_display}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

        # ── Final Answer ───────────────────────────────────────────────
        st.success(f"**Final Answer:** {result.answer}")

        # ── Stats ──────────────────────────────────────────────────────
        st.markdown("### Run Statistics")
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric("Total Steps", len(result.steps))
        with stat_cols[1]:
            unique_tools = sorted(set(result.tools_used)) if result.tools_used else ["none"]
            st.metric("Tools Used", ", ".join(unique_tools))
        with stat_cols[2]:
            st.metric("Total Time", f"{wall_time:.0f} ms")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/2_🤖_Agents/02_🔧_Tools.py", label="← Tools & Function Calling", icon="🔧")
with col2:
    st.page_link("pages/2_🤖_Agents/04_🌐_Multi_Agent.py", label="Next: Multi-Agent Systems →", icon="🌐")
