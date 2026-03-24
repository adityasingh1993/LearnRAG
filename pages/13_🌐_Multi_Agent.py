"""
Module 12: Multi-Agent Systems
Learn about Router, Orchestrator, and Debate patterns for coordinating multiple agents.
"""

import streamlit as st

st.set_page_config(page_title="Multi-Agent Systems | RAG Lab", page_icon="🌐", layout="wide")

from components.sidebar import render_provider_config, get_llm_provider
from core.tools import BUILTIN_TOOLS, create_tool_registry, Tool, ToolParameter
from core.agent_loop import AgentExecutor, AGENT_PATTERNS

render_provider_config()

st.title("🌐 Multi-Agent Systems")
st.markdown("*Coordinate multiple specialised agents to solve complex problems.*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 1 — What Are Multi-Agent Systems?
# ═══════════════════════════════════════════════════════════════════════════

st.header("What Are Multi-Agent Systems?")

st.markdown("""
A **multi-agent system** (MAS) uses two or more agents that collaborate, compete,
or specialise to handle tasks that a single agent would struggle with alone.
Instead of building one monolithic agent with every tool and every prompt,
you decompose work across several focused agents.

Three foundational coordination patterns dominate modern MAS design:

| Pattern | Idea | Analogy |
|---------|------|---------|
| **Router** | One agent classifies the query and dispatches to the best specialist | A receptionist directing you to the right department |
| **Orchestrator** | A master agent decomposes a complex task into subtasks and delegates each | A project manager assigning work to team members |
| **Debate** | Multiple agents argue different perspectives, then a judge synthesises | A panel discussion followed by an editorial summary |
""")

st.markdown("#### How a Router Agent works")

st.html("""
<div style="display:flex;align-items:center;justify-content:center;gap:10px;padding:24px 0;flex-wrap:wrap;">
  <div style="text-align:center;padding:14px 18px;border-radius:14px;border:2px solid #4ECDC4;
              background:#4ECDC415;min-width:100px;">
    <div style="font-size:26px;">👤</div>
    <div style="font-size:11px;color:#4ECDC4;font-weight:600;margin-top:4px;">User Query</div>
  </div>
  <div style="font-size:22px;color:#4ECDC4;">→</div>
  <div style="text-align:center;padding:14px 18px;border-radius:14px;border:2px solid #FF6B6B;
              background:#FF6B6B15;min-width:110px;">
    <div style="font-size:26px;">🧭</div>
    <div style="font-size:11px;color:#FF6B6B;font-weight:600;margin-top:4px;">Router Agent</div>
    <div style="font-size:9px;color:#aaa;margin-top:2px;">Classifies intent</div>
  </div>
  <div style="font-size:22px;color:#FF6B6B;">→</div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="text-align:center;padding:10px 14px;border-radius:12px;border:2px solid #45B7D1;
                background:#45B7D115;min-width:120px;">
      <div style="font-size:18px;">📚</div>
      <div style="font-size:10px;color:#45B7D1;font-weight:600;">RAG Agent</div>
    </div>
    <div style="text-align:center;padding:10px 14px;border-radius:12px;border:2px solid #96CEB4;
                background:#96CEB415;min-width:120px;">
      <div style="font-size:18px;">🔢</div>
      <div style="font-size:10px;color:#96CEB4;font-weight:600;">Calculator Agent</div>
    </div>
    <div style="text-align:center;padding:10px 14px;border-radius:12px;border:2px solid #DDA0DD;
                background:#DDA0DD15;min-width:120px;">
      <div style="font-size:18px;">🔍</div>
      <div style="font-size:10px;color:#DDA0DD;font-weight:600;">Search Agent</div>
    </div>
  </div>
  <div style="font-size:22px;color:#96CEB4;">→</div>
  <div style="text-align:center;padding:14px 18px;border-radius:14px;border:2px solid #FFEAA7;
              background:#FFEAA715;min-width:100px;">
    <div style="font-size:26px;">✅</div>
    <div style="font-size:11px;color:#FFEAA7;font-weight:600;margin-top:4px;">Final Answer</div>
  </div>
</div>
""")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 — Enterprise Patterns
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("🏗️ Enterprise Patterns")
st.markdown("Explore each coordination pattern in detail.")

pattern_tabs = st.tabs(["🧭 Router Pattern", "🎯 Orchestrator Pattern", "💬 Debate Pattern"])

with pattern_tabs[0]:
    st.subheader("Router Pattern")
    st.markdown("""
    A **Router** acts as a classifier at the front of the system. It inspects the
    incoming query, determines which category it falls into, and forwards it to the
    specialist agent best equipped to handle that category.

    **Strengths:**
    - Fast — only one specialist runs per query
    - Clean separation of concerns — each specialist has its own tools and prompt
    - Easy to extend — add a new specialist without modifying existing ones

    **Weaknesses:**
    - Only as good as the classification step — mis-routing leads to bad answers
    - Cannot handle queries that span multiple domains in a single pass
    """)

    st.html("""
    <div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:18px 0;flex-wrap:wrap;">
      <div style="padding:12px 16px;border-radius:12px;border:2px solid #FF6B6B;background:#FF6B6B15;text-align:center;min-width:90px;">
        <div style="font-size:22px;">📩</div><div style="font-size:10px;color:#FF6B6B;font-weight:600;">Query In</div>
      </div>
      <div style="color:#FF6B6B;font-size:20px;">→</div>
      <div style="padding:14px 20px;border-radius:14px;border:2px solid #FF6B6B;background:#FF6B6B20;text-align:center;">
        <div style="font-size:24px;">🧭</div>
        <div style="font-size:11px;color:#FF6B6B;font-weight:700;">Router</div>
        <div style="font-size:9px;color:#aaa;">classify(query)</div>
      </div>
      <div style="color:#aaa;font-size:20px;">→</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <div style="padding:10px;border-radius:10px;border:2px dashed #45B7D1;text-align:center;min-width:80px;">
          <div style="font-size:11px;color:#45B7D1;font-weight:600;">Specialist A</div>
        </div>
        <div style="padding:10px;border-radius:10px;border:2px dashed #96CEB4;text-align:center;min-width:80px;">
          <div style="font-size:11px;color:#96CEB4;font-weight:600;">Specialist B</div>
        </div>
        <div style="padding:10px;border-radius:10px;border:2px dashed #DDA0DD;text-align:center;min-width:80px;">
          <div style="font-size:11px;color:#DDA0DD;font-weight:600;">Specialist C</div>
        </div>
      </div>
      <div style="color:#aaa;font-size:20px;">→</div>
      <div style="padding:12px 16px;border-radius:12px;border:2px solid #FFEAA7;background:#FFEAA715;text-align:center;min-width:90px;">
        <div style="font-size:22px;">✅</div><div style="font-size:10px;color:#FFEAA7;font-weight:600;">Answer</div>
      </div>
    </div>
    """)

with pattern_tabs[1]:
    st.subheader("Orchestrator Pattern")
    st.markdown("""
    An **Orchestrator** (or "manager") agent receives a complex query, breaks it into
    subtasks, and delegates each subtask to the best available specialist. After all
    subtasks complete, the orchestrator synthesises the results into one answer.

    **Strengths:**
    - Handles multi-step, multi-domain queries (e.g., "search the web for X, calculate Y, then summarise")
    - Explicit planning step means better transparency and debuggability
    - Natural parallelism — independent subtasks can run concurrently

    **Weaknesses:**
    - Slower — requires a planning LLM call plus all subtask calls
    - More expensive — more tokens overall
    - Risk of over-decomposition on simple queries
    """)

    st.html("""
    <div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:18px 0;flex-wrap:wrap;">
      <div style="padding:12px 16px;border-radius:12px;border:2px solid #4ECDC4;background:#4ECDC415;text-align:center;">
        <div style="font-size:22px;">📩</div><div style="font-size:10px;color:#4ECDC4;font-weight:600;">Complex Query</div>
      </div>
      <div style="color:#4ECDC4;font-size:20px;">→</div>
      <div style="padding:14px 20px;border-radius:14px;border:2px solid #FF8C94;background:#FF8C9420;text-align:center;">
        <div style="font-size:24px;">🎯</div>
        <div style="font-size:11px;color:#FF8C94;font-weight:700;">Orchestrator</div>
        <div style="font-size:9px;color:#aaa;">decompose → delegate → synthesise</div>
      </div>
      <div style="color:#aaa;font-size:20px;">→</div>
      <div style="display:flex;flex-direction:column;gap:6px;">
        <div style="padding:8px 12px;border-radius:10px;border:1px solid #45B7D1;background:#45B7D110;text-align:center;">
          <span style="font-size:10px;color:#45B7D1;">Subtask 1 → Worker A</span>
        </div>
        <div style="padding:8px 12px;border-radius:10px;border:1px solid #96CEB4;background:#96CEB410;text-align:center;">
          <span style="font-size:10px;color:#96CEB4;">Subtask 2 → Worker B</span>
        </div>
        <div style="padding:8px 12px;border-radius:10px;border:1px solid #DDA0DD;background:#DDA0DD10;text-align:center;">
          <span style="font-size:10px;color:#DDA0DD;">Subtask 3 → Worker C</span>
        </div>
      </div>
      <div style="color:#aaa;font-size:20px;">→</div>
      <div style="padding:12px 16px;border-radius:12px;border:2px solid #FFEAA7;background:#FFEAA715;text-align:center;">
        <div style="font-size:22px;">🧩</div><div style="font-size:10px;color:#FFEAA7;font-weight:600;">Synthesised Answer</div>
      </div>
    </div>
    """)

with pattern_tabs[2]:
    st.subheader("Debate Pattern")
    st.markdown("""
    In the **Debate** (or "adversarial") pattern, multiple agents independently answer
    the same question — potentially from different perspectives or with different prompts.
    A **judge** agent then evaluates all responses and synthesises the best answer.

    **Strengths:**
    - Reduces single-point-of-failure reasoning errors
    - Surfaces diverse perspectives and edge cases
    - The judge sees multiple drafts, improving final quality

    **Weaknesses:**
    - Most expensive — N agents + 1 judge per query
    - Slowest — all agents must finish before the judge can act
    - Overkill for simple factual queries
    """)

    st.html("""
    <div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:18px 0;flex-wrap:wrap;">
      <div style="padding:12px 16px;border-radius:12px;border:2px solid #4ECDC4;background:#4ECDC415;text-align:center;">
        <div style="font-size:22px;">📩</div><div style="font-size:10px;color:#4ECDC4;font-weight:600;">Query</div>
      </div>
      <div style="color:#4ECDC4;font-size:20px;">→</div>
      <div style="display:flex;flex-direction:column;gap:6px;">
        <div style="padding:10px 14px;border-radius:10px;border:2px solid #FF6B6B;background:#FF6B6B15;text-align:center;">
          <div style="font-size:10px;color:#FF6B6B;font-weight:600;">🅰️ Agent A (Optimist)</div>
        </div>
        <div style="padding:10px 14px;border-radius:10px;border:2px solid #45B7D1;background:#45B7D115;text-align:center;">
          <div style="font-size:10px;color:#45B7D1;font-weight:600;">🅱️ Agent B (Sceptic)</div>
        </div>
        <div style="padding:10px 14px;border-radius:10px;border:2px solid #96CEB4;background:#96CEB415;text-align:center;">
          <div style="font-size:10px;color:#96CEB4;font-weight:600;">🅲 Agent C (Neutral)</div>
        </div>
      </div>
      <div style="color:#aaa;font-size:20px;">→</div>
      <div style="padding:14px 20px;border-radius:14px;border:2px solid #DDA0DD;background:#DDA0DD20;text-align:center;">
        <div style="font-size:24px;">⚖️</div>
        <div style="font-size:11px;color:#DDA0DD;font-weight:700;">Judge</div>
        <div style="font-size:9px;color:#aaa;">evaluate & synthesise</div>
      </div>
      <div style="color:#aaa;font-size:20px;">→</div>
      <div style="padding:12px 16px;border-radius:12px;border:2px solid #FFEAA7;background:#FFEAA715;text-align:center;">
        <div style="font-size:22px;">✅</div><div style="font-size:10px;color:#FFEAA7;font-weight:600;">Best Answer</div>
      </div>
    </div>
    """)

# ═══════════════════════════════════════════════════════════════════════════
#  Section 3 — Live Demo: Router Agent
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("🚀 Live Demo: Router Agent")
st.markdown("""
Enter a question below. The system will:
1. Use the LLM to **classify** your query into a category
2. **Route** it to the specialist agent with the right tools
3. Show the full routing flow and the specialist's answer
""")

SPECIALISTS = {
    "math": {
        "label": "🔢 Math Specialist",
        "color": "#96CEB4",
        "tools": ["calculator", "unit_convert"],
        "description": "Handles calculations and unit conversions",
    },
    "search": {
        "label": "🔍 Search Specialist",
        "color": "#45B7D1",
        "tools": ["web_search", "datetime"],
        "description": "Searches the web and checks current info",
    },
    "text": {
        "label": "📝 Text Specialist",
        "color": "#DDA0DD",
        "tools": ["text_stats", "string_transform", "json_parse"],
        "description": "Analyses and transforms text",
    },
    "general": {
        "label": "💬 General Specialist",
        "color": "#FFEAA7",
        "tools": ["web_search", "calculator", "datetime"],
        "description": "Handles broad or mixed queries",
    },
}

question = st.text_input(
    "Ask anything:",
    placeholder="e.g.  What is sqrt(144) + 25?  /  Search for Python news  /  Count the words in 'hello world'",
    key="router_question",
)

if st.button("🧭 Route & Answer", type="primary", disabled=not question):
    with st.spinner("Classifying query..."):
        try:
            llm = get_llm_provider()

            classification_prompt = (
                "You are a query classifier. Given a user query, respond with EXACTLY one word — "
                "the category that best matches:\n"
                "- math (calculations, conversions, numbers)\n"
                "- search (web lookup, current events, factual questions)\n"
                "- text (word counts, text analysis, string manipulation, JSON)\n"
                "- general (everything else)\n\n"
                f"Query: {question}\n\nCategory:"
            )
            classification_resp = llm.generate(classification_prompt, temperature=0.0, max_tokens=10)
            category = classification_resp.text.strip().lower().rstrip(".")

            valid_categories = list(SPECIALISTS.keys())
            if category not in valid_categories:
                for cat in valid_categories:
                    if cat in category:
                        category = cat
                        break
                else:
                    category = "general"

            spec = SPECIALISTS[category]

            # Show routing decision
            st.html(f"""
            <div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:20px 0;flex-wrap:wrap;">
              <div style="padding:14px 18px;border-radius:14px;border:2px solid #4ECDC4;background:#4ECDC415;text-align:center;">
                <div style="font-size:22px;">👤</div>
                <div style="font-size:10px;color:#4ECDC4;font-weight:600;">User</div>
                <div style="font-size:9px;color:#aaa;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{question[:40]}</div>
              </div>
              <div style="color:#FF6B6B;font-size:20px;">→</div>
              <div style="padding:14px 18px;border-radius:14px;border:2px solid #FF6B6B;background:#FF6B6B15;text-align:center;">
                <div style="font-size:22px;">🧭</div>
                <div style="font-size:10px;color:#FF6B6B;font-weight:600;">Router</div>
                <div style="font-size:9px;color:#aaa;">Category: <b>{category}</b></div>
              </div>
              <div style="color:{spec['color']};font-size:20px;">→</div>
              <div style="padding:14px 18px;border-radius:14px;border:2px solid {spec['color']};background:{spec['color']}15;text-align:center;">
                <div style="font-size:22px;">{spec['label'].split()[0]}</div>
                <div style="font-size:10px;color:{spec['color']};font-weight:600;">{spec['label'].split(' ', 1)[1]}</div>
                <div style="font-size:9px;color:#aaa;">Tools: {', '.join(spec['tools'])}</div>
              </div>
              <div style="color:#FFEAA7;font-size:20px;">→</div>
              <div style="padding:14px 18px;border-radius:14px;border:2px solid #FFEAA7;background:#FFEAA715;text-align:center;">
                <div style="font-size:22px;">✅</div>
                <div style="font-size:10px;color:#FFEAA7;font-weight:600;">Answer</div>
              </div>
            </div>
            """)

            st.info(f"**Routing decision:** `{category}` → **{spec['label']}** — {spec['description']}")

            with st.spinner(f"Running {spec['label']}..."):
                registry = create_tool_registry(spec["tools"])
                agent = AgentExecutor(
                    llm_provider=llm,
                    tool_registry=registry,
                    pattern="react",
                    max_steps=6,
                )
                result = agent.run(question)

            st.success(f"**Answer:** {result.answer}")

            with st.expander("🔍 Agent Reasoning Steps"):
                for step in result.steps:
                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        st.markdown(
                            f'<div style="border-left:3px solid #45B7D1;padding:8px 12px;margin:4px 0;'
                            f'background:#45B7D110;border-radius:0 8px 8px 0;">'
                            f'<b style="color:#45B7D1;">Step {step.step_number} — Thought</b><br>'
                            f'<span style="font-size:13px;">{step.thought or "—"}</span></div>',
                            unsafe_allow_html=True,
                        )
                    with cols[1]:
                        action_color = "#4CAF50" if not step.is_final else "#FFEAA7"
                        st.markdown(
                            f'<div style="border-left:3px solid {action_color};padding:8px 12px;margin:4px 0;'
                            f'background:{action_color}10;border-radius:0 8px 8px 0;">'
                            f'<b style="color:{action_color};">Action</b><br>'
                            f'<span style="font-size:13px;">{step.action or "—"}</span></div>',
                            unsafe_allow_html=True,
                        )
                    with cols[2]:
                        obs = step.observation or (step.action_input.get("answer", "—") if isinstance(step.action_input, dict) else "—")
                        st.markdown(
                            f'<div style="border-left:3px solid #FFC107;padding:8px 12px;margin:4px 0;'
                            f'background:#FFC10710;border-radius:0 8px 8px 0;">'
                            f'<b style="color:#FFC107;">Observation</b><br>'
                            f'<span style="font-size:13px;">{obs[:300]}</span></div>',
                            unsafe_allow_html=True,
                        )

            st.caption(
                f"Tools used: {', '.join(result.tools_used) or 'none'} | "
                f"Steps: {len(result.steps)} | "
                f"Duration: {result.total_duration_ms:.0f} ms"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/12_🔄_Agent_Patterns.py", label="← Agent Patterns", icon="🔄")
with col2:
    st.page_link("pages/14_🎮_Agent_Playground.py", label="Next: Agent Playground →", icon="🎮")
