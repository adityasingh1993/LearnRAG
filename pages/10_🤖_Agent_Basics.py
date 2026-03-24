"""
Module 10: Agent Basics
Introduction to AI Agents — what they are, how the agent loop works, and key concepts.
"""

import streamlit as st

st.set_page_config(page_title="Agent Basics | RAG Lab", page_icon="🤖", layout="wide")

from components.sidebar import render_provider_config

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
    .analogy-box {
        background: #2a1f3d;
        border: 1px solid #6C63FF44;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agent Basics")
st.markdown("*Understand what AI Agents are, how they reason and act, and when to use them.*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 1 — What Are AI Agents?
# ═══════════════════════════════════════════════════════════════════════════
st.header("What Are AI Agents?")

st.markdown("""
<div class="concept-box">
<h4>AI Agents</h4>
Systems that use LLMs as a <strong>reasoning engine</strong> to decide what actions to take,
execute those actions using <strong>tools</strong>, observe the results, and continue reasoning
until the task is complete. Unlike a plain LLM call, agents operate in a <strong>loop</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="analogy-box">
<strong>🎓 Analogy: Research Assistant</strong><br><br>
A regular LLM is like asking a friend a question — they answer from memory immediately.
An <strong>AI Agent</strong> is like hiring a research assistant — they <em>think</em> about what
information they need, <em>use tools</em> (search the web, run calculations, look up databases),
<em>check</em> the results, and iterate until they have a thorough answer.
</div>
""", unsafe_allow_html=True)

st.markdown("")
st.markdown("### LLM vs RAG vs Agent")

comparison_data = {
    "": ["Plain LLM", "RAG", "Agent"],
    "What it does": [
        "Generates text from training data",
        "Retrieves relevant docs, then generates",
        "Reasons, plans, and uses tools in a loop",
    ],
    "Has memory of external data?": ["❌ No", "✅ Yes (vector store)", "✅ Yes (tools + context)"],
    "Can take actions?": ["❌ No", "❌ No", "✅ Yes (via tools)"],
    "Multi-step reasoning?": ["🟡 Single pass", "🟡 Single pass", "✅ Iterative loop"],
    "When to use": [
        "Simple Q&A, creative writing",
        "Knowledge-base Q&A, document search",
        "Complex tasks, research, automation",
    ],
}
st.table(comparison_data)

# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 — The Agent Loop
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("The Agent Loop")
st.markdown("Agents work in a **Think → Act → Observe** cycle until the task is done.")

agent_loop_html = """
<div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:30px 0;flex-wrap:wrap;">
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#6C63FF22,#6C63FF11);
                border:2px solid #6C63FF;border-radius:16px;min-width:130px;">
        <div style="font-size:32px;">❓</div>
        <div style="color:#6C63FF;font-weight:600;margin-top:8px;">User Question</div>
        <div style="color:#666;font-size:0.8rem;">Task to accomplish</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#45B7D122,#45B7D111);
                border:2px solid #45B7D1;border-radius:16px;min-width:130px;">
        <div style="font-size:32px;">🧠</div>
        <div style="color:#45B7D1;font-weight:600;margin-top:8px;">Think</div>
        <div style="color:#666;font-size:0.8rem;">LLM reasons about next step</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#00CC9622,#00CC9611);
                border:2px solid #00CC96;border-radius:16px;min-width:130px;">
        <div style="font-size:32px;">🔧</div>
        <div style="color:#00CC96;font-weight:600;margin-top:8px;">Pick Tool</div>
        <div style="color:#666;font-size:0.8rem;">Choose action to take</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#FFEAA722,#FFEAA711);
                border:2px solid #FFEAA7;border-radius:16px;min-width:130px;">
        <div style="font-size:32px;">⚡</div>
        <div style="color:#FFEAA7;font-weight:600;margin-top:8px;">Execute</div>
        <div style="color:#666;font-size:0.8rem;">Run the tool</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#DDA0DD22,#DDA0DD11);
                border:2px solid #DDA0DD;border-radius:16px;min-width:130px;">
        <div style="font-size:32px;">👁️</div>
        <div style="color:#DDA0DD;font-weight:600;margin-top:8px;">Observe</div>
        <div style="color:#666;font-size:0.8rem;">Read tool output</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#45B7D122,#45B7D111);
                border:2px solid #45B7D1;border-radius:16px;min-width:130px;">
        <div style="font-size:32px;">🔄</div>
        <div style="color:#45B7D1;font-weight:600;margin-top:8px;">Think Again</div>
        <div style="color:#666;font-size:0.8rem;">Need more info? Loop!</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#4ECDC422,#4ECDC411);
                border:2px solid #4ECDC4;border-radius:16px;min-width:130px;">
        <div style="font-size:32px;">✅</div>
        <div style="color:#4ECDC4;font-weight:600;margin-top:8px;">Answer</div>
        <div style="color:#666;font-size:0.8rem;">Final response</div>
    </div>
</div>
"""
st.html(agent_loop_html)

st.info(
    "💡 The key insight: the LLM **decides** when to use tools and when it has enough "
    "information to answer. This decision-making loop is what makes it an *agent*, "
    "not just a chatbot."
)

# ═══════════════════════════════════════════════════════════════════════════
#  Section 3 — Key Concepts
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Key Concepts")

concepts = {
    "🔧 Tools": {
        "summary": "Functions the agent can call to interact with the outside world.",
        "detail": (
            "Tools give agents **capabilities** beyond text generation. Examples include "
            "calculators, web search, database queries, API calls, and file operations. "
            "Each tool has a name, description, and typed parameters. The LLM reads tool "
            "descriptions and decides which one to call, with what arguments."
        ),
    },
    "🧠 Reasoning": {
        "summary": "The agent's ability to think step-by-step before acting.",
        "detail": (
            "Reasoning is the **Thought** step in the agent loop. The LLM analyses the "
            "current situation, considers what information it has and what it still needs, "
            "and decides the best next action. Techniques like Chain-of-Thought prompting "
            "improve reasoning quality."
        ),
    },
    "📋 Planning": {
        "summary": "Creating a multi-step plan before executing any actions.",
        "detail": (
            "Some agent patterns (like Plan-and-Execute) have the LLM create an explicit "
            "plan first: a numbered list of steps to solve the problem. The agent then "
            "executes each step, adapting the plan if unexpected results occur. Planning "
            "helps with complex, multi-step tasks."
        ),
    },
    "💾 Memory": {
        "summary": "Retaining information across steps and conversations.",
        "detail": (
            "**Short-term memory** is the conversation history within a single agent run — "
            "the chain of thoughts, actions, and observations. "
            "**Long-term memory** persists across conversations using vector stores or databases. "
            "Memory lets agents build on past work and avoid repeating steps."
        ),
    },
    "👥 Multi-Agent": {
        "summary": "Multiple specialised agents collaborating on a task.",
        "detail": (
            "Complex tasks can be split among specialised agents. For example, a Researcher "
            "agent gathers information, a Coder agent writes code, and an Editor agent "
            "reviews the output. Multi-agent systems can work in sequence (pipeline), "
            "in parallel, or in a debate/discussion format."
        ),
    },
    "🛡️ Guardrails": {
        "summary": "Safety checks that constrain agent behaviour.",
        "detail": (
            "Guardrails prevent agents from taking harmful actions: input validation "
            "rejects dangerous requests, tool-call limits prevent infinite loops, "
            "output checks catch hallucinations or sensitive data leaks. In production, "
            "guardrails are essential for safe, reliable agent deployment."
        ),
    },
}

for title, info in concepts.items():
    with st.expander(f"**{title}** — {info['summary']}"):
        st.markdown(info["detail"])

# ═══════════════════════════════════════════════════════════════════════════
#  Section 4 — Quiz
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🧪 Quick Knowledge Check")

with st.form("quiz_agents"):
    q1 = st.radio(
        "**Q1:** What distinguishes an AI Agent from a plain LLM call?",
        [
            "Agents are always more accurate",
            "Agents can reason, use tools, and loop until the task is done",
            "Agents don't need a language model",
            "Agents only work with RAG pipelines",
        ],
        index=None,
    )
    q2 = st.radio(
        "**Q2:** What is the correct order in the basic agent loop?",
        [
            "Execute → Think → Observe → Answer",
            "Think → Pick Tool → Execute → Observe → (repeat or Answer)",
            "Answer → Think → Execute → Observe",
            "Pick Tool → Execute → Think → Answer",
        ],
        index=None,
    )
    q3 = st.radio(
        "**Q3:** What are 'tools' in the context of AI Agents?",
        [
            "Physical hardware the agent controls",
            "Functions the agent can call to interact with external systems",
            "Pre-written answers the agent selects from",
            "Other LLMs that the agent delegates to",
        ],
        index=None,
    )
    q4 = st.radio(
        "**Q4:** Why is planning useful for agents?",
        [
            "It makes the agent respond faster",
            "It reduces the number of LLM calls to exactly one",
            "It helps break complex tasks into manageable steps before acting",
            "It removes the need for tools",
        ],
        index=None,
    )
    submitted = st.form_submit_button("Check Answers", type="primary")

if submitted:
    score = 0
    answers = {
        "q1": ("Agents can reason, use tools, and loop until the task is done",
               "Correct! The reasoning-action loop with tools is what defines an agent."),
        "q2": ("Think → Pick Tool → Execute → Observe → (repeat or Answer)",
               "Correct! Think first, act, observe, then decide whether to loop or finish."),
        "q3": ("Functions the agent can call to interact with external systems",
               "Correct! Tools are callable functions — calculators, search APIs, databases, etc."),
        "q4": ("It helps break complex tasks into manageable steps before acting",
               "Correct! Planning decomposes hard problems into a sequence of solvable steps."),
    }

    for i, (key, (correct, explanation)) in enumerate(answers.items(), 1):
        response = [q1, q2, q3, q4][i - 1]
        if response == correct:
            score += 1
            st.success(f"Q{i}: {explanation}")
        elif response:
            st.error(f"Q{i}: Not quite. {explanation}")

    if score == 4:
        st.balloons()
        st.success(f"🎉 Perfect score! {score}/4 — You're ready to explore Tools!")
    elif score > 0:
        st.info(f"Score: {score}/4 — Good effort! Review the sections above and try again.")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/9_❓_Help.py", label="← Help & Reference", icon="❓")
with col2:
    st.page_link("pages/11_🔧_Tools.py", label="Next: Tools & Function Calling →", icon="🔧")
