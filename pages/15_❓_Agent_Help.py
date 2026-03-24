"""
Module 14: Agent Help & Reference
Comprehensive reference page for AI Agents — patterns, tools, multi-agent, RAG integration.
"""

import streamlit as st

st.set_page_config(page_title="Agent Help | RAG Lab", page_icon="❓", layout="wide")

from components.sidebar import render_provider_config

render_provider_config()

st.title("❓ Agent Help & Reference")
st.markdown("*Everything you need to know about AI Agents — patterns, tools, and integration.*")
st.markdown("---")

# ── Table of Contents ─────────────────────────────────────────────────────
st.markdown("""
**Jump to a section:**
[Agent Patterns](#agent-patterns) · [Built-in Tools](#built-in-tools) ·
[Custom Tools](#custom-tools) · [Multi-Agent Patterns](#multi-agent-patterns) ·
[RAG + Agents](#rag-agents) · [Token & Cost](#token-cost) ·
[Resources & Further Reading](#resources-further-reading)
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🔄 Agent Patterns")

st.markdown("""
An agent pattern defines *how* the LLM reasons about a task and uses tools.
Each pattern makes different trade-offs between accuracy, cost, and speed.
""")

patterns = {
    "ReAct (Reason + Act)": {
        "how": (
            "The agent enters a loop: **Think** about the task → **Act** by calling a tool "
            "→ **Observe** the result → repeat until it has enough information to give a "
            "final answer."
        ),
        "when": "General-purpose tasks that may need multiple tool calls. The most versatile pattern.",
        "tradeoffs": (
            "Most token-heavy because the full reasoning chain is sent to the LLM at every "
            "step. Can loop indefinitely on ambiguous tasks — the max-steps limit prevents this."
        ),
    },
    "Plan-and-Execute": {
        "how": (
            "Phase 1 — the LLM creates a numbered step-by-step **plan**. "
            "Phase 2 — each step is executed in order (using tools or reasoning). "
            "Phase 3 — the results are **synthesised** into a final answer."
        ),
        "when": "Complex, multi-step tasks where a clear plan improves reliability (e.g., research + calculate + summarise).",
        "tradeoffs": (
            "More predictable than ReAct because steps are explicit. However, the plan is "
            "static — if step 3 reveals the plan was wrong, it can't easily re-plan."
        ),
    },
    "Reflection": {
        "how": (
            "The agent generates an **initial answer** (optionally using tools), then "
            "writes a **critique** of its own answer, and finally produces a **refined answer** "
            "that addresses the weaknesses."
        ),
        "when": "Tasks where quality matters more than speed — writing, analysis, nuanced questions.",
        "tradeoffs": (
            "Higher quality answers at the cost of 2–3× the tokens. The critique step "
            "catches errors that a single-pass approach would miss."
        ),
    },
    "Tool Choice": {
        "how": (
            "The LLM picks the **single best tool** for the query, calls it once, and "
            "then formulates the final answer from the tool output."
        ),
        "when": "Simple, single-tool queries where the answer comes from one tool call (e.g., a calculation, a lookup).",
        "tradeoffs": (
            "Cheapest and fastest pattern — only 2 LLM calls total. But it cannot chain "
            "multiple tools or recover from a wrong tool choice."
        ),
    },
}

for name, info in patterns.items():
    with st.expander(f"**{name}**"):
        st.markdown(f"**How it works:** {info['how']}")
        st.markdown(f"**Best used when:** {info['when']}")
        st.markdown(f"**Trade-offs:** {info['tradeoffs']}")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🔧 Built-in Tools")

st.markdown("""
The agent toolkit includes 7 built-in tools across three categories.
Each tool has a defined interface that the agent can call during its reasoning loop.
""")

st.markdown("""
| Tool | Category | Description | Parameters |
|------|----------|-------------|------------|
| **calculator** | math | Evaluate math expressions (`+`, `-`, `*`, `/`, `sqrt`, `sin`, `log`, `pi`, …) | `expression: string` |
| **datetime** | utility | Get the current date and time | `format: string` *(optional)* |
| **text_stats** | text | Analyse text — character, word, sentence counts, average word length | `text: string` |
| **web_search** | search | Search the web for information (simulated for educational purposes) | `query: string` |
| **json_parse** | utility | Parse and pretty-print a JSON string | `data: string` |
| **unit_convert** | math | Convert between units: km/miles, kg/lbs, °C/°F, m/ft, L/gal | `value: string`, `from_unit: string`, `to_unit: string` |
| **string_transform** | text | Transform text: uppercase, lowercase, title, reverse, word/char count, strip | `text: string`, `operation: string` |
""")

st.info(
    "**Tip:** In the Agent Playground you can select any combination of these tools. "
    "Agents with fewer, more focused tools tend to perform better — the LLM has less "
    "to choose from and makes better decisions."
)

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🛠️ Custom Tools")

st.markdown("""
You can create your own tools by writing a Python function and registering it
with the tool system. Custom tools run in a **sandboxed environment** with access
only to safe builtins (`str`, `int`, `float`, `len`, `range`, `list`, `dict`,
`sum`, `min`, `max`, `abs`, `round`).
""")

with st.expander("**How to create a custom tool**"):
    st.markdown("""
    Use `create_custom_tool()` with four arguments:

    1. **name** — a unique identifier (e.g., `"word_frequency"`)
    2. **description** — what the tool does (the LLM reads this to decide when to use it)
    3. **parameters** — a list of `{"name": ..., "type": ..., "description": ...}` dicts
    4. **code** — a Python code string that sets a `result` variable

    Example:
    ```python
    tool = create_custom_tool(
        name="word_frequency",
        description="Count how often each word appears in a text",
        parameters=[{"name": "text", "type": "string", "description": "Input text"}],
        code=\"\"\"
    words = text.lower().split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    result = str(dict(sorted(freq.items(), key=lambda x: -x[1])[:10]))
    \"\"\",
    )
    ```

    The agent can then call this tool by name during its reasoning loop.
    """)

with st.expander("**Security note**"):
    st.markdown("""
    Custom tool code runs with `__builtins__` restricted to a safe subset. Imports,
    file I/O, network access, and `exec`/`eval` are **not available** inside custom
    tool code. This is intentional — the sandbox is designed for educational
    experimentation, not production use.
    """)

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🌐 Multi-Agent Patterns")

st.markdown("""
Multi-agent systems coordinate multiple specialised agents to handle complex scenarios.
Three patterns are explored in the Multi-Agent page:
""")

multi_agent = {
    "Router Pattern": {
        "how": (
            "A **router agent** classifies each incoming query into a category "
            "(e.g., math, search, text) and dispatches it to the specialist agent "
            "best suited for that category."
        ),
        "when": "Systems that handle diverse query types and need fast, focused answers.",
        "tradeoffs": (
            "Fastest multi-agent pattern — only one specialist runs. "
            "Failure mode: mis-classification routes to the wrong specialist."
        ),
    },
    "Orchestrator Pattern": {
        "how": (
            "A **master agent** decomposes a complex query into subtasks, assigns each "
            "to a specialist worker agent, collects results, and synthesises a unified answer."
        ),
        "when": "Complex queries that span multiple domains (e.g., 'research X, calculate Y, and summarise').",
        "tradeoffs": (
            "Most capable pattern for complex tasks. "
            "Costs more tokens (planning + N subtasks + synthesis) and is slower."
        ),
    },
    "Debate Pattern": {
        "how": (
            "Multiple agents independently answer the same question (possibly with "
            "different perspectives or prompts). A **judge agent** evaluates all "
            "responses and produces the best synthesis."
        ),
        "when": "High-stakes decisions where diverse viewpoints reduce reasoning errors.",
        "tradeoffs": (
            "Highest quality but also the most expensive (N agents + 1 judge). "
            "Overkill for simple factual queries."
        ),
    },
}

for name, info in multi_agent.items():
    with st.expander(f"**{name}**"):
        st.markdown(f"**How it works:** {info['how']}")
        st.markdown(f"**Best used when:** {info['when']}")
        st.markdown(f"**Trade-offs:** {info['tradeoffs']}")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📚 RAG + Agents")

st.markdown("""
RAG (Retrieval-Augmented Generation) and Agents are complementary. An agent can
use a **rag_search** tool that internally runs a full RAG pipeline — chunking,
embedding, retrieval, and generation — to answer questions grounded in a knowledge base.

**How it works in the Agent Playground:**

1. Enable **"Give agent RAG capabilities"** and paste your knowledge base text
2. The app builds a lightweight RAG pipeline (TF-IDF embeddings + NumPy vector store)
3. A `rag_search` tool is registered in the agent's tool registry
4. When the agent decides it needs knowledge-base information, it calls `rag_search(query="...")`
5. The RAG pipeline retrieves relevant chunks, generates an answer, and returns it to the agent

**Why is this powerful?**

The agent can decide *when* to use RAG versus other tools. For example:
- "What does our policy say about refunds?" → agent calls `rag_search`
- "What is 15% of $249?" → agent calls `calculator`
- "What is our refund policy for items over $200?" → agent calls `rag_search`, then `calculator`

This is the foundation of **agentic RAG** — the dominant architecture for enterprise
AI assistants.
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("💰 Token & Cost")

st.markdown("""
Different agent patterns have very different token footprints. Understanding this
helps you choose the right pattern for your budget and latency requirements.
""")

st.markdown("""
| Pattern | LLM Calls per Query | Relative Token Cost | Notes |
|---------|---------------------|---------------------|-------|
| **Tool Choice** | 2 | ⭐ Lowest | Pick tool → generate answer |
| **ReAct** | 2–N (up to max_steps) | ⭐⭐⭐⭐ Highest | Full history resent each step; grows quadratically |
| **Plan-and-Execute** | 2 + N (plan steps) | ⭐⭐⭐ High | Plan + execute each step + synthesise |
| **Reflection** | 2–3 | ⭐⭐ Moderate | Initial + critique + refine |
""")

st.markdown("""
**Optimisation tips:**
- Use **Tool Choice** for simple, single-tool queries — cheapest and fastest
- Use **ReAct** with a low `max_steps` (3–4) for moderately complex tasks
- Reserve **Plan-and-Execute** for genuinely multi-step problems
- **Reflection** is worth the cost when answer quality is critical
- Fewer tools = smaller system prompt = fewer tokens per LLM call
- RAG tool calls add their own embedding + generation tokens on top of the agent's tokens
""")

# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📚 Resources & Further Reading")

resources = {
    "Foundational Papers": [
        ("[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)",
         "The original paper introducing the Thought → Action → Observation loop."),
        ("[Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)",
         "Decompose a problem into a plan, then execute each step."),
        ("[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)",
         "Agents that reflect on and learn from their mistakes across episodes."),
    ],
    "Agent Frameworks": [
        ("[LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)",
         "The most popular Python framework for building LLM agents."),
        ("[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)",
         "Autonomous agent that chains LLM calls to accomplish goals."),
        ("[CrewAI](https://github.com/joaomdmoura/crewAI)",
         "Framework for orchestrating role-playing AI agents in teams."),
        ("[AutoGen (Microsoft)](https://github.com/microsoft/autogen)",
         "Multi-agent conversation framework supporting diverse patterns."),
    ],
    "Function Calling & Tools": [
        ("[OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)",
         "How to give GPT models the ability to call functions with structured arguments."),
        ("[Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)",
         "Claude's tool-use interface for structured tool calls."),
    ],
    "Multi-Agent Systems": [
        ("[Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)",
         "ChatDev — multiple agents collaborate to write software end-to-end."),
        ("[MetaGPT: Multi-Agent Framework](https://arxiv.org/abs/2308.00352)",
         "Assigns different roles (PM, architect, engineer) to different agents."),
    ],
    "Agentic RAG": [
        ("[Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)",
         "Agents that evaluate retrieval quality and self-correct."),
        ("[Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)",
         "LLM decides when to retrieve and critiques its own generation."),
    ],
}

for category, links in resources.items():
    with st.expander(f"**{category}**", expanded=True):
        for link, desc in links:
            st.markdown(f"- {link} — {desc}")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/14_🎮_Agent_Playground.py", label="← Agent Playground", icon="🎮")
with col2:
    st.page_link("app.py", label="Home →", icon="🏠")
