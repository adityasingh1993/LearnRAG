"""Page 27 — A2A Help: Comprehensive reference for the A2A protocol track."""

import streamlit as st

st.set_page_config(page_title="A2A Help", page_icon="❓", layout="wide")

st.title("❓ A2A Protocol — Help & Reference")
st.markdown("Complete reference for the Agent-to-Agent protocol learning track.")

# ── Table of Contents ───────────────────────────────────────────────────
st.markdown("""
**Contents:**
1. [What is A2A?](#what-is-a2a)
2. [Agent Cards](#agent-cards)
3. [Task Lifecycle](#task-lifecycle)
4. [Messages & Parts](#messages-parts)
5. [Artifacts](#artifacts)
6. [Collaboration Patterns](#collaboration-patterns)
7. [Playground Guide](#playground-guide)
8. [A2A vs MCP vs Function Calling](#a2a-vs-mcp-vs-function-calling)
9. [Resources & Further Reading](#resources-further-reading)
""")

# ── What is A2A ─────────────────────────────────────────────────────────
st.header("What is A2A?")
st.markdown("""
The **Agent-to-Agent (A2A) Protocol** is an open standard by Google for
enabling AI agents to communicate, collaborate, and delegate tasks to each
other regardless of their underlying framework or vendor.

**Key principles:**
- **Agentic** — Agents collaborate as peers, not just tool providers
- **Discoverable** — Agent Cards let agents find each other
- **Task-oriented** — All work is structured as tasks with clear lifecycle
- **Modality-agnostic** — Supports text, files, and structured data
- **Opaque execution** — Agents don't need to share internal reasoning
""")

# ── Agent Cards ─────────────────────────────────────────────────────────
st.header("Agent Cards")
st.markdown("""
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Agent's display name |
| `description` | string | What the agent does |
| `url` | string | Endpoint for task submission |
| `version` | string | Agent version |
| `capabilities.streaming` | boolean | Supports SSE streaming |
| `capabilities.pushNotifications` | boolean | Supports push updates |
| `skills[].id` | string | Skill identifier |
| `skills[].name` | string | Skill display name |
| `skills[].description` | string | What the skill does |
| `skills[].tags` | string[] | Keywords for matching |
| `skills[].examples` | string[] | Example queries |
| `defaultInputModes` | string[] | Accepted input types |
| `defaultOutputModes` | string[] | Output types produced |

**Discovery URL:** `GET {agent_url}/.well-known/agent.json`
""")

# ── Task Lifecycle ──────────────────────────────────────────────────────
st.header("Task Lifecycle")
st.markdown("""
| State | Description | Transitions To |
|-------|-------------|----------------|
| `submitted` | Task created, awaiting processing | `working` |
| `working` | Agent is processing the task | `completed`, `failed`, `canceled`, `input-required` |
| `input-required` | Agent needs more information | `working` (after client responds) |
| `completed` | Task finished successfully | — (terminal) |
| `failed` | Task encountered an error | — (terminal) |
| `canceled` | Task was canceled | — (terminal) |

**Task API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `tasks/send` | POST | Create or continue a task |
| `tasks/get` | GET | Query task status |
| `tasks/cancel` | POST | Cancel a running task |
| `tasks/sendSubscribe` | POST | Send task + subscribe to SSE updates |
""")

# ── Messages & Parts ───────────────────────────────────────────────────
st.header("Messages & Parts")
st.markdown("""
| Part Type | Fields | Use Case |
|-----------|--------|----------|
| **TextPart** | `type: "text"`, `text: string` | Plain text communication |
| **FilePart** | `type: "file"`, `file.name`, `file.mimeType`, `file.bytes` | Document exchange |
| **DataPart** | `type: "data"`, `data: object` | Structured JSON data |

**Message roles:**
- `user` — From the client agent (requester)
- `agent` — From the server agent (executor)
""")

# ── Artifacts ───────────────────────────────────────────────────────────
st.header("Artifacts")
st.markdown("""
Artifacts are **deliverables** produced during task execution, distinct from
conversational messages.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Artifact identifier |
| `description` | string | What the artifact contains |
| `parts` | Part[] | Content (text, files, data) |
| `index` | integer | Ordering within the task |

**Messages vs Artifacts:**
- Messages = conversation ("I'm working on it", "Here's what I found")
- Artifacts = deliverables (the actual report, code, data)
""")

# ── Collaboration Patterns ──────────────────────────────────────────────
st.header("Collaboration Patterns")

patterns = {
    "Router": {
        "description": "Routes requests to the best specialist agent",
        "agents": "1 router + N specialists",
        "latency": "Low (single hop)",
        "use_case": "General-purpose assistant with specialist backends",
    },
    "Pipeline": {
        "description": "Tasks flow through agents in sequence",
        "agents": "N agents in series",
        "latency": "Medium (sum of stages)",
        "use_case": "Multi-step workflows (research → write → review)",
    },
    "Parallel": {
        "description": "Multiple agents work on subtasks concurrently",
        "agents": "1 orchestrator + N workers",
        "latency": "Low (max of agents)",
        "use_case": "Comparison tasks, data gathering from multiple sources",
    },
    "Debate": {
        "description": "Agents argue different positions, judge picks best",
        "agents": "2-3 debaters + 1 judge",
        "latency": "High (multiple rounds)",
        "use_case": "Critical decisions, quality assurance",
    },
    "Hierarchical": {
        "description": "Manager agents coordinate teams of worker agents",
        "agents": "Manager → Team Leads → Workers",
        "latency": "Variable",
        "use_case": "Complex enterprise workflows",
    },
}

for name, info in patterns.items():
    with st.expander(f"**{name} Pattern**"):
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Agents:** {info['agents']}")
        st.markdown(f"**Latency:** {info['latency']}")
        st.markdown(f"**Use case:** {info['use_case']}")

# ── Playground Guide ────────────────────────────────────────────────────
st.header("Playground Guide")
st.markdown("""
The **A2A Playground** offers three collaboration modes:

| Mode | Description | How it works |
|------|-------------|--------------|
| **🔀 Auto-Router** | Automatically picks the best agent | Analyzes keywords, matches against agent skills |
| **🎯 Direct Agent** | Talk to a specific agent | You choose which agent receives every message |
| **⛓️ Pipeline** | Chain agents in sequence | Output of each agent feeds into the next |

**Features:**
- **Register custom agents** — Define name, skills, and response logic
- **Task Inspector** — View full task details, messages, artifacts, state history
- **Protocol details** — See routing decisions and A2A metadata for every interaction
""")

# ── Comparison ──────────────────────────────────────────────────────────
st.header("A2A vs MCP vs Function Calling")
st.markdown("""
| Aspect | A2A | MCP | Function Calling |
|--------|-----|-----|-----------------|
| **Purpose** | Agent ↔ Agent collaboration | LLM ↔ Tool access | LLM → single function |
| **Discovery** | Agent Cards (/.well-known/) | Server capabilities | Schema registration |
| **Communication** | HTTP + JSON-RPC | JSON-RPC over stdio/SSE | Provider API |
| **Unit of work** | Task (with lifecycle) | Single request/response | Single call |
| **State** | Stateful (task tracking) | Stateless | Stateless |
| **Outputs** | Artifacts + Messages | Tool results | Function return |
| **Multi-party** | Native (many agents) | Single client-server | Single LLM-function |
| **Streaming** | SSE for task updates | SSE transport | Provider-dependent |
| **Created by** | Google | Anthropic | OpenAI (popularized) |

**How they work together:**
```
┌─────────────────────────────────────────────────┐
│            Your Application                      │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │              AI Agent                       │ │
│  │                                             │ │
│  │  Function Calling: invoke model functions   │ │
│  │  MCP: access tools & data from servers      │ │
│  │  A2A: delegate tasks to other agents        │ │
│  │                                             │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```
""")

# ── Resources & Further Reading ────────────────────────────────────────
st.header("Resources & Further Reading")
st.markdown("""
**Official Resources:**
- [A2A Protocol Specification](https://google.github.io/A2A/) — The full protocol spec
- [A2A GitHub Repository](https://github.com/google/A2A) — Reference implementations
- [Google A2A Announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — Official blog post

**Technical Deep Dives:**
- [Agent Card Schema](https://google.github.io/A2A/#/documentation?id=agent-card) — Full Agent Card specification
- [Task Object Schema](https://google.github.io/A2A/#/documentation?id=task-object) — Task lifecycle details
- [A2A Samples](https://github.com/google/A2A/tree/main/samples) — Example implementations

**Complementary Protocols:**
- [MCP Documentation](https://modelcontextprotocol.io/) — Model Context Protocol
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) — Function calling guide

**Community & Ecosystem:**
- [A2A Discussion Forum](https://github.com/google/A2A/discussions) — Community discussions
- [LangChain A2A](https://blog.langchain.dev/a2a/) — LangChain integration
- [CrewAI](https://www.crewai.com/) — Multi-agent framework
""")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/4_🤝_A2A/05_🎮_A2A_Playground.py", label="← A2A Playground", icon="🎮")
with cols[2]:
    st.page_link("app.py", label="🏠 Home", icon="🏠")
