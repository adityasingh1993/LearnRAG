"""Page 22 — A2A Basics: Introduction to the Agent-to-Agent Protocol."""

import streamlit as st


st.title("🤝 Agent-to-Agent (A2A) Protocol — Basics")
st.markdown("""
> **A2A** is an open protocol by Google that enables AI agents to **communicate,
> collaborate, and delegate tasks** to each other — regardless of framework or vendor.
> Think of it as HTTP for agent collaboration.
""")

# ── Why A2A? ────────────────────────────────────────────────────────────
st.header("Why Does A2A Matter?")

cols = st.columns(3)
with cols[0]:
    st.markdown("""
    ### 🏝️ Before A2A
    - Agents are **isolated silos**
    - Can't discover other agents
    - No standard way to delegate
    - Custom integration per pair
    """)
with cols[1]:
    st.markdown("""
    ### ⚡ With A2A
    - Agents **find each other** via Agent Cards
    - Standard **task delegation** protocol
    - Rich message passing (text, files, data)
    - Framework-agnostic collaboration
    """)
with cols[2]:
    st.markdown("""
    ### 🏆 Benefits
    - **Interoperability** across platforms
    - **Specialization** — each agent does one thing well
    - **Scalability** — add agents without rewriting
    - **Transparency** — auditable task lifecycle
    """)

# ── Architecture Overview ───────────────────────────────────────────────
st.header("A2A Architecture at a Glance")

st.markdown("""
```
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│   Client    │  A2A     │   Server    │  A2A     │   Server    │
│   Agent     │ protocol │   Agent 1   │ protocol │   Agent 2   │
│ (Requester) ├─────────►│ (Executor)  ├─────────►│ (Executor)  │
│             │◄─────────┤             │◄─────────┤             │
└─────────────┘  Tasks   └─────────────┘  Tasks   └─────────────┘
       │                        │                        │
       │  /.well-known/agent.json                       │
       │◄───────────────────────┘                       │
       │◄───────────────────────────────────────────────┘
                   Agent Discovery
```
""")

st.info("""
**Key roles:**
- **Client Agent** — Sends tasks to other agents; discovers agents via their Agent Cards
- **Server Agent** — Receives and executes tasks; publishes an Agent Card
- **Agent Card** — JSON metadata at `/.well-known/agent.json` describing capabilities
""")

# ── Core Concepts ───────────────────────────────────────────────────────
st.header("Core Concepts")

tab1, tab2, tab3, tab4 = st.tabs(["Agent Cards", "Tasks", "Messages & Parts", "Artifacts"])

with tab1:
    st.markdown("""
    ### 🪪 Agent Cards
    An Agent Card is a JSON document published at `/.well-known/agent.json`
    that describes an agent's capabilities.

    | Field | Description |
    |-------|-------------|
    | `name` | Agent's human-readable name |
    | `description` | What the agent does |
    | `url` | Endpoint for sending tasks |
    | `skills` | List of capabilities with tags |
    | `capabilities` | Streaming, push notifications |
    | `defaultInputModes` | Accepted input types (text, image, …) |
    | `defaultOutputModes` | Output types the agent produces |

    **Use case:** A client agent fetches Agent Cards to discover who can help.
    """)

with tab2:
    st.markdown("""
    ### 📋 Tasks
    A **Task** is the fundamental unit of work in A2A.

    | State | Meaning |
    |-------|---------|
    | `submitted` | Task created, waiting to be picked up |
    | `working` | Agent is actively processing |
    | `input-required` | Agent needs more info from the client |
    | `completed` | Task finished successfully |
    | `failed` | Task encountered an error |
    | `canceled` | Task was canceled |

    ```
    submitted → working → completed
                  ↓            ↑
            input-required ────┘
                  ↓
                failed / canceled
    ```
    """)

with tab3:
    st.markdown("""
    ### 💬 Messages & Parts
    Communication happens through **Messages** containing **Parts**:

    | Part Type | Description | Example |
    |-----------|-------------|---------|
    | `TextPart` | Plain or markdown text | "Summarize this article" |
    | `FilePart` | Binary file with MIME type | PDF, image, audio |
    | `DataPart` | Structured JSON data | `{"key": "value"}` |

    Messages have a `role`:
    - **`user`** — From the client agent
    - **`agent`** — From the server agent
    """)

with tab4:
    st.markdown("""
    ### 📦 Artifacts
    **Artifacts** are the outputs produced by an agent during task execution.

    | Field | Description |
    |-------|-------------|
    | `name` | Artifact identifier |
    | `description` | What this artifact contains |
    | `parts` | List of Parts (text, files, data) |
    | `index` | Ordering index |

    **Examples:** Generated reports, transformed data, code files, images.

    Artifacts are distinct from messages — they represent **deliverables**
    rather than conversational content.
    """)

# ── MCP vs A2A ──────────────────────────────────────────────────────────
st.header("MCP vs A2A — Complementary Protocols")

st.markdown("""
MCP and A2A solve **different problems** and work together:

| Aspect | MCP | A2A |
|--------|-----|-----|
| **Purpose** | Connect LLMs to tools & data | Connect agents to agents |
| **Relationship** | Client → Server (tool use) | Agent → Agent (delegation) |
| **Discovery** | Server capabilities | Agent Cards at well-known URL |
| **Communication** | JSON-RPC 2.0 | HTTP + JSON (REST-like) |
| **Unit of work** | Single tool call | Task with lifecycle |
| **Statefulness** | Stateless calls | Stateful task tracking |
| **Outputs** | Tool results | Artifacts |
| **Created by** | Anthropic | Google |

**Together:** An agent uses **MCP** to access tools and data, and uses **A2A**
to collaborate with other agents. They are complementary, not competing.
""")

st.markdown("""
```
┌─────────────────────────────────────────────────┐
│                  Your Agent                      │
│                                                  │
│  Uses MCP to access:     Uses A2A to delegate:  │
│  ┌──────────┐            ┌──────────┐           │
│  │ Database │            │ Research │           │
│  │  Server  │            │  Agent   │           │
│  └──────────┘            └──────────┘           │
│  ┌──────────┐            ┌──────────┐           │
│  │  File    │            │  Writer  │           │
│  │  Server  │            │  Agent   │           │
│  └──────────┘            └──────────┘           │
└─────────────────────────────────────────────────┘
```
""")

# ── Quick Quiz ──────────────────────────────────────────────────────────
st.header("Quick Quiz")

q1 = st.radio("1. What is an Agent Card?",
               ["A tool definition with JSON Schema",
                "JSON metadata describing an agent's capabilities",
                "A message format for agent communication"],
               index=None, key="a2a_q1")
if q1:
    if q1 == "JSON metadata describing an agent's capabilities":
        st.success("Correct! Agent Cards describe what an agent can do, published at /.well-known/agent.json")
    else:
        st.error("Not quite — Agent Cards are JSON metadata describing an agent's capabilities.")

q2 = st.radio("2. What states can a Task be in?",
               ["pending, running, done",
                "submitted, working, input-required, completed, failed, canceled",
                "open, closed, archived"],
               index=None, key="a2a_q2")
if q2:
    if "submitted" in q2:
        st.success("Correct! Tasks follow a lifecycle: submitted → working → completed (with branches for input-required, failed, canceled).")
    else:
        st.error("Not quite — A2A tasks use: submitted, working, input-required, completed, failed, canceled.")

q3 = st.radio("3. How do MCP and A2A relate?",
               ["They are competing standards",
                "A2A replaces MCP",
                "They are complementary — MCP for tools, A2A for agent collaboration"],
               index=None, key="a2a_q3")
if q3:
    if "complementary" in q3:
        st.success("Correct! MCP connects agents to tools; A2A connects agents to other agents.")
    else:
        st.error("Not quite — MCP and A2A are complementary protocols solving different problems.")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/21_❓_MCP_Help.py", label="← MCP Help", icon="❓")
with cols[2]:
    st.page_link("pages/23_🪪_Agent_Cards.py", label="Agent Cards →", icon="🪪")
