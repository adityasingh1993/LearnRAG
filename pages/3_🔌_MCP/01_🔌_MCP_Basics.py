"""Page 16 — MCP Basics: Introduction to the Model Context Protocol."""

import streamlit as st

st.set_page_config(page_title="MCP Basics", page_icon="🔌", layout="wide")

# ── Hero ────────────────────────────────────────────────────────────────
st.title("🔌 Model Context Protocol (MCP) — Basics")
st.markdown("""
> **MCP** is an open standard created by Anthropic that provides a **universal way**
> for AI assistants to connect to external data sources and tools — like a "USB-C port
> for AI".  Instead of building custom integrations for every data source, MCP gives
> a single protocol that any AI host can use to talk to any server.
""")

# ── Why MCP? ────────────────────────────────────────────────────────────
st.header("Why Does MCP Matter?")

cols = st.columns(3)
with cols[0]:
    st.markdown("""
    ### 🔗 Before MCP
    - Every AI app builds **custom connectors**
    - N apps × M tools = **N×M integrations**
    - Fragmented, hard to maintain
    """)
with cols[1]:
    st.markdown("""
    ### ⚡ With MCP
    - One **standard protocol** for all
    - N apps + M servers = **N+M integrations**
    - Build once, connect anywhere
    """)
with cols[2]:
    st.markdown("""
    ### 🏆 Benefits
    - **Interoperability** across AI platforms
    - **Security** — controlled data access
    - **Reusability** — share MCP servers
    """)

# ── Architecture Overview ───────────────────────────────────────────────
st.header("MCP Architecture at a Glance")

st.markdown("""
```
┌──────────────────────────────────────────────────────────────┐
│                         MCP HOST                             │
│                   (e.g. Claude Desktop, IDE)                 │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ MCP Client A│  │ MCP Client B│  │ MCP Client C│         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                  │
└─────────┼────────────────┼────────────────┼──────────────────┘
          │  JSON-RPC      │  JSON-RPC      │  JSON-RPC
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │ MCP Server │   │ MCP Server │   │ MCP Server │
   │  (Weather) │   │ (Database) │   │   (Files)  │
   └────────────┘   └────────────┘   └────────────┘
```
""")

st.info("""
**Key roles:**
- **Host** — The AI application (Claude Desktop, an IDE, your custom app)
- **Client** — Maintains a 1:1 connection to a server; created by the host
- **Server** — Exposes resources, tools, and prompts to clients
""")

# ── Core Concepts ───────────────────────────────────────────────────────
st.header("Core Concepts")

tab1, tab2, tab3, tab4 = st.tabs(["Resources", "Tools", "Prompts", "Transports"])

with tab1:
    st.markdown("""
    ### 📄 Resources
    Resources provide **read-only data** that the AI can use as context.
    Think of them as files, database records, or API responses that the
    model can read but not modify.

    | Property | Description |
    |----------|-------------|
    | `uri` | Unique identifier (e.g. `file:///readme.md`) |
    | `name` | Human-readable name |
    | `mimeType` | Content type (`text/plain`, `application/json`, …) |
    | `text` / `blob` | The actual content |

    **Use cases:** Database schemas, config files, documentation, live data feeds.
    """)

with tab2:
    st.markdown("""
    ### 🔧 Tools
    Tools are **executable actions** that the AI can invoke.
    The server defines the tool's name, description, and input schema;
    the LLM decides when and how to call them.

    | Property | Description |
    |----------|-------------|
    | `name` | Unique tool identifier |
    | `description` | What the tool does (used by the LLM) |
    | `inputSchema` | JSON Schema for parameters |

    **Use cases:** Run SQL queries, call APIs, send emails, execute code.
    """)

with tab3:
    st.markdown("""
    ### 💬 Prompts
    Prompts are **reusable prompt templates** that servers expose.
    They allow the server to suggest structured interactions.

    | Property | Description |
    |----------|-------------|
    | `name` | Prompt identifier |
    | `description` | What the prompt is for |
    | `arguments` | Parameters that fill template slots |

    **Use cases:** "Generate a code review", "Write a SQL query for X",
    "Summarize this document".
    """)

with tab4:
    st.markdown("""
    ### 🔀 Transports
    MCP uses **JSON-RPC 2.0** messages over two transport types:

    | Transport | How it works | When to use |
    |-----------|-------------|-------------|
    | **stdio** | Server runs as a subprocess; messages over stdin/stdout | Local tools, CLI integrations |
    | **HTTP + SSE** | Server is a web service; Server-Sent Events for streaming | Remote servers, cloud deployments |

    Both transports carry the same JSON-RPC messages — the protocol
    layer is transport-agnostic.
    """)

# ── MCP vs Direct API ──────────────────────────────────────────────────
st.header("MCP vs Direct API Integration")

comparison = {
    "Aspect": ["Protocol", "Discovery", "Security", "Reusability", "Streaming", "Maintenance"],
    "Direct API": [
        "Custom per service",
        "Manual configuration",
        "Per-integration auth",
        "Not reusable",
        "Varies",
        "High — every change breaks things",
    ],
    "MCP": [
        "Standard JSON-RPC",
        "Auto-discovery of capabilities",
        "Controlled by transport layer",
        "Any MCP client can use any server",
        "Built-in via SSE",
        "Low — server updates don't break clients",
    ],
}
st.table(comparison)

# ── Quick Quiz ──────────────────────────────────────────────────────────
st.header("Quick Quiz")

q1 = st.radio("1. What role does the MCP Host play?",
               ["Exposes tools and resources", "Manages MCP clients and contains the LLM",
                "Stores data for the protocol"], index=None, key="mcp_q1")
if q1:
    if q1 == "Manages MCP clients and contains the LLM":
        st.success("Correct! The Host is the AI application that creates and manages clients.")
    else:
        st.error("Not quite — the Host is the AI application (like Claude Desktop) that creates MCP clients.")

q2 = st.radio("2. Which MCP primitive lets the AI *execute* an action?",
               ["Resource", "Prompt", "Tool"], index=None, key="mcp_q2")
if q2:
    if q2 == "Tool":
        st.success("Correct! Tools are executable actions the AI can invoke.")
    else:
        st.error("Not quite — Tools are the primitive for executing actions.")

q3 = st.radio("3. What message format does MCP use?",
               ["GraphQL", "REST", "JSON-RPC 2.0"], index=None, key="mcp_q3")
if q3:
    if q3 == "JSON-RPC 2.0":
        st.success("Correct! MCP uses JSON-RPC 2.0 over stdio or HTTP+SSE transports.")
    else:
        st.error("Not quite — MCP is built on JSON-RPC 2.0.")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/2_🤖_Agents/06_❓_Agent_Help.py", label="← Agent Help", icon="❓")
with cols[2]:
    st.page_link("pages/3_🔌_MCP/02_🏗️_MCP_Architecture.py", label="MCP Architecture →", icon="🏗️")
