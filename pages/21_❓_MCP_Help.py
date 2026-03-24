"""Page 21 — MCP Help: Comprehensive reference for MCP features."""

import streamlit as st

st.set_page_config(page_title="MCP Help", page_icon="❓", layout="wide")

st.title("❓ MCP — Help & Reference")
st.markdown("Complete reference for the Model Context Protocol learning track.")

# ── Table of Contents ───────────────────────────────────────────────────
st.markdown("""
**Contents:**
1. [What is MCP?](#what-is-mcp)
2. [Architecture Components](#architecture-components)
3. [The Three Primitives](#the-three-primitives)
4. [Transport Layer](#transport-layer)
5. [Server Builder Guide](#server-builder-guide)
6. [Playground Guide](#playground-guide)
7. [MCP vs Alternatives](#mcp-vs-alternatives)
8. [Resources & Further Reading](#resources-further-reading)
""")

# ── What is MCP ─────────────────────────────────────────────────────────
st.header("What is MCP?")
st.markdown("""
The **Model Context Protocol (MCP)** is an open standard created by Anthropic
that defines how AI applications communicate with external data sources and
tools. Think of it as a **universal adapter** — like USB-C for AI.

**Key goals:**
- **Standardization** — One protocol instead of N×M custom integrations
- **Interoperability** — Any MCP client can connect to any MCP server
- **Security** — Controlled, auditable access to tools and data
- **Composability** — Connect to multiple servers simultaneously
""")

# ── Architecture Components ─────────────────────────────────────────────
st.header("Architecture Components")

components = {
    "Host": {
        "description": "The AI application that contains the LLM and manages MCP clients.",
        "examples": "Claude Desktop, Cursor IDE, custom applications",
        "responsibilities": "Creates clients, manages lifecycle, enforces security, handles user consent",
    },
    "Client": {
        "description": "Maintains a stateful 1:1 session with an MCP server.",
        "examples": "One client per server connection inside the host",
        "responsibilities": "Protocol negotiation, message routing, capability tracking",
    },
    "Server": {
        "description": "Exposes resources, tools, and prompts to connected clients.",
        "examples": "Database server, file system server, API wrapper server",
        "responsibilities": "Declare capabilities, handle requests, return results",
    },
}

for name, info in components.items():
    with st.expander(f"**{name}**", expanded=True):
        st.markdown(f"**What:** {info['description']}")
        st.markdown(f"**Examples:** {info['examples']}")
        st.markdown(f"**Responsibilities:** {info['responsibilities']}")

# ── The Three Primitives ────────────────────────────────────────────────
st.header("The Three Primitives")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📄 Resources
    **Purpose:** Provide context data to the LLM

    | Aspect | Detail |
    |--------|--------|
    | Direction | Server → Client (read-only) |
    | Control | Application-controlled |
    | URI format | `scheme://path` |
    | Types | Static or dynamic |
    | Updates | Subscription-based |

    **Methods:**
    - `resources/list` — List available resources
    - `resources/read` — Read a specific resource
    - `resources/subscribe` — Watch for changes
    """)

with col2:
    st.markdown("""
    ### 🔧 Tools
    **Purpose:** Let the LLM execute actions

    | Aspect | Detail |
    |--------|--------|
    | Direction | Client → Server (execute) |
    | Control | Model-controlled |
    | Schema | JSON Schema for inputs |
    | Safety | Requires user consent |
    | Idempotency | Not guaranteed |

    **Methods:**
    - `tools/list` — List available tools
    - `tools/call` — Execute a tool
    """)

with col3:
    st.markdown("""
    ### 💬 Prompts
    **Purpose:** Reusable interaction templates

    | Aspect | Detail |
    |--------|--------|
    | Direction | Server → Client (template) |
    | Control | User-controlled |
    | Arguments | Named, typed parameters |
    | Output | List of messages |
    | Discovery | Listed by the server |

    **Methods:**
    - `prompts/list` — List available prompts
    - `prompts/get` — Render a prompt template
    """)

# ── Transport Layer ─────────────────────────────────────────────────────
st.header("Transport Layer")

st.markdown("""
| Feature | stdio | HTTP + SSE |
|---------|-------|------------|
| **Server location** | Local (subprocess) | Remote (web service) |
| **Message flow** | stdin/stdout | POST + Server-Sent Events |
| **Setup complexity** | Low | Medium |
| **Use case** | Local dev tools, CLIs | Cloud services, shared servers |
| **Authentication** | OS-level | HTTP auth (OAuth, API keys) |
| **Streaming** | Via stdout | Native SSE streaming |

Both transports carry identical **JSON-RPC 2.0** messages.
""")

# ── Server Builder Guide ───────────────────────────────────────────────
st.header("Server Builder Guide")

st.markdown("""
The **Server Builder** page lets you create a custom MCP server interactively:

1. **Configure** — Set server name and version
2. **Add Resources** — Define read-only data with URIs
3. **Add Tools** — Create executable actions with parameters and handler code
4. **Add Prompts** — Build reusable prompt templates with arguments
5. **Build & Test** — Instantiate the server and test with a simulated client
6. **Export Code** — Get the generated Python code using the `FastMCP` SDK

**Tips:**
- Start with a simple resource to see how data flows
- Tool handlers receive parameters as keyword arguments
- Prompt templates use `{arg_name}` syntax for placeholders
""")

# ── Playground Guide ────────────────────────────────────────────────────
st.header("Playground Guide")

st.markdown("""
The **MCP Playground** simulates a complete MCP environment:

| Feature | Description |
|---------|-------------|
| **Multi-server** | Connect to multiple demo servers simultaneously |
| **Explore** | Discover all resources, tools, and prompts across servers |
| **Interact** | Send raw JSON-RPC requests and see responses |
| **Chat** | Simulated LLM that automatically uses MCP to answer questions |
| **Protocol Log** | See every JSON-RPC message exchanged |

**Demo Servers Available:**
- 🌤️ **Weather** — Weather data, alerts, forecasts
- 🗄️ **Database** — SQL queries on users/orders tables
- 📁 **Filesystem** — Read files, list directories
""")

# ── MCP vs Alternatives ────────────────────────────────────────────────
st.header("MCP vs Alternatives")

st.markdown("""
| Aspect | MCP | Function Calling | LangChain Tools | Custom APIs |
|--------|-----|-----------------|-----------------|-------------|
| **Standard** | Open protocol | Provider-specific | Framework-specific | Proprietary |
| **Discovery** | Automatic | Manual registration | Manual registration | Manual |
| **Transport** | stdio / HTTP+SSE | HTTP | In-process | HTTP |
| **Multi-server** | Yes (native) | No | Via config | Manual |
| **Resources** | First-class | Not supported | Not standard | Manual |
| **Prompts** | First-class | Not supported | Prompt templates | Manual |
| **Security** | Transport-level | API key | Varies | Custom |
| **Streaming** | Built-in (SSE) | Provider-dependent | Varies | Custom |
""")

# ── Resources & Further Reading ────────────────────────────────────────
st.header("Resources & Further Reading")

st.markdown("""
**Official Resources:**
- [MCP Specification](https://spec.modelcontextprotocol.io/) — The full protocol spec
- [MCP Documentation](https://modelcontextprotocol.io/) — Official docs and guides
- [MCP GitHub](https://github.com/modelcontextprotocol) — Reference implementations
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Python server/client SDK
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript SDK

**Tutorials & Guides:**
- [Building an MCP Server](https://modelcontextprotocol.io/quickstart/server) — Official quickstart
- [Building an MCP Client](https://modelcontextprotocol.io/quickstart/client) — Client quickstart
- [Anthropic MCP Blog Post](https://www.anthropic.com/news/model-context-protocol) — Announcement

**Community:**
- [MCP Servers Directory](https://github.com/modelcontextprotocol/servers) — Community MCP servers
- [Awesome MCP](https://github.com/punkpeye/awesome-mcp-servers) — Curated list of MCP servers
""")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/20_🎮_MCP_Playground.py", label="← MCP Playground", icon="🎮")
with cols[2]:
    st.page_link("pages/22_🤝_A2A_Basics.py", label="A2A Basics →", icon="🤝")
