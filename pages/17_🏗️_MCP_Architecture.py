"""Page 17 — MCP Architecture: Deep dive into Hosts, Clients, Servers, Transports."""

import streamlit as st
import json


st.title("🏗️ MCP Architecture Deep Dive")
st.markdown("Understand how Hosts, Clients, and Servers interact through the MCP protocol.")

# ── Connection Lifecycle ────────────────────────────────────────────────
st.header("1 · Connection Lifecycle")

st.markdown("""
```
 Host                    Client                  Server
  │                        │                       │
  │── create client ──────►│                       │
  │                        │── initialize ────────►│
  │                        │◄── capabilities ──────│
  │                        │── initialized ───────►│
  │                        │                       │
  │  (Normal operation — requests & notifications) │
  │                        │── tools/list ────────►│
  │                        │◄── tool list ─────────│
  │                        │── tools/call ────────►│
  │                        │◄── result ────────────│
  │                        │                       │
  │                        │── shutdown ──────────►│
  │                        │◄── ack ───────────────│
  │── destroy client ─────►│                       │
```
""")

st.info("""
**Three phases:**
1. **Initialization** — Client sends `initialize`, server responds with capabilities, client confirms with `initialized`.
2. **Operation** — Client sends requests (`tools/call`, `resources/read`, etc.), server responds.
3. **Shutdown** — Graceful disconnection.
""")

# ── Interactive Handshake ───────────────────────────────────────────────
st.header("2 · Interactive Handshake Simulator")
st.markdown("Watch the JSON-RPC messages exchanged during initialization.")

if st.button("▶️ Run Initialization Handshake", key="run_handshake"):
    from core.mcp_simulator import MCPServer, MCPClient

    server = MCPServer("demo-server", "1.0.0")
    client = MCPClient()

    with st.status("Running handshake...", expanded=True) as status:
        st.write("**Step 1:** Client sends `initialize` request")
        result = client.connect(server)

        init_request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "MCPClient", "version": "1.0.0"},
            }
        }
        st.code(json.dumps(init_request, indent=2), language="json")

        st.write("**Step 2:** Server responds with capabilities")
        st.code(json.dumps(result.get("result", {}), indent=2), language="json")

        st.write("**Step 3:** Client sends `initialized` notification")
        st.code(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}, indent=2), language="json")

        status.update(label="Handshake complete!", state="complete")

    st.success("Connection established! The client now knows what the server can do.")

# ── Host Architecture ───────────────────────────────────────────────────
st.header("3 · The Host")

st.markdown("""
The **Host** is the top-level application — think of Claude Desktop, an IDE like Cursor,
or a custom AI application. Its responsibilities:

| Responsibility | Description |
|----------------|-------------|
| **Create clients** | One client per MCP server connection |
| **Manage lifecycle** | Start, monitor, and shut down connections |
| **Security boundary** | Controls which servers can be accessed |
| **LLM integration** | Passes server capabilities to the LLM |
| **User consent** | Asks user before executing sensitive tools |
""")

st.markdown("""
```
┌──────────────────────────────────────────────────┐
│                    MCP HOST                       │
│                                                   │
│  ┌─────────┐  ┌─────────────────────────────┐    │
│  │   LLM   │  │      Client Manager         │    │
│  │ (GPT /  │  │  ┌────────┐  ┌────────┐     │    │
│  │  Claude  │◄─┤  │Client 1│  │Client 2│ ... │    │
│  │  / etc.) │  │  └───┬────┘  └───┬────┘     │    │
│  └─────────┘  └──────┼───────────┼───────────┘    │
│                      │           │                 │
└──────────────────────┼───────────┼─────────────────┘
                       ▼           ▼
                 ┌──────────┐ ┌──────────┐
                 │ Server A │ │ Server B │
                 └──────────┘ └──────────┘
```
""")

# ── Client Architecture ────────────────────────────────────────────────
st.header("4 · The Client")

st.markdown("""
Each **Client** maintains a **1:1 stateful session** with a single server:

- Sends JSON-RPC requests and receives responses
- Manages protocol version negotiation
- Handles message routing and serialization
- Maintains connection state

**Key rule:** One client connects to exactly one server. To connect to multiple
servers, the host creates multiple clients.
""")

# ── Server Architecture ────────────────────────────────────────────────
st.header("5 · The Server")

st.markdown("""
An **MCP Server** exposes capabilities through the three primitives:
""")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    ### 📄 Resources
    - Read-only data
    - Static or dynamic URIs
    - Subscribe for updates
    - Example: `db://schema/users`
    """)
with c2:
    st.markdown("""
    ### 🔧 Tools
    - Executable actions
    - JSON Schema inputs
    - LLM decides when to call
    - Example: `query(sql)`
    """)
with c3:
    st.markdown("""
    ### 💬 Prompts
    - Template interactions
    - Server-defined workflows
    - Parameterized messages
    - Example: `code_review(code)`
    """)

# ── Transport Deep Dive ────────────────────────────────────────────────
st.header("6 · Transports")

t1, t2 = st.tabs(["stdio", "HTTP + SSE"])

with t1:
    st.markdown("""
    ### Standard I/O (stdio)
    The server runs as a **child process** of the host. Messages flow through
    `stdin` and `stdout`.

    ```
    Host Process
      └── spawns Server Process
            stdin  ← JSON-RPC requests
            stdout → JSON-RPC responses
            stderr → logging (not protocol)
    ```

    **Advantages:** Simple setup, no networking, great for local tools.

    **Disadvantages:** Server must be locally installed, can't share across machines.
    """)

with t2:
    st.markdown("""
    ### HTTP + Server-Sent Events (SSE)
    The server runs as a **web service**. The client connects via HTTP:

    ```
    Client ──POST──► /message    (send requests)
    Client ◄──SSE─── /sse        (receive responses & notifications)
    ```

    **Advantages:** Remote servers, cloud-hosted, shareable across users.

    **Disadvantages:** More complex setup, requires network access, authentication needed.
    """)

# ── JSON-RPC Message Explorer ──────────────────────────────────────────
st.header("7 · JSON-RPC Message Explorer")
st.markdown("Explore the actual message formats used in MCP communication.")

msg_type = st.selectbox("Select message type:", [
    "initialize (request)",
    "tools/list (request)",
    "tools/call (request)",
    "resources/read (request)",
    "prompts/get (request)",
    "Successful response",
    "Error response",
    "Notification",
], key="msg_explorer")

messages = {
    "initialize (request)": {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05",
                   "capabilities": {},
                   "clientInfo": {"name": "MyApp", "version": "1.0"}}
    },
    "tools/list (request)": {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    "tools/call (request)": {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "query", "arguments": {"sql": "SELECT * FROM users"}}
    },
    "resources/read (request)": {
        "jsonrpc": "2.0", "id": 4, "method": "resources/read",
        "params": {"uri": "file:///config.json"}
    },
    "prompts/get (request)": {
        "jsonrpc": "2.0", "id": 5, "method": "prompts/get",
        "params": {"name": "code_review", "arguments": {"code": "def hello(): ..."}}
    },
    "Successful response": {
        "jsonrpc": "2.0", "id": 3,
        "result": {"content": [{"type": "text", "text": "Query returned 3 rows"}]}
    },
    "Error response": {
        "jsonrpc": "2.0", "id": 4,
        "error": {"code": -32601, "message": "Method not found"}
    },
    "Notification": {"jsonrpc": "2.0", "method": "notifications/resources/updated",
                     "params": {"uri": "file:///config.json"}},
}
st.code(json.dumps(messages[msg_type], indent=2), language="json")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/16_🔌_MCP_Basics.py", label="← MCP Basics", icon="🔌")
with cols[2]:
    st.page_link("pages/18_🧱_MCP_Primitives.py", label="MCP Primitives →", icon="🧱")
