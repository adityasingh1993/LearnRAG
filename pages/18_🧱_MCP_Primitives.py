"""Page 18 — MCP Primitives: Interactive exploration of Resources, Tools, Prompts."""

import streamlit as st
import json
from core.mcp_simulator import (
    MCPServer, MCPClient, MCPResource, MCPTool, MCPToolParam, MCPPrompt,
    create_weather_server, create_database_server, create_filesystem_server,
    DEMO_SERVERS,
)

st.set_page_config(page_title="MCP Primitives", page_icon="🧱", layout="wide")

st.title("🧱 MCP Primitives — Interactive Explorer")
st.markdown("""
MCP servers expose three kinds of capabilities. On this page you can
**connect to demo servers** and interact with each primitive live.
""")

# ── Server Picker ───────────────────────────────────────────────────────
st.header("1 · Connect to a Demo Server")

server_name = st.selectbox(
    "Choose a server:",
    list(DEMO_SERVERS.keys()),
    format_func=lambda k: f"{DEMO_SERVERS[k]['icon']} {k.title()} — {DEMO_SERVERS[k]['description']}",
    key="prim_server",
)

server: MCPServer = DEMO_SERVERS[server_name]["factory"]()
client = MCPClient()
conn = client.connect(server)

st.success(f"Connected to **{server.name}** v{server.version}")

caps = server.capabilities
cap_cols = st.columns(3)
cap_cols[0].metric("Resources", "✅" if caps.resources else "❌")
cap_cols[1].metric("Tools", "✅" if caps.tools else "❌")
cap_cols[2].metric("Prompts", "✅" if caps.prompts else "❌")

# ── Resources Tab ───────────────────────────────────────────────────────
st.header("2 · Explore Primitives")

tab_r, tab_t, tab_p, tab_log = st.tabs(["📄 Resources", "🔧 Tools", "💬 Prompts", "📋 Request Log"])

with tab_r:
    resources = client.list_resources()
    if not resources:
        st.info("This server exposes no resources.")
    else:
        st.markdown(f"**{len(resources)} resources** available:")
        for res in resources:
            with st.expander(f"`{res['uri']}` — {res['name']}"):
                st.markdown(f"**Description:** {res['description']}")
                st.markdown(f"**MIME Type:** `{res.get('mimeType', 'text/plain')}`")
                if st.button(f"Read `{res['uri']}`", key=f"read_{res['uri']}"):
                    data = client.read_resource(res["uri"])
                    contents = data.get("contents", [])
                    if contents:
                        st.code(contents[0].get("text", ""), language="text")
                    else:
                        st.warning("No content returned.")

with tab_t:
    tools = client.list_tools()
    if not tools:
        st.info("This server exposes no tools.")
    else:
        st.markdown(f"**{len(tools)} tools** available:")
        for tool in tools:
            with st.expander(f"🔧 `{tool['name']}` — {tool['description']}"):
                schema = tool.get("inputSchema", {})
                props = schema.get("properties", {})
                required = schema.get("required", [])

                st.markdown("**Input Schema:**")
                st.json(schema)

                st.markdown("**Try it:**")
                args = {}
                for pname, pinfo in props.items():
                    args[pname] = st.text_input(
                        f"{pname} ({'required' if pname in required else 'optional'}): {pinfo.get('description', '')}",
                        key=f"tool_{tool['name']}_{pname}",
                    )

                if st.button(f"Call `{tool['name']}`", key=f"call_{tool['name']}"):
                    filtered_args = {k: v for k, v in args.items() if v}
                    result = client.call_tool(tool["name"], filtered_args)
                    if result.get("success"):
                        st.success(f"**Result:** {result['result']}")
                    else:
                        st.error(f"**Error:** {result.get('result', 'Unknown error')}")
                    st.caption(f"Duration: {result.get('duration_ms', 0):.1f} ms")

with tab_p:
    prompts = client.list_prompts()
    if not prompts:
        st.info("This server exposes no prompts.")
    else:
        st.markdown(f"**{len(prompts)} prompts** available:")
        for prompt in prompts:
            with st.expander(f"💬 `{prompt['name']}` — {prompt['description']}"):
                prompt_args = prompt.get("arguments", [])
                st.markdown("**Arguments:**")
                for arg in prompt_args:
                    st.markdown(f"- `{arg['name']}`: {arg.get('description', '')} {'(required)' if arg.get('required') else '(optional)'}")

                st.markdown("**Try it:**")
                values = {}
                for arg in prompt_args:
                    values[arg["name"]] = st.text_input(
                        f"{arg['name']}:",
                        key=f"prompt_{prompt['name']}_{arg['name']}",
                    )

                if st.button(f"Render `{prompt['name']}`", key=f"render_{prompt['name']}"):
                    filtered = {k: v for k, v in values.items() if v}
                    result = client.get_prompt(prompt["name"], filtered)
                    messages = result.get("messages", [])
                    for msg in messages:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", {})
                        text = content.get("text", str(content)) if isinstance(content, dict) else str(content)
                        st.chat_message(role).markdown(text)

with tab_log:
    st.markdown("All JSON-RPC messages exchanged with this server:")
    log = server.get_request_log()
    if log:
        for entry in log:
            with st.expander(f"`{entry['method']}` — {entry.get('duration_ms', 0):.1f} ms"):
                st.json(entry)
    else:
        st.info("No requests yet. Interact with the primitives above to see messages here.")

# ── Build Your Own ──────────────────────────────────────────────────────
st.header("3 · How Primitives Work Together")

st.markdown("""
In a real scenario, an LLM uses all three primitives together:

```
User: "What's the weather like in London?"

1. LLM reads RESOURCE → weather://current/london  (gets context)
2. LLM calls TOOL     → get_temperature("london") (gets precise data)
3. LLM uses PROMPT    → weather_report("London")  (structures response)
4. LLM generates a natural language answer using all gathered information
```

The **server** decides what to expose.  The **LLM** decides what to use.
The **protocol** ensures they can communicate.
""")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/17_🏗️_MCP_Architecture.py", label="← MCP Architecture", icon="🏗️")
with cols[2]:
    st.page_link("pages/19_🛠️_MCP_Server_Builder.py", label="MCP Server Builder →", icon="🛠️")
