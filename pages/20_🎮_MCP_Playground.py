"""Page 20 — MCP Playground: Connect a Host to multiple servers and interact."""

import streamlit as st
import json
import time
from core.mcp_simulator import (
    MCPHost, MCPServer, MCPClient, DEMO_SERVERS,
)


st.title("🎮 MCP Playground")
st.markdown("""
Simulate a full MCP environment: connect a **Host** to multiple **Servers**,
discover capabilities, call tools, read resources, and watch the protocol messages fly.
""")

# ── Session State ───────────────────────────────────────────────────────
if "mcp_host" not in st.session_state:
    st.session_state.mcp_host = MCPHost("Learning Lab Host")
    st.session_state.mcp_servers = {}
    st.session_state.mcp_messages = []
    st.session_state.mcp_chat = []

host: MCPHost = st.session_state.mcp_host

# ── Sidebar: Server Connections ─────────────────────────────────────────
with st.sidebar:
    st.header("🔌 Server Connections")

    for sname, sinfo in DEMO_SERVERS.items():
        connected = sname in st.session_state.mcp_servers
        label = f"{sinfo['icon']} {sname.title()}"
        if connected:
            st.success(f"{label} — Connected")
        else:
            if st.button(f"Connect {label}", key=f"connect_{sname}"):
                server = sinfo["factory"]()
                result = host.connect_server(server)
                st.session_state.mcp_servers[sname] = server
                st.session_state.mcp_messages.append({
                    "type": "connection",
                    "server": sname,
                    "result": result,
                    "timestamp": time.time(),
                })
                st.rerun()

    st.divider()
    if st.session_state.mcp_servers:
        st.metric("Connected Servers", len(st.session_state.mcp_servers))

        all_tools = host.get_all_tools()
        total_tools = sum(len(v) for v in all_tools.values())
        st.metric("Available Tools", total_tools)

        all_resources = host.get_all_resources()
        total_res = sum(len(v) for v in all_resources.values())
        st.metric("Available Resources", total_res)

    if st.button("🔄 Reset All", key="mcp_reset"):
        st.session_state.mcp_host = MCPHost("Learning Lab Host")
        st.session_state.mcp_servers = {}
        st.session_state.mcp_messages = []
        st.session_state.mcp_chat = []
        st.rerun()

# ── Main Area ───────────────────────────────────────────────────────────
if not st.session_state.mcp_servers:
    st.info("👈 Connect to one or more servers from the sidebar to get started.")
    st.stop()

# ── Flow Diagram ────────────────────────────────────────────────────────
st.header("Connected Architecture")

server_names = list(st.session_state.mcp_servers.keys())
server_boxes = "  ".join(
    f"┌─{'─' * max(8, len(n)+2)}─┐" for n in server_names
)
server_labels = "  ".join(
    f"│ {DEMO_SERVERS[n]['icon']} {n.title():<{max(6, len(n))}} │" for n in server_names
)
server_bottoms = "  ".join(
    f"└─{'─' * max(8, len(n)+2)}─┘" for n in server_names
)

st.code(f"""
┌──────────────────────────────────────┐
│           MCP HOST                   │
│  ┌────────────────────────────────┐  │
│  │  Clients: {len(server_names):<22} │  │
│  └────────────────────────────────┘  │
└──────────────┬───────────────────────┘
               │ JSON-RPC
    {server_boxes}
    {server_labels}
    {server_bottoms}
""", language="text")

# ── Tabs ────────────────────────────────────────────────────────────────
tab_explore, tab_interact, tab_chat, tab_log = st.tabs([
    "🔍 Explore", "⚡ Interact", "💬 Chat Simulation", "📋 Protocol Log"
])

with tab_explore:
    st.subheader("Discover All Capabilities")

    for sname, server in st.session_state.mcp_servers.items():
        client = host.clients.get(sname)
        if not client:
            continue

        with st.expander(f"{DEMO_SERVERS[sname]['icon']} **{sname.title()}** Server", expanded=True):
            ec1, ec2, ec3 = st.columns(3)

            with ec1:
                st.markdown("**📄 Resources**")
                resources = client.list_resources()
                for r in resources:
                    st.markdown(f"- `{r['uri']}`\n  {r['name']}")
                if not resources:
                    st.caption("None")

            with ec2:
                st.markdown("**🔧 Tools**")
                tools = client.list_tools()
                for t in tools:
                    st.markdown(f"- `{t['name']}`\n  {t['description']}")
                if not tools:
                    st.caption("None")

            with ec3:
                st.markdown("**💬 Prompts**")
                prompts = client.list_prompts()
                for p in prompts:
                    st.markdown(f"- `{p['name']}`\n  {p['description']}")
                if not prompts:
                    st.caption("None")

with tab_interact:
    st.subheader("Direct Interaction")

    inter_server = st.selectbox(
        "Select server:",
        list(st.session_state.mcp_servers.keys()),
        format_func=lambda k: f"{DEMO_SERVERS[k]['icon']} {k.title()}",
        key="inter_server",
    )

    server_obj = st.session_state.mcp_servers[inter_server]
    client_obj = host.clients.get(inter_server)

    action = st.selectbox("Action:", [
        "resources/list", "resources/read",
        "tools/list", "tools/call",
        "prompts/list", "prompts/get",
    ], key="inter_action")

    params = {}
    if action == "resources/read":
        resources = client_obj.list_resources()
        if resources:
            uri = st.selectbox("Resource URI:", [r["uri"] for r in resources], key="inter_uri")
            params = {"uri": uri}
    elif action == "tools/call":
        tools = client_obj.list_tools()
        if tools:
            tool_name = st.selectbox("Tool:", [t["name"] for t in tools], key="inter_tool")
            params["name"] = tool_name
            schema = next((t for t in tools if t["name"] == tool_name), {}).get("inputSchema", {})
            arguments = {}
            for pname, pinfo in schema.get("properties", {}).items():
                arguments[pname] = st.text_input(f"{pname}:", key=f"inter_arg_{pname}")
            params["arguments"] = {k: v for k, v in arguments.items() if v}
    elif action == "prompts/get":
        prompts = client_obj.list_prompts()
        if prompts:
            p_name = st.selectbox("Prompt:", [p["name"] for p in prompts], key="inter_prompt")
            params["name"] = p_name
            prompt_obj = next((p for p in prompts if p["name"] == p_name), {})
            arguments = {}
            for arg in prompt_obj.get("arguments", []):
                arguments[arg["name"]] = st.text_input(f"{arg['name']}:", key=f"inter_parg_{arg['name']}")
            params["arguments"] = {k: v for k, v in arguments.items() if v}

    if st.button("▶️ Send Request", key="send_inter"):
        result = server_obj.handle_request(action, params)
        st.session_state.mcp_messages.append({
            "type": "interaction",
            "server": inter_server,
            "method": action,
            "params": params,
            "result": result,
            "timestamp": time.time(),
        })
        st.json(result)

with tab_chat:
    st.subheader("Simulated LLM + MCP Chat")
    st.markdown("""
    This simulates how an LLM Host would use MCP: you type a message,
    the "LLM" identifies which server and tool to use, and executes it.
    """)

    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.mcp_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("protocol"):
                    with st.expander("🔍 Protocol Details"):
                        st.json(msg["protocol"])

    user_input = st.chat_input("Ask something (e.g. 'What's the temperature in London?')")

    if user_input:
        st.session_state.mcp_chat.append({"role": "user", "content": user_input})

        response_parts = []
        protocol_details = []

        query_lower = user_input.lower()

        for sname, server in st.session_state.mcp_servers.items():
            client = host.clients.get(sname)
            if not client:
                continue

            tools = client.list_tools()
            for tool in tools:
                tool_words = tool["name"].lower().split("_") + tool["description"].lower().split()
                if any(w in query_lower for w in tool_words if len(w) > 3):
                    first_param = list(tool.get("inputSchema", {}).get("properties", {}).keys())
                    arg_val = user_input
                    for word in ["what", "is", "the", "in", "for", "get", "show", "me", "please", "?", "'"]:
                        arg_val = arg_val.replace(word, "")
                    arg_val = arg_val.strip()

                    args = {first_param[0]: arg_val} if first_param else {}
                    result = client.call_tool(tool["name"], args)
                    response_parts.append(f"**{sname.title()}** → `{tool['name']}`: {result.get('result', 'No result')}")
                    protocol_details.append({
                        "server": sname,
                        "method": "tools/call",
                        "tool": tool["name"],
                        "arguments": args,
                        "result": result,
                    })

            resources = client.list_resources()
            for res in resources:
                res_words = res["name"].lower().split() + res["uri"].lower().split("/")
                if any(w in query_lower for w in res_words if len(w) > 3):
                    data = client.read_resource(res["uri"])
                    contents = data.get("contents", [{}])
                    text = contents[0].get("text", "") if contents else ""
                    response_parts.append(f"**{sname.title()}** → 📄 `{res['uri']}`:\n{text}")
                    protocol_details.append({
                        "server": sname,
                        "method": "resources/read",
                        "uri": res["uri"],
                        "content_preview": text[:200],
                    })

        if response_parts:
            full_response = "Here's what I found using MCP:\n\n" + "\n\n".join(response_parts)
        else:
            full_response = "I couldn't find a matching tool or resource for that query. Try asking about weather, database queries, or files."

        st.session_state.mcp_chat.append({
            "role": "assistant",
            "content": full_response,
            "protocol": protocol_details if protocol_details else None,
        })
        st.rerun()

with tab_log:
    st.subheader("Full Protocol Message Log")

    if not st.session_state.mcp_messages:
        st.info("No messages yet. Connect to servers and interact to see protocol traffic.")
    else:
        for i, msg in enumerate(reversed(st.session_state.mcp_messages)):
            ts = time.strftime("%H:%M:%S", time.localtime(msg["timestamp"]))
            if msg["type"] == "connection":
                st.success(f"**[{ts}]** Connected to **{msg['server']}**")
            else:
                with st.expander(f"[{ts}] `{msg.get('method', '?')}` → {msg['server']}"):
                    st.json(msg)

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/19_🛠️_MCP_Server_Builder.py", label="← Server Builder", icon="🛠️")
with cols[2]:
    st.page_link("pages/21_❓_MCP_Help.py", label="MCP Help →", icon="❓")
