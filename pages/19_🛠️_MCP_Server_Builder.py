"""Page 19 — MCP Server Builder: Interactively build a custom MCP server."""

import streamlit as st
import json
from core.mcp_simulator import (
    MCPServer, MCPClient, MCPResource, MCPTool, MCPToolParam, MCPPrompt,
)

st.set_page_config(page_title="MCP Server Builder", page_icon="🛠️", layout="wide")

st.title("🛠️ MCP Server Builder")
st.markdown("""
Build your own MCP server step-by-step. Add resources, tools, and prompts,
then test them with a simulated client.
""")

# ── Session State ───────────────────────────────────────────────────────
if "custom_server" not in st.session_state:
    st.session_state.custom_server = {
        "name": "my-server",
        "version": "1.0.0",
        "resources": [],
        "tools": [],
        "prompts": [],
    }

cfg = st.session_state.custom_server

# ── Server Config ───────────────────────────────────────────────────────
st.header("1 · Server Configuration")

c1, c2 = st.columns(2)
cfg["name"] = c1.text_input("Server Name", value=cfg["name"], key="sb_name")
cfg["version"] = c2.text_input("Version", value=cfg["version"], key="sb_ver")

# ── Add Resources ───────────────────────────────────────────────────────
st.header("2 · Add Resources")

with st.expander("➕ Add a new Resource", expanded=not cfg["resources"]):
    rc1, rc2 = st.columns(2)
    r_uri = rc1.text_input("URI", value="data://example", key="r_uri")
    r_name = rc2.text_input("Name", value="Example Data", key="r_name")
    r_desc = st.text_input("Description", value="An example resource", key="r_desc")
    r_mime = st.selectbox("MIME Type", ["text/plain", "application/json", "text/markdown"], key="r_mime")
    r_content = st.text_area("Content", value="Hello from my resource!", key="r_content", height=100)

    if st.button("Add Resource", key="add_res"):
        cfg["resources"].append({
            "uri": r_uri, "name": r_name, "description": r_desc,
            "mime_type": r_mime, "content": r_content,
        })
        st.rerun()

if cfg["resources"]:
    st.markdown(f"**{len(cfg['resources'])} resources defined:**")
    for i, res in enumerate(cfg["resources"]):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"📄 `{res['uri']}` — {res['name']}")
        if c2.button("🗑️", key=f"del_res_{i}"):
            cfg["resources"].pop(i)
            st.rerun()

# ── Add Tools ───────────────────────────────────────────────────────────
st.header("3 · Add Tools")

with st.expander("➕ Add a new Tool", expanded=not cfg["tools"]):
    t_name = st.text_input("Tool Name", value="my_tool", key="t_name")
    t_desc = st.text_input("Tool Description", value="Does something useful", key="t_desc")

    st.markdown("**Parameters:**")
    num_params = st.number_input("Number of parameters", 1, 5, 1, key="t_nparams")
    params_def = []
    for p in range(int(num_params)):
        pc1, pc2, pc3 = st.columns(3)
        pname = pc1.text_input(f"Param {p+1} name", value=f"param{p+1}", key=f"tp_name_{p}")
        ptype = pc2.selectbox(f"Param {p+1} type", ["string", "number", "boolean"], key=f"tp_type_{p}")
        pdesc = pc3.text_input(f"Param {p+1} description", value="", key=f"tp_desc_{p}")
        params_def.append({"name": pname, "type": ptype, "description": pdesc})

    t_code = st.text_area(
        "Handler code (Python — receives params as kwargs, return a string):",
        value='return f"Processed: {param1}"',
        key="t_code", height=100,
    )

    if st.button("Add Tool", key="add_tool"):
        cfg["tools"].append({
            "name": t_name, "description": t_desc,
            "parameters": params_def, "code": t_code,
        })
        st.rerun()

if cfg["tools"]:
    st.markdown(f"**{len(cfg['tools'])} tools defined:**")
    for i, tool in enumerate(cfg["tools"]):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"🔧 `{tool['name']}` — {tool['description']} ({len(tool['parameters'])} params)")
        if c2.button("🗑️", key=f"del_tool_{i}"):
            cfg["tools"].pop(i)
            st.rerun()

# ── Add Prompts ─────────────────────────────────────────────────────────
st.header("4 · Add Prompts")

with st.expander("➕ Add a new Prompt", expanded=not cfg["prompts"]):
    p_name = st.text_input("Prompt Name", value="my_prompt", key="p_name")
    p_desc = st.text_input("Prompt Description", value="A custom prompt", key="p_desc")
    p_template = st.text_area(
        "Template (use {arg_name} for placeholders):",
        value="Help the user with {topic}. Be concise and helpful.",
        key="p_template", height=100,
    )
    p_args_str = st.text_input("Arguments (comma-separated names):", value="topic", key="p_args_str")

    if st.button("Add Prompt", key="add_prompt"):
        args_list = [{"name": a.strip(), "description": f"Value for {a.strip()}", "required": True}
                     for a in p_args_str.split(",") if a.strip()]
        cfg["prompts"].append({
            "name": p_name, "description": p_desc,
            "template": p_template, "arguments": args_list,
        })
        st.rerun()

if cfg["prompts"]:
    st.markdown(f"**{len(cfg['prompts'])} prompts defined:**")
    for i, prompt in enumerate(cfg["prompts"]):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"💬 `{prompt['name']}` — {prompt['description']}")
        if c2.button("🗑️", key=f"del_prompt_{i}"):
            cfg["prompts"].pop(i)
            st.rerun()

# ── Build & Test ────────────────────────────────────────────────────────
st.header("5 · Build & Test Your Server")

total = len(cfg["resources"]) + len(cfg["tools"]) + len(cfg["prompts"])

if total == 0:
    st.info("Add at least one resource, tool, or prompt above, then build your server.")
else:
    if st.button("🚀 Build Server & Connect", key="build_server", type="primary"):
        server = MCPServer(cfg["name"], cfg["version"])

        for res in cfg["resources"]:
            server.add_resource(MCPResource(
                uri=res["uri"], name=res["name"], description=res["description"],
                mime_type=res["mime_type"], content=res["content"],
            ))

        for tool_def in cfg["tools"]:
            params = [MCPToolParam(p["name"], p["type"], p["description"]) for p in tool_def["parameters"]]
            code = tool_def["code"]
            try:
                param_names = [p["name"] for p in tool_def["parameters"]]
                func_code = f"def _handler({', '.join(param_names)}):\n"
                for line in code.split("\n"):
                    func_code += f"    {line}\n"
                local_ns: dict = {}
                exec(func_code, {}, local_ns)
                handler = local_ns["_handler"]
            except Exception:
                handler = lambda **kw: f"Handler error — raw code: {code}"
            server.add_tool(MCPTool(
                name=tool_def["name"], description=tool_def["description"],
                parameters=params, handler=handler,
            ))

        for prompt_def in cfg["prompts"]:
            server.add_prompt(MCPPrompt(
                name=prompt_def["name"], description=prompt_def["description"],
                arguments=prompt_def["arguments"], template=prompt_def["template"],
            ))

        client = MCPClient()
        conn_result = client.connect(server)

        st.session_state["built_server"] = server
        st.session_state["built_client"] = client

        st.success(f"Server **{server.name}** built and connected!")
        st.json(conn_result.get("result", {}))

    if "built_client" in st.session_state:
        st.subheader("Test Your Server")

        client = st.session_state["built_client"]
        server = st.session_state["built_server"]

        test_tab1, test_tab2, test_tab3 = st.tabs(["Resources", "Tools", "Prompts"])

        with test_tab1:
            resources = client.list_resources()
            for res in resources:
                if st.button(f"Read {res['uri']}", key=f"test_read_{res['uri']}"):
                    data = client.read_resource(res["uri"])
                    st.json(data)

        with test_tab2:
            tools = client.list_tools()
            for tool in tools:
                with st.expander(f"🔧 {tool['name']}"):
                    args = {}
                    for pname in tool.get("inputSchema", {}).get("properties", {}):
                        args[pname] = st.text_input(f"{pname}:", key=f"test_tool_{tool['name']}_{pname}")
                    if st.button(f"Call {tool['name']}", key=f"test_call_{tool['name']}"):
                        result = client.call_tool(tool["name"], {k: v for k, v in args.items() if v})
                        st.json(result)

        with test_tab3:
            prompts = client.list_prompts()
            for prompt in prompts:
                with st.expander(f"💬 {prompt['name']}"):
                    args = {}
                    for arg in prompt.get("arguments", []):
                        args[arg["name"]] = st.text_input(f"{arg['name']}:", key=f"test_prompt_{prompt['name']}_{arg['name']}")
                    if st.button(f"Render {prompt['name']}", key=f"test_render_{prompt['name']}"):
                        result = client.get_prompt(prompt["name"], {k: v for k, v in args.items() if v})
                        st.json(result)

# ── Generated Code ──────────────────────────────────────────────────────
st.header("6 · Generated Server Code")
st.markdown("Here's what your server would look like as Python code:")

code_lines = [
    "from mcp.server.fastmcp import FastMCP",
    "",
    f'mcp = FastMCP("{cfg["name"]}")',
    "",
]

for res in cfg["resources"]:
    code_lines.append(f'@mcp.resource("{res["uri"]}")')
    code_lines.append(f'def {res["name"].lower().replace(" ", "_")}() -> str:')
    code_lines.append(f'    """{ res["description"] }"""')
    code_lines.append(f'    return """{res["content"]}"""')
    code_lines.append("")

for tool_def in cfg["tools"]:
    param_str = ", ".join(f'{p["name"]}: {p["type"]}' for p in tool_def["parameters"])
    code_lines.append(f'@mcp.tool()')
    code_lines.append(f'def {tool_def["name"]}({param_str}) -> str:')
    code_lines.append(f'    """{tool_def["description"]}"""')
    for line in tool_def["code"].split("\n"):
        code_lines.append(f'    {line}')
    code_lines.append("")

for prompt_def in cfg["prompts"]:
    arg_names = [a["name"] for a in prompt_def["arguments"]]
    code_lines.append(f'@mcp.prompt()')
    code_lines.append(f'def {prompt_def["name"]}({", ".join(arg_names)}) -> str:')
    code_lines.append(f'    """{prompt_def["description"]}"""')
    code_lines.append(f'    return f"""{prompt_def["template"]}"""')
    code_lines.append("")

st.code("\n".join(code_lines), language="python")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/18_🧱_MCP_Primitives.py", label="← MCP Primitives", icon="🧱")
with cols[2]:
    st.page_link("pages/20_🎮_MCP_Playground.py", label="MCP Playground →", icon="🎮")
