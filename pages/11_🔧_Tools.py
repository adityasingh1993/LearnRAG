"""
Module 11: Tools & Function Calling
Explore built-in tools, test them interactively, build custom tools, and learn how function calling works.
"""

import streamlit as st

st.set_page_config(page_title="Tools & Function Calling | RAG Lab", page_icon="🔧", layout="wide")

from components.sidebar import render_provider_config, get_llm_provider
from core.tools import (
    BUILTIN_TOOLS,
    Tool,
    ToolParameter,
    create_custom_tool,
    create_tool_registry,
)

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
    .tool-card {
        background: linear-gradient(145deg, #1a1d29, #22263a);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .tool-card .tool-name {
        color: #00CC96;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .tool-card .tool-category {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #6C63FF22;
        color: #6C63FF;
        margin-left: 8px;
    }
    .param-chip {
        display: inline-block;
        background: #2a2d3a;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 3px 10px;
        margin: 3px 4px 3px 0;
        font-size: 0.85rem;
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔧 Tools & Function Calling")
st.markdown("*Learn how agents use tools, try them out, and build your own.*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 1 — What Are Tools?
# ═══════════════════════════════════════════════════════════════════════════
st.header("What Are Tools?")

st.markdown("""
<div class="concept-box">
<h4>Function Calling / Tool Use</h4>
Tools are <strong>functions</strong> that an LLM can call to interact with the outside world.
The LLM doesn't run the functions directly — instead, it outputs a structured request
(tool name + arguments), the system executes the function, and the result is fed back
to the LLM as an <strong>observation</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("""
Each tool is defined by:
- **Name** — a unique identifier (e.g. `calculator`)
- **Description** — what the tool does (the LLM reads this to decide when to use it)
- **Parameters** — typed inputs the tool expects (name, type, description, required/optional)
- **Function** — the actual code that runs when the tool is called
""")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 — Built-in Tools Gallery
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Built-in Tools Gallery")
st.markdown("Explore all available tools. Each card shows the tool's description and parameters.")

SAMPLE_INPUTS: dict[str, dict[str, str]] = {
    "calculator": {"expression": "sqrt(144) + 3.14 * 2"},
    "datetime": {"format": "%A, %B %d, %Y at %H:%M"},
    "text_stats": {"text": "The quick brown fox jumps over the lazy dog. It was a sunny day."},
    "web_search": {"query": "What is RAG in AI?"},
    "json_parse": {"data": '{"name": "Alice", "score": 95, "tags": ["ai", "ml"]}'},
    "unit_convert": {"value": "100", "from_unit": "celsius", "to_unit": "fahrenheit"},
    "string_transform": {"text": "hello world from rag lab", "operation": "title"},
}

for tool_name, tool in BUILTIN_TOOLS.items():
    with st.expander(f"**{tool.name}** — {tool.description}"):
        cols = st.columns([3, 1])
        with cols[0]:
            param_html = ""
            for p in tool.parameters:
                req_label = "required" if p.required else "optional"
                param_html += (
                    f'<span class="param-chip">'
                    f"<code>{p.name}</code>: {p.type} ({req_label}) — {p.description}"
                    f"</span> "
                )
            st.markdown(f"**Parameters:** {param_html}", unsafe_allow_html=True)
            st.markdown(f"**Category:** `{tool.category}`")
            st.code(tool.schema_for_prompt(), language="text")
        with cols[1]:
            sample = SAMPLE_INPUTS.get(tool_name, {})
            if st.button(f"▶ Try it", key=f"try_{tool_name}"):
                with st.spinner("Running..."):
                    result = tool.run(**sample)
                st.code(result, language="text")
            if sample:
                st.caption(f"Sample: `{sample}`")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 3 — Interactive Tool Tester
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Interactive Tool Tester")
st.markdown("Select any tool, provide inputs, and see the results live.")

selected_tool_name = st.selectbox(
    "Choose a tool",
    options=list(BUILTIN_TOOLS.keys()),
    format_func=lambda x: f"{x} — {BUILTIN_TOOLS[x].description[:60]}",
)

selected_tool = BUILTIN_TOOLS[selected_tool_name]
st.markdown(f"**{selected_tool.description}**")

tester_inputs: dict[str, str] = {}
param_cols = st.columns(min(len(selected_tool.parameters), 3) or 1)
for i, param in enumerate(selected_tool.parameters):
    col_idx = i % len(param_cols)
    with param_cols[col_idx]:
        default = SAMPLE_INPUTS.get(selected_tool_name, {}).get(param.name, "")
        tester_inputs[param.name] = st.text_input(
            f"{param.name} ({param.type})",
            value=default,
            key=f"tester_{selected_tool_name}_{param.name}",
            help=param.description,
        )

if st.button("🚀 Run Tool", key="run_tool_tester", type="primary"):
    with st.spinner("Executing tool..."):
        result = selected_tool.run(**tester_inputs)
    st.success("Result:")
    st.code(result, language="text")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 4 — Custom Tool Builder
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Custom Tool Builder")
st.markdown(
    "Define your own tool with a name, description, parameters, and Python code. "
    "The code must set a `result` variable."
)

col_def, col_code = st.columns([1, 1])

with col_def:
    custom_name = st.text_input("Tool name", value="my_tool", key="custom_tool_name")
    custom_desc = st.text_input(
        "Description", value="A custom tool that does something useful", key="custom_tool_desc"
    )
    st.markdown("**Parameter**")
    c1, c2 = st.columns(2)
    with c1:
        param_name = st.text_input("Parameter name", value="input_text", key="custom_param_name")
    with c2:
        param_type = st.selectbox("Type", ["string", "number", "boolean"], key="custom_param_type")

with col_code:
    st.markdown("**Python code** (use parameter names as variables; set `result`)")
    custom_code = st.text_area(
        "Code",
        value='# Example: reverse the input and count characters\nreversed_text = input_text[::-1]\nresult = f"Reversed: {reversed_text} (length: {len(input_text)})"',
        height=160,
        key="custom_tool_code",
    )

col_test_input, col_test_run = st.columns([2, 1])
with col_test_input:
    custom_test_value = st.text_input(
        f"Test value for '{param_name}'", value="Hello Agents!", key="custom_test_val"
    )
with col_test_run:
    st.markdown("")
    if st.button("🛠️ Create & Test", key="create_custom_tool", type="primary"):
        try:
            new_tool = create_custom_tool(
                name=custom_name,
                description=custom_desc,
                parameters=[{"name": param_name, "type": param_type, "description": f"User param: {param_name}"}],
                code=custom_code,
            )
            test_result = new_tool.run(**{param_name: custom_test_value})
            st.success(f"Tool `{custom_name}` created successfully!")
            st.code(test_result, language="text")
        except Exception as e:
            st.error(f"Error creating tool: {e}")

# ═══════════════════════════════════════════════════════════════════════════
#  Section 5 — How Function Calling Works
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("How Function Calling Works")
st.markdown(
    "When an agent has access to tools, the system prompt includes tool descriptions "
    "in a structured format. Here's what the LLM actually sees:"
)

registry = create_tool_registry()
st.markdown("**Tool descriptions sent to the LLM:**")
st.code(
    f"Available tools:\n{registry.format_for_prompt()}",
    language="text",
)

st.markdown("**The Thought / Action / Observation cycle:**")
st.code(
    """User Question: What is 25 * 17 + sqrt(144)?

Thought: I need to calculate this mathematical expression.
         Let me use the calculator tool.
Action: calculator
Action Input: {"expression": "25 * 17 + sqrt(144)"}

Observation: 437.0

Thought: I have the answer from the calculator.
Action: finish
Action Input: {"answer": "25 × 17 + √144 = 437.0"}""",
    language="text",
)

st.info(
    "💡 The LLM outputs **structured text** (Action + Action Input), not actual function calls. "
    "The agent framework **parses** this text, runs the tool, and feeds the observation back. "
    "This text-based protocol is how tools work with any LLM."
)

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/10_🤖_Agent_Basics.py", label="← Agent Basics", icon="🤖")
with col2:
    st.page_link("pages/12_🔄_Agent_Patterns.py", label="Next: Agent Patterns →", icon="🔄")
