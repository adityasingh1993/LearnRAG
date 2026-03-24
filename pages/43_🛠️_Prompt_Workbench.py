"""
PromptCraft — Prompt Workbench Page
Custom prompt editor with template variables and evaluation.
"""

import streamlit as st
from prompting_core import config
from prompting_core.prompt_templates import ALL_TEMPLATES, format_prompt
from prompting_core.llm_client import generate, is_api_key_configured
from prompting_core.evaluator import evaluate_quality
from prompting_core.scenarios import ALL_SCENARIOS
from prompting_utils.ui_components import (
    inject_custom_css, page_header, score_badge, response_panel,
    metric_card, api_key_warning,
)
from prompting_utils.scoring import get_score_color, format_latency

inject_custom_css()

page_header(
    title="Prompt Workbench",
    subtitle="Build your own prompts, test them, and compare against built-in techniques",
    icon="🛠️",
)

if not is_api_key_configured():
    api_key_warning()
    st.stop()

# ── Template Starter ────────────────────────────────────────────
st.markdown("### 📝 Compose Your Prompt")

col_starter, col_settings = st.columns([3, 1])

with col_starter:
    starter = st.selectbox(
        "Start from a template (optional):",
        options=["Blank"] + list(ALL_TEMPLATES.keys()),
        format_func=lambda x: "✨ Start from scratch" if x == "Blank"
                              else f"{ALL_TEMPLATES[x].icon} {ALL_TEMPLATES[x].name}",
    )

with col_settings:
    st.markdown("**Custom Settings**")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=config.DEFAULT_TEMPERATURE,
        step=0.1,
        help="Lower = more deterministic, Higher = more creative",
    )
    top_p = st.slider(
        "Top-P (Nucleus)",
        min_value=0.0,
        max_value=1.0,
        value=getattr(config, "DEFAULT_TOP_P", 1.0),
        step=0.05,
        help="0.1 = only consider top 10% mass, 1.0 = consider all tokens",
    )
    top_k = st.number_input(
        "Top-K",
        min_value=0,
        max_value=500,
        value=getattr(config, "DEFAULT_TOP_K", 0) or 0,
        help="0 to disable. Restricts to the top k most likely tokens.",
    )

    st.markdown("**Comparison Settings**")
    sync_params = st.checkbox("Sync with Custom", value=True)
    if sync_params:
        comp_temperature = temperature
        comp_top_p = top_p
        comp_top_k = top_k
    else:
        comp_temperature = st.slider(
            "Comp. Temp",
            min_value=0.0,
            max_value=2.0,
            value=config.DEFAULT_TEMPERATURE,
            step=0.1,
        )
        comp_top_p = st.slider(
            "Comp. Top-P",
            min_value=0.0,
            max_value=1.0,
            value=getattr(config, "DEFAULT_TOP_P", 1.0),
            step=0.05,
        )
        comp_top_k = st.number_input(
            "Comp. Top-K",
            min_value=0,
            max_value=500,
            value=getattr(config, "DEFAULT_TOP_K", 0) or 0,
        )

# Populate based on starter
if starter != "Blank":
    template = ALL_TEMPLATES[starter]
    default_system = template.system_prompt
    default_user = template.user_prompt_template
else:
    default_system = ""
    default_user = "{question}"

# ── Prompt Editor ───────────────────────────────────────────────
system_prompt = st.text_area(
    "System Prompt (optional):",
    value=default_system,
    height=100,
    placeholder="Define the AI's role, personality, and constraints...",
    help="This sets the context and behavior for the AI. Use this to assign roles or set rules.",
)

user_prompt = st.text_area(
    "User Prompt Template:",
    value=default_user,
    height=200,
    help="Use {question} as a placeholder for the actual question. You can also use {context} and {examples}.",
)

st.caption("💡 **Available variables:** `{question}` — your question will be inserted here")

st.divider()

# ── Question Input ──────────────────────────────────────────────
st.markdown("### 🎯 Test Your Prompt")

col_question, col_scenario = st.columns([3, 1])

with col_question:
    question = st.text_area(
        "Enter a question to test:",
        value="What causes the Northern Lights (Aurora Borealis)?",
        height=80,
    )

with col_scenario:
    st.markdown("**Quick-fill:**")
    scenario_options = [f"{s.title}" for s in ALL_SCENARIOS[:6]]
    selected_scenario = st.selectbox("Pick a scenario:", ["—"] + scenario_options)
    if selected_scenario != "—":
        idx = scenario_options.index(selected_scenario)
        question = ALL_SCENARIOS[idx].question

# ── Compare Option ──────────────────────────────────────────────
compare = st.checkbox("📊 Compare with a built-in technique", value=True)

if compare:
    compare_technique = st.selectbox(
        "Compare against:",
        options=list(ALL_TEMPLATES.keys()),
        format_func=lambda x: f"{ALL_TEMPLATES[x].icon} {ALL_TEMPLATES[x].name}",
        index=0,
    )

# ── Run ─────────────────────────────────────────────────────────
if st.button("🚀 Run Prompt", type="primary", use_container_width=True):
    # Format the custom prompt
    try:
        formatted_prompt = user_prompt.format(question=question)
    except KeyError as e:
        st.error(f"Template variable error: {e}. Make sure to use {{question}} as the placeholder.")
        st.stop()

    col_custom, col_compare = st.columns(2) if compare else (st.container(), None)

    with col_custom if compare else col_custom:
        with st.spinner("Running your custom prompt..."):
            custom_result = generate(
                prompt=formatted_prompt,
                system_prompt=system_prompt if system_prompt else None,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
            )
            custom_quality = evaluate_quality(question, custom_result["response"])

        st.markdown("#### 🛠️ Your Custom Prompt")
        response_panel(
            custom_result["response"],
            label="Custom Prompt Response",
            metrics={
                "Tokens": custom_result["tokens_used"],
                "Latency": format_latency(custom_result["latency_ms"]),
            }
        )
        score_badge(custom_quality.get("overall_score", 0), "Quality")

    if compare and col_compare:
        comp_template = ALL_TEMPLATES[compare_technique]

        with col_compare:
            with st.spinner(f"Running {comp_template.name}..."):
                sys_c, user_c = format_prompt(compare_technique, question)
                comp_result = generate(
                    prompt=user_c, 
                    system_prompt=sys_c if sys_c else None,
                    temperature=comp_temperature,
                    top_p=comp_top_p,
                    top_k=comp_top_k if comp_top_k > 0 else None,
                )
                comp_quality = evaluate_quality(question, comp_result["response"])

            st.markdown(f"#### {comp_template.icon} {comp_template.name}")
            response_panel(
                comp_result["response"],
                label=f"{comp_template.name} Response",
                metrics={
                    "Tokens": comp_result["tokens_used"],
                    "Latency": format_latency(comp_result["latency_ms"]),
                }
            )
            score_badge(comp_quality.get("overall_score", 0), "Quality")

    # Add to run history
    if "run_history" not in st.session_state:
        st.session_state.run_history = []

    st.session_state.run_history.append({
        "technique": "custom",
        "question": question,
        "response": custom_result["response"],
        "quality": custom_quality,
        "tokens": custom_result["tokens_used"],
        "latency_ms": custom_result["latency_ms"],
        "temperature": custom_result.get("temperature"),
        "top_p": custom_result.get("top_p"),
        "top_k": custom_result.get("top_k"),
        "page": "Workbench",
    })

    # Quality breakdown
    with st.expander("📋 Detailed Quality Scores"):
        dimensions = ["accuracy", "completeness", "clarity", "reasoning", "conciseness"]
        cols = st.columns(len(dimensions))
        for i, dim in enumerate(dimensions):
            with cols[i]:
                score = custom_quality.get(dim, 0)
                color = get_score_color(score)
                metric_card(dim.capitalize(), str(int(score)), color=color)

        if custom_quality.get("summary"):
            st.info(f"**Summary:** {custom_quality['summary']}")

    # Show formatted prompt
    with st.expander("👁️ View Formatted Prompt"):
        if system_prompt:
            st.markdown("**System Prompt:**")
            st.code(system_prompt, language="text")
        st.markdown("**User Prompt (formatted):**")
        st.code(formatted_prompt, language="text")
