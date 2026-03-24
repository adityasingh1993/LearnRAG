"""
PromptCraft — Prompt Arena Page
Side-by-side comparison of two prompting techniques.
"""

import streamlit as st
from prompting_core import config
from prompting_core.prompt_templates import ALL_TEMPLATES, format_prompt
from prompting_core.llm_client import generate, generate_multiple, is_api_key_configured
from prompting_core.evaluator import evaluate_quality, evaluate_hallucination
from prompting_core.scenarios import ALL_SCENARIOS
from prompting_utils.ui_components import (
    inject_custom_css, page_header, comparison_header, score_badge,
    response_panel, metric_card, api_key_warning, render_prompting_settings,
)
from prompting_utils.scoring import (
    get_score_color, get_score_label, format_latency,
    composite_score, improvement_percentage
)

inject_custom_css()
render_prompting_settings()

page_header(
    title="Prompt Arena",
    subtitle="Pick two techniques, same question — see which one wins",
    icon="🏟️",
)

if not is_api_key_configured():
    api_key_warning()
    st.stop()

# ── Setup ───────────────────────────────────────────────────────
technique_names = {key: f"{t.icon} {t.name}" for key, t in ALL_TEMPLATES.items()}
technique_keys = list(ALL_TEMPLATES.keys())

col_a_sel, col_b_sel = st.columns(2)

with col_a_sel:
    technique_a = st.selectbox(
        "🅰️ Technique A",
        options=technique_keys,
        format_func=lambda x: technique_names[x],
        index=0,
        key="arena_tech_a",
    )

with col_b_sel:
    default_b = min(3, len(technique_keys) - 1)
    technique_b = st.selectbox(
        "🅱️ Technique B",
        options=technique_keys,
        format_func=lambda x: technique_names[x],
        index=default_b,
        key="arena_tech_b",
    )

st.divider()

# ── Question Input ──────────────────────────────────────────────
input_mode = st.radio(
    "Choose input method:",
    ["📚 Pick from Scenarios", "✏️ Custom Question"],
    horizontal=True,
)

if input_mode == "📚 Pick from Scenarios":
    scenario = st.selectbox(
        "Select a scenario:",
        options=ALL_SCENARIOS,
        format_func=lambda s: f"{s.category} • {s.title} ({s.difficulty})",
    )
    question = scenario.question
    st.markdown(f"**Question:** {question}")
    st.caption(f"💡 {scenario.why_interesting}")
else:
    question = st.text_area(
        "Enter your question:",
        value="Explain the difference between TCP and UDP protocols.",
        height=80,
    )

# ── Run Arena ───────────────────────────────────────────────────
run_arena = st.button(
    "⚔️ Battle!",
    use_container_width=True,
    type="primary",
    disabled=(technique_a == technique_b),
)

if technique_a == technique_b:
    st.warning("Please select two **different** techniques to compare.")

if run_arena and technique_a != technique_b:
    template_a = ALL_TEMPLATES[technique_a]
    template_b = ALL_TEMPLATES[technique_b]

    comparison_header(
        template_a.name, template_b.name,
        template_a.icon, template_b.icon,
    )

    col_res_a, col_res_b = st.columns(2)

    # Run both techniques
    with col_res_a:
        with st.spinner(f"Running {template_a.name}..."):
            sys_a, user_a = format_prompt(technique_a, question)
            if technique_a == "self_consistency":
                multi = generate_multiple(user_a, sys_a if sys_a else None, n=config.SELF_CONSISTENCY_RUNS)
                result_a = multi[0]
                result_a["all_responses"] = [r["response"] for r in multi]
            else:
                result_a = generate(user_a, sys_a if sys_a else None)

    with col_res_b:
        with st.spinner(f"Running {template_b.name}..."):
            sys_b, user_b = format_prompt(technique_b, question)
            if technique_b == "self_consistency":
                multi = generate_multiple(user_b, sys_b if sys_b else None, n=config.SELF_CONSISTENCY_RUNS)
                result_b = multi[0]
                result_b["all_responses"] = [r["response"] for r in multi]
            else:
                result_b = generate(user_b, sys_b if sys_b else None)

    # Evaluate both
    with st.spinner("🧪 Evaluating responses..."):
        quality_a = evaluate_quality(question, result_a["response"])
        quality_b = evaluate_quality(question, result_b["response"])

    # Store in session
    st.session_state[f"arena_result_a"] = {**result_a, "quality": quality_a, "technique": technique_a}
    st.session_state[f"arena_result_b"] = {**result_b, "quality": quality_b, "technique": technique_b}

    # Add to run history
    if "run_history" not in st.session_state:
        st.session_state.run_history = []

    for tech, res, qual in [(technique_a, result_a, quality_a), (technique_b, result_b, quality_b)]:
        st.session_state.run_history.append({
            "technique": tech,
            "question": question,
            "response": res["response"],
            "quality": qual,
            "tokens": res["tokens_used"],
            "latency_ms": res["latency_ms"],
            "temperature": res.get("temperature"),
            "top_p": res.get("top_p"),
            "top_k": res.get("top_k"),
            "page": "Arena",
        })

# ── Display Results ─────────────────────────────────────────────
if "arena_result_a" in st.session_state and "arena_result_b" in st.session_state:
    result_a = st.session_state.arena_result_a
    result_b = st.session_state.arena_result_b
    template_a = ALL_TEMPLATES[result_a["technique"]]
    template_b = ALL_TEMPLATES[result_b["technique"]]

    st.divider()

    # Determine winner
    score_a = result_a["quality"].get("overall_score", 0)
    score_b = result_b["quality"].get("overall_score", 0)

    if score_a > score_b:
        winner_text = f"🏆 {template_a.icon} {template_a.name} wins!"
        winner_color = "#7C4DFF"
    elif score_b > score_a:
        winner_text = f"🏆 {template_b.icon} {template_b.name} wins!"
        winner_color = "#00E676"
    else:
        winner_text = "🤝 It's a tie!"
        winner_color = "#FFD600"

    st.markdown(f"""
    <div style="text-align:center; padding:16px; background:linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius:12px; border:1px solid {winner_color}50; margin-bottom:20px;">
        <div style="font-size:1.5rem; font-weight:700; color:{winner_color};">{winner_text}</div>
        <div style="color:#aaa; font-size:0.85rem; margin-top:4px;">
            Score: {score_a} vs {score_b}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Side-by-side responses
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"#### {template_a.icon} {template_a.name}")
        response_panel(
            result_a["response"],
            label=f"{template_a.name} Response",
            metrics={
                "Tokens": result_a["tokens_used"],
                "Latency": format_latency(result_a["latency_ms"]),
            }
        )
        score_badge(score_a, "Quality")

    with col_b:
        st.markdown(f"#### {template_b.icon} {template_b.name}")
        response_panel(
            result_b["response"],
            label=f"{template_b.name} Response",
            metrics={
                "Tokens": result_b["tokens_used"],
                "Latency": format_latency(result_b["latency_ms"]),
            }
        )
        score_badge(score_b, "Quality")

    # Detailed comparison
    st.divider()
    st.markdown("### 📋 Dimension-by-Dimension Comparison")

    dimensions = ["accuracy", "completeness", "clarity", "reasoning", "conciseness"]
    for dim in dimensions:
        val_a = result_a["quality"].get(dim, 0)
        val_b = result_b["quality"].get(dim, 0)
        diff = val_a - val_b

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.markdown(f"**{dim.capitalize()}**")
        with col2:
            # Visual bar comparison
            st.markdown(f"""
            <div style="display:flex; gap:8px; align-items:center;">
                <div style="flex:1; text-align:right; color:#7C4DFF; font-weight:600;">{val_a}</div>
                <div style="width:200px; height:8px; background:#1a1a2e; border-radius:4px; overflow:hidden; display:flex;">
                    <div style="width:{val_a}%; background:#7C4DFF; border-radius:4px;"></div>
                </div>
                <div style="width:200px; height:8px; background:#1a1a2e; border-radius:4px; overflow:hidden; display:flex;">
                    <div style="width:{val_b}%; background:#00E676; border-radius:4px;"></div>
                </div>
                <div style="flex:1; color:#00E676; font-weight:600;">{val_b}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            if diff > 0:
                st.markdown(f'<span style="color:#7C4DFF;">A +{diff}</span>', unsafe_allow_html=True)
            elif diff < 0:
                st.markdown(f'<span style="color:#00E676;">B +{abs(diff)}</span>', unsafe_allow_html=True)
            else:
                st.markdown('Tied')

    # View prompts
    with st.expander("👁️ View Prompts Sent"):
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            sys_a, user_a = format_prompt(result_a["technique"], question)
            st.markdown(f"**{template_a.name} Prompt:**")
            if sys_a:
                st.code(sys_a, language="text")
            st.code(user_a, language="text")
        with p_col2:
            sys_b, user_b = format_prompt(result_b["technique"], question)
            st.markdown(f"**{template_b.name} Prompt:**")
            if sys_b:
                st.code(sys_b, language="text")
            st.code(user_b, language="text")
