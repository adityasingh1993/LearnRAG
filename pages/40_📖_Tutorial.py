"""
PromptCraft — Tutorial Page
Guided walkthrough of 8 prompting levels.
"""

import streamlit as st
from prompting_core import config
from prompting_core.prompt_templates import ALL_TEMPLATES, format_prompt
from prompting_core.llm_client import generate, generate_multiple, is_api_key_configured
from prompting_core.evaluator import evaluate_quality
from prompting_core.scenarios import get_recommended_scenario
from prompting_utils.ui_components import (
    inject_custom_css, page_header, metric_card, score_badge,
    technique_card, response_panel, api_key_warning, render_prompting_settings,
)
from prompting_utils.scoring import get_score_color, improvement_percentage, format_latency

inject_custom_css()
render_prompting_settings()

page_header(
    title="Guided Tutorial",
    subtitle="Walk through 8 prompting levels — see how each technique improves LLM responses",
    icon="📖",
)

if not is_api_key_configured():
    api_key_warning()
    st.stop()

# ── Session State ───────────────────────────────────────────────
if "tutorial_level" not in st.session_state:
    st.session_state.tutorial_level = 0
if "tutorial_results" not in st.session_state:
    st.session_state.tutorial_results = {}
if "baseline_score" not in st.session_state:
    st.session_state.baseline_score = None

# ── Level Selector ──────────────────────────────────────────────
st.markdown("### Select a Level")

cols = st.columns(8)
for i, key in enumerate(config.TECHNIQUE_ORDER):
    template = ALL_TEMPLATES[key]
    with cols[i]:
        completed = key in st.session_state.tutorial_results
        btn_label = f"{template.icon}\n{'✓' if completed else i+1}"
        if st.button(
            btn_label,
            key=f"level_btn_{key}",
            use_container_width=True,
            type="primary" if i == st.session_state.tutorial_level else "secondary",
        ):
            st.session_state.tutorial_level = i
            st.rerun()

st.divider()

# ── Current Level Content ───────────────────────────────────────
current_key = config.TECHNIQUE_ORDER[st.session_state.tutorial_level]
current_template = ALL_TEMPLATES[current_key]
level_num = st.session_state.tutorial_level + 1

# Level Header
st.markdown(f"""
<div style="padding:16px 20px; background:linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius:12px; border:1px solid #7C4DFF30; margin-bottom:20px;">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-size:3rem;">{current_template.icon}</div>
        <div>
            <div style="color:#7C4DFF; font-size:0.8rem; text-transform:uppercase;
                        letter-spacing:2px; font-weight:600;">Level {level_num}</div>
            <div style="font-size:1.5rem; font-weight:700; color:white;">{current_template.name}</div>
            <div style="color:#aaa; font-size:0.9rem; margin-top:4px;">{current_template.description}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Explanation
col_explain, col_why = st.columns(2)

with col_explain:
    st.markdown("#### 🔧 How It Works")
    st.info(current_template.how_it_works)

with col_why:
    st.markdown("#### 💡 Why It Helps")
    st.success(current_template.why_it_helps)

st.markdown(f"**🎯 Expected Improvement:** {current_template.example_boost}")

st.divider()

# ── Demo Scenario ───────────────────────────────────────────────
st.markdown("### 🎮 Live Demo")

# Get the recommended scenario for this technique
recommended = get_recommended_scenario(current_key)

# Let user choose between recommended scenario or custom question
demo_mode = st.radio(
    "Choose your input:",
    ["📚 Recommended Scenario", "✏️ Custom Question"],
    horizontal=True,
    key=f"demo_mode_{current_key}",
)

if demo_mode == "📚 Recommended Scenario" and recommended:
    st.markdown(f"""
    <div style="padding:12px 16px; background:rgba(124,77,255,0.08); border-left:3px solid #7C4DFF;
                border-radius:0 8px 8px 0; margin:8px 0;">
        <div style="font-weight:600; color:white;">{recommended.title}</div>
        <div style="color:#aaa; font-size:0.85rem; margin-top:4px;">
            Category: {recommended.category} · Difficulty: {recommended.difficulty}
        </div>
        <div style="color:#bbb; font-size:0.85rem; margin-top:4px; font-style:italic;">
            {recommended.why_interesting}
        </div>
    </div>
    """, unsafe_allow_html=True)
    question = recommended.question
    st.markdown(f"**Question:** {question}")
else:
    question = st.text_area(
        "Enter your question:",
        value="What are the 5 largest lakes in Africa by surface area?",
        height=80,
        key=f"custom_q_{current_key}",
    )

# ── Run the Prompt ──────────────────────────────────────────────
col_run, col_compare = st.columns(2)

with col_run:
    run_current = st.button(
        f"🚀 Run {current_template.name}",
        key=f"run_{current_key}",
        use_container_width=True,
        type="primary",
    )

with col_compare:
    run_baseline = st.button(
        "📊 Run Baseline (Zero-Shot) for Comparison",
        key=f"run_baseline_{current_key}",
        use_container_width=True,
        disabled=(current_key == "zero_shot"),
    )

if run_current or run_baseline:
    # Retain existing results so we can show side-by-side comparisons
    results = st.session_state.tutorial_results.get(current_key, {}).copy()
    just_run = []

    # Run baseline if requested or if this IS the baseline
    if run_baseline or current_key == "zero_shot":
        with st.spinner("Running Zero-Shot baseline..."):
            sys_prompt, user_prompt = format_prompt("zero_shot", question)
            baseline_result = generate(
                prompt=user_prompt,
                system_prompt=sys_prompt if sys_prompt else None,
                temperature=ALL_TEMPLATES["zero_shot"].recommended_temperature,
            )
            baseline_quality = evaluate_quality(question, baseline_result["response"])
            results["baseline"] = {**baseline_result, "quality": baseline_quality}
            st.session_state.baseline_score = baseline_quality.get("overall_score", 50)
            just_run.append("baseline")

    # Run current technique
    if run_current:
        if current_key == "self_consistency":
            with st.spinner(f"Running {current_template.name} ({config.SELF_CONSISTENCY_RUNS} passes)..."):
                sys_prompt, user_prompt = format_prompt(current_key, question)
                multi_results = generate_multiple(
                    prompt=user_prompt,
                    system_prompt=sys_prompt if sys_prompt else None,
                    n=config.SELF_CONSISTENCY_RUNS,
                )
                # Use the first response as primary, but show all
                current_result = multi_results[0]
                current_result["all_responses"] = [r["response"] for r in multi_results]
                current_quality = evaluate_quality(question, current_result["response"])
                results["current"] = {**current_result, "quality": current_quality}
                just_run.append("current")
        else:
            with st.spinner(f"Running {current_template.name}..."):
                sys_prompt, user_prompt = format_prompt(current_key, question)
                current_result = generate(
                    prompt=user_prompt,
                    system_prompt=sys_prompt if sys_prompt else None,
                    temperature=current_template.recommended_temperature,
                )
                current_quality = evaluate_quality(question, current_result["response"])
                results["current"] = {**current_result, "quality": current_quality}
                just_run.append("current")

    # Store results
    st.session_state.tutorial_results[current_key] = results

    # Add to history
    if "run_history" not in st.session_state:
        st.session_state.run_history = []

    for run_key in just_run:
        run_data = results[run_key]
        st.session_state.run_history.append({
            "technique": current_key if run_key == "current" else "zero_shot",
            "question": question,
            "response": run_data["response"],
            "quality": run_data["quality"],
            "tokens": run_data["tokens_used"],
            "latency_ms": run_data["latency_ms"],
            "temperature": run_data.get("temperature"),
            "top_p": run_data.get("top_p"),
            "top_k": run_data.get("top_k"),
            "page": "Tutorial",
        })

# ── Display Results ─────────────────────────────────────────────
if current_key in st.session_state.tutorial_results:
    results = st.session_state.tutorial_results[current_key]

    st.divider()
    st.markdown("### 📊 Results")

    if "baseline" in results and "current" in results and current_key != "zero_shot":
        # Side-by-side comparison
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🎯 Zero-Shot Baseline")
            baseline = results["baseline"]
            response_panel(
                baseline["response"],
                label="Baseline Response",
                metrics={
                    "Tokens": baseline["tokens_used"],
                    "Latency": format_latency(baseline["latency_ms"]),
                }
            )
            score_badge(baseline["quality"].get("overall_score", 0), "Quality")

        with col_b:
            st.markdown(f"#### {current_template.icon} {current_template.name}")
            current = results["current"]
            response_panel(
                current["response"],
                label=f"{current_template.name} Response",
                metrics={
                    "Tokens": current["tokens_used"],
                    "Latency": format_latency(current["latency_ms"]),
                }
            )
            score_badge(current["quality"].get("overall_score", 0), "Quality")

            # Show improvement
            baseline_score = baseline["quality"].get("overall_score", 0)
            current_score = current["quality"].get("overall_score", 0)
            improvement = improvement_percentage(baseline_score, current_score)

            if improvement > 0:
                st.success(f"📈 **+{improvement}% improvement** over baseline!")
            elif improvement < 0:
                st.warning(f"📉 {improvement}% vs baseline")
            else:
                st.info("➡️ Similar to baseline")

            # Show all responses for self-consistency
            if "all_responses" in current:
                with st.expander("🔄 All Self-Consistency Responses"):
                    for j, resp in enumerate(current["all_responses"]):
                        st.markdown(f"**Pass {j+1}:**")
                        st.markdown(resp)
                        st.divider()

    elif "current" in results:
        # Single result (baseline or only current)
        current = results["current"]
        response_panel(
            current["response"],
            label=f"{current_template.name} Response",
            metrics={
                "Tokens": current["tokens_used"],
                "Latency": format_latency(current["latency_ms"]),
            }
        )
        score_badge(current["quality"].get("overall_score", 0), "Quality")

    elif "baseline" in results:
        baseline = results["baseline"]
        response_panel(
            baseline["response"],
            label="Zero-Shot Response",
            metrics={
                "Tokens": baseline["tokens_used"],
                "Latency": format_latency(baseline["latency_ms"]),
            }
        )
        score_badge(baseline["quality"].get("overall_score", 0), "Quality")

    # Quality Breakdown
    with st.expander("📋 Detailed Quality Scores"):
        result_to_show = results.get("current", results.get("baseline", {}))
        quality = result_to_show.get("quality", {})

        dimensions = ["accuracy", "completeness", "clarity", "reasoning", "conciseness"]
        cols = st.columns(len(dimensions))
        for i, dim in enumerate(dimensions):
            with cols[i]:
                score = quality.get(dim, 0)
                color = get_score_color(score)
                metric_card(dim.capitalize(), str(int(score)), color=color)

        if quality.get("summary"):
            st.markdown(f"**Summary:** {quality['summary']}")

    # Prompt Preview
    with st.expander("👁️ View the Actual Prompt Sent"):
        sys_prompt, user_prompt = format_prompt(current_key, question)
        if sys_prompt:
            st.markdown("**System Prompt:**")
            st.code(sys_prompt, language="text")
        st.markdown("**User Prompt:**")
        st.code(user_prompt, language="text")


# ── Navigation ──────────────────────────────────────────────────
st.divider()
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

with nav_col1:
    if st.session_state.tutorial_level > 0:
        if st.button("⬅️ Previous Level", use_container_width=True):
            st.session_state.tutorial_level -= 1
            st.rerun()

with nav_col3:
    if st.session_state.tutorial_level < len(config.TECHNIQUE_ORDER) - 1:
        if st.button("Next Level ➡️", use_container_width=True, type="primary"):
            st.session_state.tutorial_level += 1
            st.rerun()
    else:
        st.success("🎉 You've completed all levels!")
