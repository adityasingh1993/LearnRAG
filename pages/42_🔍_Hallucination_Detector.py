"""
PromptCraft — Hallucination Detector Page
Claim-by-claim analysis of LLM responses.
"""

import streamlit as st
from prompting_core.llm_client import generate, is_api_key_configured
from prompting_core.prompt_templates import ALL_TEMPLATES, format_prompt
from prompting_core.evaluator import evaluate_hallucination, extract_claims
from prompting_core.scenarios import ALL_SCENARIOS
from prompting_core import config
from prompting_utils.ui_components import (
    inject_custom_css, page_header, claim_card, score_badge,
    metric_card, api_key_warning,
)
from prompting_utils.scoring import get_score_color

inject_custom_css()

page_header(
    title="Hallucination Detector",
    subtitle="Analyze any LLM response claim-by-claim — find hallucinations before they cause harm",
    icon="🔍",
)

if not is_api_key_configured():
    api_key_warning()
    st.stop()

# ── Input Mode ──────────────────────────────────────────────────
st.markdown("### 📝 Input")

input_mode = st.radio(
    "How would you like to provide the response?",
    ["🤖 Generate a response first", "📋 Paste an existing response"],
    horizontal=True,
)

if input_mode == "🤖 Generate a response first":
    col_q, col_t = st.columns([3, 1])

    with col_q:
        # Question input
        scenario = st.selectbox(
            "Pick a scenario or type your own:",
            options=["Custom..."] + [f"{s.title} — {s.question[:60]}..." for s in ALL_SCENARIOS],
        )

        if scenario == "Custom...":
            question = st.text_area(
                "Enter a question:",
                value="What are the main health benefits of drinking green tea?",
                height=80,
            )
        else:
            idx = [f"{s.title} — {s.question[:60]}..." for s in ALL_SCENARIOS].index(scenario)
            question = ALL_SCENARIOS[idx].question
            st.markdown(f"**Question:** {question}")

    with col_t:
        technique = st.selectbox(
            "Prompting technique:",
            options=list(ALL_TEMPLATES.keys()),
            format_func=lambda x: f"{ALL_TEMPLATES[x].icon} {ALL_TEMPLATES[x].name}",
            index=0,  # Default to zero-shot for hallucination testing
        )

    context = st.text_area(
        "Reference context (optional — provide ground truth for more accurate detection):",
        placeholder="Paste any reference context, source material, or known facts here...",
        height=100,
    )

    if st.button("🚀 Generate & Analyze", type="primary", use_container_width=True):
        with st.spinner("Generating response..."):
            sys_prompt, user_prompt = format_prompt(technique, question)
            result = generate(user_prompt, sys_prompt if sys_prompt else None)
            response_text = result["response"]
            st.session_state.hallucination_response = response_text
            st.session_state.hallucination_context = context
            st.session_state.hallucination_question = question

        with st.spinner("🔍 Analyzing for hallucinations (this may take a moment)..."):
            eval_result = evaluate_hallucination(response_text, context)
            st.session_state.hallucination_eval = eval_result

else:
    response_text = st.text_area(
        "Paste the LLM response to analyze:",
        height=200,
        placeholder="Paste any LLM response here to check it for hallucinations...",
    )

    context = st.text_area(
        "Reference context (optional):",
        placeholder="Paste any reference context, source material, or known facts here...",
        height=100,
    )

    if st.button("🔍 Analyze for Hallucinations", type="primary", use_container_width=True):
        if not response_text.strip():
            st.warning("Please paste a response to analyze.")
        else:
            st.session_state.hallucination_response = response_text
            st.session_state.hallucination_context = context
            st.session_state.hallucination_question = ""

            with st.spinner("🔍 Analyzing claims (this may take a moment)..."):
                eval_result = evaluate_hallucination(response_text, context)
                st.session_state.hallucination_eval = eval_result


# ── Display Results ─────────────────────────────────────────────
if "hallucination_eval" in st.session_state:
    eval_result = st.session_state.hallucination_eval
    response_text = st.session_state.get("hallucination_response", "")

    st.divider()
    st.markdown("### 📊 Analysis Results")

    # Score overview
    col_faith, col_hall, col_claims = st.columns(3)

    with col_faith:
        faith_score = eval_result["faithfulness_score"]
        score_badge(faith_score, "Faithfulness")

    with col_hall:
        hall_score = eval_result["hallucination_score"]
        inv_score = 100 - hall_score  # Invert for display (higher = better = less hallucination)
        color = get_score_color(inv_score)
        metric_card(
            "Hallucination Risk",
            f"{hall_score}%",
            color=color,
        )

    with col_claims:
        total_claims = len(eval_result["claims"])
        metric_card(
            "Claims Analyzed",
            str(total_claims),
            color="#448AFF",
        )

    # Original response
    with st.expander("📄 Original Response", expanded=False):
        st.markdown(response_text)

    # Claim-by-claim breakdown
    st.divider()
    st.markdown("### 🔬 Claim-by-Claim Breakdown")

    evaluations = eval_result.get("evaluations", [])

    # Summary counts
    supported = sum(1 for e in evaluations if e.get("verdict") == "supported")
    uncertain = sum(1 for e in evaluations if e.get("verdict") == "uncertain")
    unsupported = sum(1 for e in evaluations if e.get("verdict") == "unsupported")

    sum_col1, sum_col2, sum_col3 = st.columns(3)
    with sum_col1:
        st.markdown(f'<div style="text-align:center; color:#00E676; font-size:1.5rem; font-weight:700;">✅ {supported}</div>', unsafe_allow_html=True)
        st.caption("Supported")
    with sum_col2:
        st.markdown(f'<div style="text-align:center; color:#FFD600; font-size:1.5rem; font-weight:700;">⚠️ {uncertain}</div>', unsafe_allow_html=True)
        st.caption("Uncertain")
    with sum_col3:
        st.markdown(f'<div style="text-align:center; color:#FF1744; font-size:1.5rem; font-weight:700;">❌ {unsupported}</div>', unsafe_allow_html=True)
        st.caption("Unsupported")

    st.write("")

    # Individual claim cards
    for evaluation in evaluations:
        claim_card(
            claim=evaluation.get("claim", ""),
            verdict=evaluation.get("verdict", "uncertain"),
            reason=evaluation.get("reason", ""),
            confidence=evaluation.get("confidence", 0.5),
        )

    # Recommendations
    if unsupported > 0 or uncertain > 0:
        st.divider()
        st.markdown("### 💡 Recommended Improvements")

        recommendations = []
        if unsupported > 0:
            recommendations.append(
                "🔗 **Chain-of-Verification (CoVe)** — Would have caught these false claims "
                "by forcing the model to verify each statement before including it."
            )
        if uncertain > 0:
            recommendations.append(
                "🔄 **Self-Consistency** — Running the prompt multiple times and comparing "
                "answers would have flagged inconsistent claims."
            )
        if unsupported > 0 or uncertain > 0:
            recommendations.append(
                "📐 **Structured Output** — Constraining the output format reduces room "
                "for speculative elaboration and fabricated details."
            )
            recommendations.append(
                "📝 **Few-Shot** — Providing example answers with verified facts sets a "
                "standard the model tries to match."
            )

        for rec in recommendations:
            st.info(rec)
