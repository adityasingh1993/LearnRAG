"""
PromptCraft — Reusable UI Components
Custom Streamlit widgets for consistent, polished UI.
"""

import streamlit as st
from prompting_utils.scoring import get_score_color, get_score_label, get_verdict_emoji


def inject_custom_css():
    """Inject custom CSS into the Streamlit app."""
    import os
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", color: str = "#00E676"):
    """Display a styled metric card."""
    delta_html = ""
    if delta:
        delta_color = "#00E676" if not delta.startswith("-") else "#FF1744"
        delta_html = f'<div style="color:{delta_color}; font-size:0.85rem; margin-top:4px;">{delta}</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#aaa; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;">{label}</div>
        <div style="color:{color}; font-size:2rem; font-weight:700; margin:4px 0;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def score_badge(score: float, label: str = "Score"):
    """Display a circular score badge."""
    color = get_score_color(score)
    grade = get_score_label(score)
    st.markdown(f"""
    <div style="text-align:center; padding:16px;">
        <div style="
            width:100px; height:100px;
            border-radius:50%;
            border:4px solid {color};
            display:inline-flex;
            align-items:center;
            justify-content:center;
            flex-direction:column;
            background: rgba(255,255,255,0.03);
        ">
            <div style="font-size:1.8rem; font-weight:700; color:{color};">{int(score)}</div>
            <div style="font-size:0.65rem; color:#aaa;">{label}</div>
        </div>
        <div style="color:{color}; font-size:0.85rem; margin-top:8px; font-weight:600;">{grade}</div>
    </div>
    """, unsafe_allow_html=True)


def technique_card(icon: str, name: str, description: str, is_active: bool = False):
    """Display a technique info card."""
    border = "border:2px solid #7C4DFF;" if is_active else "border:1px solid #333;"
    st.markdown(f"""
    <div class="technique-card" style="{border}">
        <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
        <div style="font-size:1.1rem; font-weight:600; color:white; margin-bottom:6px;">{name}</div>
        <div style="font-size:0.85rem; color:#aaa; line-height:1.4;">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def comparison_header(label_a: str, label_b: str, icon_a: str = "🅰️", icon_b: str = "🅱️"):
    """Display a comparison header for arena mode."""
    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        st.markdown(f"""
        <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #1a1a2e, #16213e);
                    border-radius:12px; border:1px solid #7C4DFF50;">
            <span style="font-size:1.5rem;">{icon_a}</span>
            <div style="font-size:1.1rem; font-weight:600; color:white; margin-top:4px;">{label_a}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_vs:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center; height:100%;">
            <div style="font-size:1.3rem; font-weight:700; color:#7C4DFF; padding-top:20px;">VS</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #1a1a2e, #16213e);
                    border-radius:12px; border:1px solid #00E67650;">
            <span style="font-size:1.5rem;">{icon_b}</span>
            <div style="font-size:1.1rem; font-weight:600; color:white; margin-top:4px;">{label_b}</div>
        </div>
        """, unsafe_allow_html=True)


def claim_card(claim: str, verdict: str, reason: str, confidence: float):
    """Display a claim evaluation card."""
    emoji = get_verdict_emoji(verdict)
    bg_colors = {
        "supported": "rgba(0,230,118,0.08)",
        "uncertain": "rgba(255,214,0,0.08)",
        "unsupported": "rgba(255,23,68,0.08)",
    }
    border_colors = {
        "supported": "#00E676",
        "uncertain": "#FFD600",
        "unsupported": "#FF1744",
    }
    bg = bg_colors.get(verdict.lower(), "rgba(255,255,255,0.05)")
    border = border_colors.get(verdict.lower(), "#555")

    st.markdown(f"""
    <div style="background:{bg}; border-left:3px solid {border}; padding:12px 16px;
                border-radius:0 8px 8px 0; margin-bottom:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:0.95rem; color:white; flex:1;">{emoji} {claim}</div>
            <div style="font-size:0.75rem; color:#aaa; margin-left:12px; white-space:nowrap;">
                {int(confidence * 100)}% conf.
            </div>
        </div>
        <div style="font-size:0.8rem; color:#aaa; margin-top:6px; font-style:italic;">{reason}</div>
    </div>
    """, unsafe_allow_html=True)


def response_panel(response_text: str, label: str = "Response", metrics: dict = None):
    """Display a response panel with optional metrics."""
    st.markdown(f"""
    <div class="response-panel">
        <div style="font-size:0.8rem; color:#7C4DFF; text-transform:uppercase;
                    letter-spacing:1px; margin-bottom:8px; font-weight:600;">{label}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(response_text)

    if metrics:
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(label=key, value=value)


def api_key_warning():
    """Show a warning if API key is not configured."""
    st.warning(
        "⚠️ **API key not configured.** Please set your `OPENROUTER_API_KEY` in a `.env` file "
        "or enter it in the sidebar to start using PromptCraft.",
        icon="🔑"
    )
    with st.expander("How to get an API key"):
        st.markdown("""
        1. Go to [OpenRouter](https://openrouter.ai/keys)
        2. Sign up or log in
        3. Click **Create Key**
        4. Copy the key
        5. Create a `.env` file in the project root with:
           ```
           OPENROUTER_API_KEY=your_key_here
           ```
        6. Or paste it in the sidebar field
        """)


def page_header(title: str, subtitle: str, icon: str = ""):
    """Display a styled page header."""
    st.markdown(f"""
    <div style="padding:20px 0 10px 0;">
        <h1 style="margin:0; font-size:2.2rem;">
            {f'<span style="margin-right:12px;">{icon}</span>' if icon else ''}{title}
        </h1>
        <p style="color:#aaa; font-size:1rem; margin-top:4px;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
