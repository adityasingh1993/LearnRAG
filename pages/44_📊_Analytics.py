"""
PromptCraft — Analytics Dashboard Page
Performance tracking, radar charts, leaderboard, and run history.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from prompting_core import config
from prompting_core.prompt_templates import ALL_TEMPLATES
from prompting_utils.ui_components import inject_custom_css, page_header, metric_card
from prompting_utils.scoring import get_score_color

inject_custom_css()

page_header(
    title="Analytics Dashboard",
    subtitle="Track performance across techniques — charts, leaderboards, and exportable history",
    icon="📊",
)

# ── Session Data ────────────────────────────────────────────────
if "run_history" not in st.session_state:
    st.session_state.run_history = []

history = st.session_state.run_history

if not history:
    st.info(
        "📭 **No data yet.** Run some prompts in the Tutorial, Arena, or Workbench "
        "pages and come back here to see your analytics!"
    )
    st.stop()

# Build DataFrame
df = pd.DataFrame(history)
if "temperature" not in df.columns:
    df["temperature"] = None
if "top_p" not in df.columns:
    df["top_p"] = None
if "top_k" not in df.columns:
    df["top_k"] = None
df["overall_score"] = df["quality"].apply(lambda q: q.get("overall_score", 0) if isinstance(q, dict) else 0)
df["accuracy"] = df["quality"].apply(lambda q: q.get("accuracy", 0) if isinstance(q, dict) else 0)
df["completeness"] = df["quality"].apply(lambda q: q.get("completeness", 0) if isinstance(q, dict) else 0)
df["clarity"] = df["quality"].apply(lambda q: q.get("clarity", 0) if isinstance(q, dict) else 0)
df["reasoning"] = df["quality"].apply(lambda q: q.get("reasoning", 0) if isinstance(q, dict) else 0)
df["conciseness"] = df["quality"].apply(lambda q: q.get("conciseness", 0) if isinstance(q, dict) else 0)
df["technique_name"] = df["technique"].apply(
    lambda t: f"{ALL_TEMPLATES[t].icon} {ALL_TEMPLATES[t].name}" if t in ALL_TEMPLATES else "🛠️ Custom"
)

# ── Overview Metrics ────────────────────────────────────────────
st.markdown("### 📈 Session Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Total Runs", str(len(df)), color="#7C4DFF")

with col2:
    avg_score = round(df["overall_score"].mean(), 1)
    metric_card("Avg Quality", str(avg_score), color=get_score_color(avg_score))

with col3:
    total_tokens = df["tokens"].sum()
    metric_card("Total Tokens", f"{total_tokens:,}", color="#448AFF")

with col4:
    avg_latency = round(df["latency_ms"].mean())
    metric_card("Avg Latency", f"{avg_latency}ms", color="#00BCD4")

st.divider()

# ── Technique Leaderboard ──────────────────────────────────────
st.markdown("### 🏆 Technique Leaderboard")

# Group by technique and calculate averages
leaderboard = df.groupby("technique_name").agg(
    avg_score=("overall_score", "mean"),
    runs=("overall_score", "count"),
    avg_tokens=("tokens", "mean"),
    avg_latency=("latency_ms", "mean"),
    avg_accuracy=("accuracy", "mean"),
    avg_reasoning=("reasoning", "mean"),
).round(1).sort_values("avg_score", ascending=False).reset_index()

# Display as a styled table
for i, row in leaderboard.iterrows():
    rank = i + 1
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
    color = get_score_color(row["avg_score"])

    st.markdown(f"""
    <div style="display:flex; align-items:center; padding:12px 16px;
                background:linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius:8px; margin-bottom:6px; border:1px solid #33333380;">
        <div style="font-size:1.5rem; width:40px; text-align:center;">{medal}</div>
        <div style="flex:1; margin-left:12px;">
            <div style="font-weight:600; color:white; font-size:1rem;">{row["technique_name"]}</div>
            <div style="color:#aaa; font-size:0.8rem;">{int(row["runs"])} runs · avg {int(row["avg_tokens"])} tokens · {int(row["avg_latency"])}ms</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:1.5rem; font-weight:700; color:{color};">{row["avg_score"]:.0f}</div>
            <div style="font-size:0.7rem; color:#aaa;">avg score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Radar Chart ─────────────────────────────────────────────────
st.markdown("### 🕸️ Technique Comparison (Radar Chart)")

# Get average scores per dimension per technique
dimensions = ["accuracy", "completeness", "clarity", "reasoning", "conciseness"]

# Let user select which techniques to compare
available_techniques = df["technique_name"].unique().tolist()

if len(available_techniques) >= 2:
    selected_techniques = st.multiselect(
        "Select techniques to compare:",
        options=available_techniques,
        default=available_techniques[:min(3, len(available_techniques))],
    )
else:
    selected_techniques = available_techniques

if selected_techniques:
    fig = go.Figure()

    colors = ["#7C4DFF", "#00E676", "#FF9100", "#448AFF", "#FF1744", "#FFD600", "#00BCD4", "#E040FB"]

    for i, tech in enumerate(selected_techniques):
        tech_df = df[df["technique_name"] == tech]
        values = [tech_df[dim].mean() for dim in dimensions]
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=dimensions + [dimensions[0]],
            fill='toself',
            name=tech,
            line=dict(color=colors[i % len(colors)]),
            fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)},{int(colors[i % len(colors)][3:5], 16)},{int(colors[i % len(colors)][5:7], 16)},0.1)",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa"),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Score Distribution ──────────────────────────────────────────
st.markdown("### 📊 Score Distribution")

fig_dist = px.box(
    df,
    x="technique_name",
    y="overall_score",
    color="technique_name",
    labels={"technique_name": "Technique", "overall_score": "Quality Score"},
)

fig_dist.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#aaa"),
    showlegend=False,
    xaxis=dict(gridcolor="rgba(51, 51, 51, 0.3)"),
    yaxis=dict(gridcolor="rgba(51, 51, 51, 0.3)", range=[0, 100]),
    height=400,
)

st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# ── Run History ─────────────────────────────────────────────────
st.markdown("### 📜 Run History")

# Filter options
col_f1, col_f2 = st.columns(2)

with col_f1:
    filter_technique = st.multiselect(
        "Filter by technique:",
        options=df["technique_name"].unique().tolist(),
        default=df["technique_name"].unique().tolist(),
    )

with col_f2:
    filter_page = st.multiselect(
        "Filter by page:",
        options=df["page"].unique().tolist(),
        default=df["page"].unique().tolist(),
    )

filtered_df = df[
    (df["technique_name"].isin(filter_technique)) &
    (df["page"].isin(filter_page))
]

# Display table
display_df = filtered_df[["technique_name", "question", "overall_score", "tokens", "latency_ms", "temperature", "top_p", "top_k", "page"]].copy()
display_df.columns = ["Technique", "Question", "Score", "Tokens", "Latency (ms)", "Temp", "Top-P", "Top-K", "Page"]
display_df["Question"] = display_df["Question"].str[:80] + "..."

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Score",
            min_value=0,
            max_value=100,
            format="%d",
        ),
    },
)

# Export
if st.button("📥 Export as CSV"):
    csv = filtered_df[["technique", "question", "response", "overall_score", "tokens", "latency_ms", "temperature", "top_p", "top_k", "page"]].to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="promptcraft_history.csv",
        mime="text/csv",
    )

# ── Clear Data ──────────────────────────────────────────────────
st.divider()
with st.expander("🗑️ Clear Session Data"):
    st.warning("This will clear all run history for this session.")
    if st.button("Clear All Data", type="secondary"):
        st.session_state.run_history = []
        st.session_state.tutorial_results = {}
        if "arena_result_a" in st.session_state:
            del st.session_state.arena_result_a
        if "arena_result_b" in st.session_state:
            del st.session_state.arena_result_b
        st.rerun()
