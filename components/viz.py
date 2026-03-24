"""
Visualization components for the RAG educational app.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def plot_embeddings_2d(embeddings: np.ndarray, labels: list[str], query_embedding: np.ndarray | None = None,
                       query_label: str = "Query", highlight_indices: list[int] | None = None):
    """Plot embeddings in 2D using PCA, optionally highlighting retrieved chunks."""
    from sklearn.decomposition import PCA

    all_emb = embeddings
    if query_embedding is not None:
        all_emb = np.vstack([embeddings, query_embedding.reshape(1, -1)])

    n_components = min(2, all_emb.shape[0], all_emb.shape[1])
    if n_components < 2:
        st.warning("Not enough data points for 2D visualization.")
        return

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_emb)

    short_labels = [l[:50] + "..." if len(l) > 50 else l for l in labels]

    colors = []
    for i in range(len(labels)):
        if highlight_indices and i in highlight_indices:
            colors.append("Retrieved")
        else:
            colors.append("Document")
    sizes = [10 if c == "Document" else 16 for c in colors]

    fig = go.Figure()
    for category, color, marker in [("Document", "#636EFA", "circle"), ("Retrieved", "#00CC96", "star")]:
        mask = [c == category for c in colors]
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]
            fig.add_trace(go.Scatter(
                x=reduced[idx, 0], y=reduced[idx, 1],
                mode="markers+text",
                marker=dict(size=[sizes[i] for i in idx], color=color, opacity=0.8),
                text=[short_labels[i] for i in idx],
                textposition="top center",
                textfont=dict(size=9),
                name=category,
            ))

    if query_embedding is not None:
        fig.add_trace(go.Scatter(
            x=[reduced[-1, 0]], y=[reduced[-1, 1]],
            mode="markers+text",
            marker=dict(size=20, color="#EF553B", symbol="diamond"),
            text=[query_label],
            textposition="top center",
            name="Query",
        ))

    fig.update_layout(
        title="Embedding Space (PCA 2D Projection)",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        height=500,
        template="plotly_dark",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_embeddings_3d(embeddings: np.ndarray, labels: list[str], query_embedding: np.ndarray | None = None):
    """Plot embeddings in 3D using PCA."""
    from sklearn.decomposition import PCA

    all_emb = embeddings
    if query_embedding is not None:
        all_emb = np.vstack([embeddings, query_embedding.reshape(1, -1)])

    n_components = min(3, all_emb.shape[0], all_emb.shape[1])
    if n_components < 3:
        st.warning("Not enough data for 3D visualization.")
        return

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(all_emb)

    short_labels = [l[:40] + "..." if len(l) > 40 else l for l in labels]
    types = ["Document"] * len(labels) + (["Query"] if query_embedding is not None else [])
    all_labels = short_labels + (["Query"] if query_embedding is not None else [])

    fig = px.scatter_3d(
        x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
        color=types, text=all_labels,
        labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Type"},
        title="Embedding Space (3D PCA Projection)",
        template="plotly_dark",
    )
    fig.update_layout(height=600)
    fig.update_traces(marker=dict(size=6))
    st.plotly_chart(fig, use_container_width=True)


def plot_similarity_heatmap(texts: list[str], embeddings: np.ndarray):
    """Plot a similarity heatmap between all document embeddings."""
    norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    sim_matrix = norms @ norms.T

    short_labels = [t[:30] + "..." if len(t) > 30 else t for t in texts]

    fig = px.imshow(
        sim_matrix,
        x=short_labels, y=short_labels,
        color_continuous_scale="Viridis",
        title="Cosine Similarity Heatmap",
        template="plotly_dark",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def plot_retrieval_scores(results: list, query: str):
    """Bar chart of retrieval scores for search results."""
    if not results:
        st.info("No results to display.")
        return

    labels = [r.text[:60] + "..." if len(r.text) > 60 else r.text for r in results]
    scores = [r.score for r in results]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=["#6C63FF" if s > 0.5 else "#FF6B6B" for s in scores],
        text=[f"{s:.3f}" for s in scores],
        textposition="auto",
    ))
    fig.update_layout(
        title=f'Retrieval Scores for: "{query[:50]}..."' if len(query) > 50 else f'Retrieval Scores for: "{query}"',
        xaxis_title="Similarity Score",
        yaxis_title="",
        height=max(300, len(results) * 60),
        template="plotly_dark",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pipeline_flow(steps: list, active_step: int | None = None):
    """Render a visual pipeline flow diagram using HTML/CSS."""
    step_configs = [
        ("📄", "Documents", "#4ECDC4"),
        ("✂️", "Chunking", "#45B7D1"),
        ("🔢", "Embedding", "#96CEB4"),
        ("📦", "Vector Store", "#FFEAA7"),
        ("🔍", "Retrieval", "#DDA0DD"),
        ("🤖", "Generation", "#98D8C8"),
    ]

    html_parts = ['<div style="display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:8px;padding:20px 0;">']
    for i, (icon, label, color) in enumerate(step_configs):
        is_active = active_step == i if active_step is not None else False
        border = f"3px solid {color}" if is_active else "2px solid #333"
        bg = f"{color}22" if is_active else "#1A1D29"
        shadow = f"0 0 15px {color}66" if is_active else "none"

        html_parts.append(f'''
            <div style="text-align:center;padding:12px 16px;border-radius:12px;border:{border};
                        background:{bg};min-width:90px;box-shadow:{shadow};">
                <div style="font-size:24px;">{icon}</div>
                <div style="font-size:12px;color:#ccc;margin-top:4px;">{label}</div>
            </div>
        ''')
        if i < len(step_configs) - 1:
            html_parts.append(f'<div style="font-size:20px;color:#555;">→</div>')

    html_parts.append('</div>')
    st.html("".join(html_parts))


def render_step_metrics(steps: list):
    """Render timing metrics for pipeline steps."""
    if not steps:
        return

    cols = st.columns(min(len(steps), 6))
    for i, step in enumerate(steps):
        with cols[i % len(cols)]:
            tokens = step.details.get("total_tokens") or step.details.get("tokens", 0)
            tok_str = f" | {tokens:,} tok" if tokens else ""
            st.metric(
                label=step.name,
                value=f"{step.duration_ms:.0f}ms{tok_str}",
            )


def render_token_usage(turn_usage, label: str = "This turn"):
    """Render a compact token usage breakdown for a single query turn."""
    if turn_usage is None:
        return

    cols = st.columns(4)
    with cols[0]:
        st.metric("Prompt tokens", f"{turn_usage.prompt_tokens:,}")
    with cols[1]:
        st.metric("Completion tokens", f"{turn_usage.completion_tokens:,}")
    with cols[2]:
        st.metric("Embedding tokens", f"{turn_usage.embedding_tokens:,}")
    with cols[3]:
        st.metric(f"Total ({label})", f"{turn_usage.total_tokens:,}")


def render_session_token_summary(tracker, model_name: str = "gpt-4o-mini"):
    """Render a session-level token summary with cost estimate."""
    if tracker is None or tracker.turn_count == 0:
        return

    from core.token_tracker import get_model_pricing
    pricing = get_model_pricing(model_name)

    cost_info = tracker.estimate_cost(
        prompt_price_per_1k=pricing.get("prompt", 0.00015),
        completion_price_per_1k=pricing.get("completion", 0.0006),
        embedding_price_per_1k=pricing.get("embedding", 0.00002),
    )

    st.markdown("---")

    mcols = st.columns(5)
    with mcols[0]:
        st.metric("Turns", tracker.turn_count)
    with mcols[1]:
        st.metric("Prompt", f"{cost_info['prompt_tokens']:,}")
    with mcols[2]:
        st.metric("Completion", f"{cost_info['completion_tokens']:,}")
    with mcols[3]:
        st.metric("Session total", f"{cost_info['total_tokens']:,}")
    with mcols[4]:
        total_cost = cost_info["total_cost"]
        if total_cost < 0.01:
            st.metric("Est. cost", f"${total_cost:.5f}")
        else:
            st.metric("Est. cost", f"${total_cost:.4f}")

    # Per-step breakdown chart
    if tracker.turn_count > 1:
        turn_data = []
        for idx, turn in enumerate(tracker.turns):
            turn_data.append({
                "Turn": idx + 1,
                "Prompt": turn.prompt_tokens,
                "Completion": turn.completion_tokens,
                "Embedding": turn.embedding_tokens,
            })

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Prompt", x=[d["Turn"] for d in turn_data],
                             y=[d["Prompt"] for d in turn_data], marker_color="#6C63FF"))
        fig.add_trace(go.Bar(name="Completion", x=[d["Turn"] for d in turn_data],
                             y=[d["Completion"] for d in turn_data], marker_color="#00CC96"))
        fig.add_trace(go.Bar(name="Embedding", x=[d["Turn"] for d in turn_data],
                             y=[d["Embedding"] for d in turn_data], marker_color="#FFA726"))
        fig.update_layout(
            barmode="stack", title="Tokens per Turn",
            xaxis_title="Turn", yaxis_title="Tokens",
            height=300, template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
