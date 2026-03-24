"""
Visualization utilities: dimensionality reduction, plotting, and heatmaps.
All plots use Plotly for interactivity and a consistent dark theme.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# ─── Consistent dark theme for all Plotly charts ───
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#FAFAFA"),
    margin=dict(l=40, r=40, t=50, b=40),
)

CATEGORY_COLORS = {
    "Animals": "#FF6B6B",
    "Technology": "#6C63FF",
    "Food": "#4ECB71",
    "Sports": "#FFB347",
    "Text 1 Only": "#FF6B6B",
    "Text 2 Only": "#6C63FF",
    "Both Texts": "#4ECB71",
}


# ═══════════════════════════════════════════════════════════
# DIMENSIONALITY REDUCTION
# ═══════════════════════════════════════════════════════════

def reduce_dimensions(embeddings: np.ndarray, method: str = "pca", n_components: int = 2):
    """
    Reduce embedding dimensions for visualization.
    method: 'pca' or 'tsne'
    """
    if embeddings.shape[0] < 2:
        return embeddings[:, :n_components] if embeddings.shape[1] >= n_components else embeddings

    if method == "tsne":
        perplexity = min(30, max(2, embeddings.shape[0] - 1))
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
        )
    else:
        reducer = PCA(n_components=min(n_components, embeddings.shape[1], embeddings.shape[0]))

    reduced = reducer.fit_transform(embeddings)
    return reduced


# ═══════════════════════════════════════════════════════════
# SCATTER PLOTS
# ═══════════════════════════════════════════════════════════

def plot_embeddings_2d(
    reduced: np.ndarray,
    labels: list[str],
    title: str = "Embedding Space",
    color_labels: list[str] = None,
    show_text: bool = True,
):
    """Interactive 2D scatter plot of embeddings."""
    if color_labels is None:
        color_labels = labels

    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        text=labels if show_text else None,
        color=color_labels,
        color_discrete_map=CATEGORY_COLORS,
        title=title,
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(size=12, line=dict(width=1, color="#FAFAFA")),
        textfont=dict(size=10),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        showlegend=True,
        height=500,
    )
    return fig


def plot_embeddings_3d(
    reduced: np.ndarray,
    labels: list[str],
    title: str = "3D Embedding Space",
    color_labels: list[str] = None,
):
    """Interactive 3D scatter plot of embeddings."""
    if color_labels is None:
        color_labels = labels

    fig = px.scatter_3d(
        x=reduced[:, 0],
        y=reduced[:, 1],
        z=reduced[:, 2],
        text=labels,
        color=color_labels,
        color_discrete_map=CATEGORY_COLORS,
        title=title,
    )
    fig.update_traces(
        marker=dict(size=6, line=dict(width=0.5, color="#FAFAFA")),
        textfont=dict(size=8),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=600,
    )
    return fig


# ═══════════════════════════════════════════════════════════
# HEATMAPS
# ═══════════════════════════════════════════════════════════

def plot_similarity_heatmap(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "Cosine Similarity Matrix",
):
    """Plot a cosine similarity heatmap."""
    sim_matrix = cosine_similarity(embeddings)

    # Truncate labels for readability
    short_labels = [l[:30] + "..." if len(l) > 30 else l for l in labels]

    fig = go.Figure(
        data=go.Heatmap(
            z=sim_matrix,
            x=short_labels,
            y=short_labels,
            colorscale=[
                [0, "#1a1d23"],
                [0.5, "#6C63FF"],
                [1, "#B794F6"],
            ],
            text=np.round(sim_matrix, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=500,
        xaxis=dict(tickangle=45),
    )
    return fig


def plot_document_term_matrix(
    matrix: np.ndarray,
    features: list[str],
    doc_labels: list[str],
    title: str = "Document-Term Matrix",
):
    """Plot a document-term matrix as a heatmap."""
    short_docs = [f"Doc {i+1}: {l[:25]}..." if len(l) > 25 else f"Doc {i+1}: {l}" for i, l in enumerate(doc_labels)]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=list(features),
            y=short_docs,
            colorscale=[
                [0, "#1a1d23"],
                [0.3, "#2d1f4e"],
                [0.6, "#6C63FF"],
                [1, "#B794F6"],
            ],
            text=np.round(matrix, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="<b>%{y}</b><br>Term: %{x}<br>Value: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=max(300, len(doc_labels) * 50 + 100),
        xaxis=dict(tickangle=45),
    )
    return fig


def plot_embedding_vector(
    vector: np.ndarray,
    title: str = "Embedding Vector",
    feature_names: list[str] = None,
    max_features: int = 50,
):
    """Bar chart showing raw vector values."""
    if len(vector) > max_features:
        # Show top features by absolute value
        top_idx = np.argsort(np.abs(vector))[-max_features:]
        vector = vector[top_idx]
        if feature_names is not None:
            feature_names = [feature_names[i] for i in top_idx]

    if feature_names is None:
        feature_names = [f"dim_{i}" for i in range(len(vector))]

    colors = ["#6C63FF" if v >= 0 else "#FF6B6B" for v in vector]

    fig = go.Figure(
        data=go.Bar(
            x=feature_names,
            y=vector,
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=350,
        xaxis=dict(tickangle=45),
        yaxis_title="Value",
    )
    return fig


def plot_comparison_bars(
    bow_vector: np.ndarray,
    tfidf_vector: np.ndarray,
    feature_names: list[str],
    title: str = "BoW vs TF-IDF Weights",
):
    """Side-by-side bar chart comparing BoW and TF-IDF weights."""
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Bar(
            name="Bag of Words",
            x=list(feature_names),
            y=bow_vector,
            marker_color="rgba(255, 107, 107, 0.7)",
        )
    )
    fig.add_trace(
        go.Bar(
            name="TF-IDF",
            x=list(feature_names),
            y=tfidf_vector,
            marker_color="rgba(108, 99, 255, 0.7)",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        barmode="group",
        height=400,
        xaxis=dict(tickangle=45),
        yaxis_title="Weight",
    )
    return fig


def plot_word_clusters(
    words: list[str],
    embeddings: np.ndarray,
    title: str = "Word Clusters",
    method: str = "pca",
):
    """Plot word embeddings as a 2D scatter with word labels."""
    reduced = reduce_dimensions(embeddings, method=method, n_components=2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode="markers+text",
            text=words,
            textposition="top center",
            marker=dict(
                size=12,
                color="#6C63FF",
                line=dict(width=1, color="#B794F6"),
            ),
            textfont=dict(size=11, color="#FAFAFA"),
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=500,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
    )
    return fig


def plot_analogy_result(
    words: list[str],
    scores: list[float],
    title: str = "Analogy Results",
):
    """Horizontal bar chart for analogy/similarity results."""
    fig = go.Figure(
        data=go.Bar(
            x=scores,
            y=words,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale=[[0, "#2d1f4e"], [1, "#6C63FF"]],
                line=dict(width=1, color="#B794F6"),
            ),
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
            textfont=dict(color="#FAFAFA"),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=max(250, len(words) * 40 + 100),
        xaxis_title="Similarity Score",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_sparsity_comparison(data: dict, title: str = "Vector Properties Comparison"):
    """
    Radar/bar chart comparing embedding properties.
    data: dict with method names as keys and dicts of properties as values.
    """
    methods = list(data.keys())
    properties = list(data[methods[0]].keys())

    fig = go.Figure()
    colors = ["#FF6B6B", "#6C63FF", "#4ECB71", "#FFB347", "#FF69B4", "#00CED1"]

    for i, method in enumerate(methods):
        fig.add_trace(
            go.Bar(
                name=method,
                x=properties,
                y=[data[method][p] for p in properties],
                marker_color=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        barmode="group",
        height=400,
    )
    return fig
