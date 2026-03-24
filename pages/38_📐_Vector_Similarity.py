"""
📐 Vector Similarity
Explains Cosine Similarity vs Dot Product vs Euclidean Distance.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from embedding_utils.common import inject_custom_css, page_header, metric_card
from embedding_utils.visualization import PLOTLY_LAYOUT

inject_custom_css()

page_header(
    "Vector Similarity Math", 
    "📐", 
    "How Vectors Compare", 
    "Once we have embeddings, how do we actually know if two texts are mathematically similar? Let's explore the three core methods!"
)

st.markdown("""
### 📏 The Three Pillars of Similarity
In AI, translating words to numbers is only half the battle. We also need a fast and accurate way to calculate the 'distance' between those numbers. 
These are the three foundational formulas powering all modern AI search algorithms:
""")

tab1, tab2, tab3 = st.tabs(["1️⃣ Euclidean Dst. (Straight Line)", "2️⃣ Dot Product (Scale & Shadow)", "🏆 3️⃣ Cosine Similarity (The Gold Standard)"])

with tab1:
    st.markdown("#### Euclidean Distance (L2 norm)")
    st.write("This is the standard 'ruler' distance you learned in geometry. It calculates the physical straight space between two points.")
    st.info("**Formula:** $\\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + ...}$")
    st.warning("⚠️ **Why it's rarely used for Text:** If you write a 10,000-word essay about dogs, and a 5-word sentence about dogs, they conceptually point in the same direction. However, the essay's vector will be *much mathematically longer*. Euclidean distance will mistakenly claim they are completely unrelated just because their physical lengths are different!")

with tab2:
    st.markdown("#### Dot Product")
    st.write("Calculates how much one vector 'projects' onto another, multiplying their magnitudes (lengths) together.")
    st.info("**Formula:** $A \\cdot B = (a_1 \\times b_1) + (a_2 \\times b_2) + ...$")
    st.success("✅ **Used for speed:** Highly optimized vector databases (like FAISS or Pinecone) sometimes use Dot Product instead of Cosine because it is computationally much faster to multiply numbers without having to calculate massive square roots to divide them.")
    
with tab3:
    st.markdown("#### Cosine Similarity")
    st.write("Instead of measuring the physical distance between points, **Cosine Similarity** calculates the *angle* between the two vectors. It perfectly ignores vector length!")
    st.info("**Formula:** $\\cos(\\theta) = \\frac{A \\cdot B}{||A|| \\times ||B||}$")
    st.success("🏆 **The Recommended Gold Standard:** Because it completely ignores how long a document is, it perfectly matches long heavy documents to tiny short search queries—as long as their core topic points in the same direction, the angle between them is 0, so the Cosine Similarity is a perfect `1.0`!")

st.markdown("---")

st.markdown("### 🎮 Interactive 2D Vector Sandbox")
st.write("Change the coordinates of Vector A and Vector B to see how the three similarity metrics react in real-time!")

col1, col2 = st.columns(2)
with col1:
    x1 = st.slider("Vector A (X):", 0.1, 10.0, 8.0, 0.1)
    y1 = st.slider("Vector A (Y):", 0.1, 10.0, 5.0, 0.1)
with col2:
    x2 = st.slider("Vector B (X):", 0.1, 10.0, 3.0, 0.1)
    y2 = st.slider("Vector B (Y):", 0.1, 10.0, 8.0, 0.1)

# Math
A = np.array([x1, y1])
B = np.array([x2, y2])

euclidean = np.linalg.norm(A - B)
dot_product = np.dot(A, B)
norm_A = np.linalg.norm(A)
norm_B = np.linalg.norm(B)
cosine = dot_product / (norm_A * norm_B)

m1, m2, m3 = st.columns(3)
with m1:
    metric_card("Euclidean Distance", f"{euclidean:.2f}", "(Distance: Lower is closer)")
with m2:
    metric_card("Dot Product", f"{dot_product:.2f}", "(Unbounded: Higher is closer)")
with m3:
    metric_card("Cosine Similarity", f"{cosine:.4f}", "(Angle: 1.0 = exact match, 0.0 = unmatched)")

# Plotting the vectors
fig = go.Figure()

# Add Vector A
fig.add_trace(go.Scatter(
    x=[0, x1], y=[0, y1],
    mode="lines+markers+text",
    name="Vector A",
    line=dict(color="#FF6B6B", width=4),
    marker=dict(size=12, symbol="arrow-bar-up", angleref="previous"),
    text=["", "A"], textposition="top center"
))

# Add Vector B
fig.add_trace(go.Scatter(
    x=[0, x2], y=[0, y2],
    mode="lines+markers+text",
    name="Vector B",
    line=dict(color="#4ECB71", width=4),
    marker=dict(size=12, symbol="arrow-bar-up", angleref="previous"),
    text=["", "B"], textposition="top center"
))

# Calculate angle arc
angle_A = np.arctan2(y1, x1)
angle_B = np.arctan2(y2, x2)
start_angle = min(angle_A, angle_B)
end_angle = max(angle_A, angle_B)
arc_radius = min(norm_A, norm_B) * 0.4
t = np.linspace(start_angle, end_angle, 50)
arc_x = arc_radius * np.cos(t)
arc_y = arc_radius * np.sin(t)

fig.add_trace(go.Scatter(
    x=arc_x, y=arc_y, mode="lines", name="Angle (θ)",
    line=dict(color="#B794F6", width=2, dash="dash"),
))

# Draw Euclidean distance line
fig.add_trace(go.Scatter(
    x=[x1, x2], y=[y1, y2],
    mode="lines",
    line=dict(color="#FAFAFA", width=2, dash="dot"),
    name="Euclidean Distance"
))

fig.update_layout(
    **PLOTLY_LAYOUT,
    xaxis=dict(range=[0, 11], showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)", title="Dimension 1"),
    yaxis=dict(range=[0, 11], showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)", title="Dimension 2"),
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### 💡 Try This Experiment!
1. Set Vector A to **(8.0, 8.0)**. (Imagine this is an 8,000 word essay about Dogs)
2. Set Vector B to **(2.0, 2.0)**. (Imagine this is a 2-word Google search for "Dogs")

**Notice what happens!** 
The Euclidean Distance says they are physically super far apart (**8.49** units of distance). But the **Cosine Similarity is exactly 1.0000**! 

Because they point in the exact same mathematical direction, Cosine realizes they mean the exact same thing despite their size differences. This is why Cosine is the king of AI text search!
""")
