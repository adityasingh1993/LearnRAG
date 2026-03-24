"""
RAG Learning Lab - Interactive educational app for understanding
Retrieval-Augmented Generation from basics to advanced.
"""

import streamlit as st

st.set_page_config(
    page_title="RAG Learning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .main-header h1 {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #6C63FF, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.3rem;
        color: #888;
    }
    .module-card {
        background: linear-gradient(145deg, #1a1d29, #22263a);
        border: 1px solid #333;
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.2s, border-color 0.2s;
    }
    .module-card:hover {
        transform: translateY(-4px);
        border-color: #6C63FF;
    }
    .module-card .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .module-card h3 {
        color: #fff;
        margin: 0.5rem 0;
    }
    .module-card p {
        color: #999;
        font-size: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-beginner { background: #00CC9622; color: #00CC96; }
    .badge-intermediate { background: #FFAA0022; color: #FFAA00; }
    .badge-advanced { background: #FF555522; color: #FF5555; }
    .stat-card {
        text-align: center;
        padding: 1rem;
        background: #1a1d29;
        border-radius: 12px;
        border: 1px solid #333;
    }
    .stat-card .number {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .stat-card .label {
        color: #888;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Hero Section ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 RAG Learning Lab</h1>
    <p>Master Retrieval-Augmented Generation — from concepts to building your own pipeline</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="number">8</div><div class="label">Interactive Modules</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="number">🎮</div><div class="label">Drag & Drop Playground</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="number">3</div><div class="label">LLM Providers</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="number">∞</div><div class="label">Hands-on Learning</div></div>', unsafe_allow_html=True)

st.markdown("")

# ── Learning Path ───────────────────────────────────────────────────────
st.markdown("## Learning Path")
st.markdown("Navigate through modules in order, or jump to any topic you're curious about.")
st.markdown("")

modules = [
    ("📖", "Learn the Basics", "What is RAG? Why does it matter? Understand the big picture.",
     "beginner", "1_📖_Learn_Basics"),
    ("🧩", "Embeddings", "Turn text into numbers. Explore how machines understand meaning.",
     "beginner", "2_🧩_Embeddings"),
    ("📦", "Vector Stores", "Store and search through embeddings efficiently.",
     "beginner", "3_📦_Vector_Stores"),
    ("🔍", "Retrieval", "Find the most relevant information for any query.",
     "intermediate", "4_🔍_Retrieval"),
    ("🤖", "Generation", "Generate answers grounded in retrieved context.",
     "intermediate", "5_🤖_Generation"),
    ("🔬", "Full Pipeline", "See all components work together end-to-end.",
     "intermediate", "6_🔬_Full_Pipeline"),
    ("📊", "Evaluation", "Measure pipeline quality with enterprise-grade metrics.",
     "advanced", "7_📊_Evaluation"),
    ("🎮", "Playground", "Build your own RAG pipeline visually!",
     "advanced", "8_🎮_Playground"),
    ("❓", "Help & Reference", "How every Playground feature works, with resources.",
     "advanced", "9_❓_Help"),
]

rows = [modules[i:i+4] for i in range(0, len(modules), 4)]
for row in rows:
    cols = st.columns(4)
    for i, (icon, title, desc, level, page) in enumerate(row):
        with cols[i]:
            st.markdown(f"""
            <div class="module-card">
                <div class="icon">{icon}</div>
                <span class="badge badge-{level}">{level.upper()}</span>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("")

# ── How it Works ────────────────────────────────────────────────────────
st.markdown("## How RAG Works — In 30 Seconds")
st.markdown("")

flow_html = """
<div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:30px 0;flex-wrap:wrap;">
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#4ECDC422,#4ECDC411);
                border:2px solid #4ECDC4;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">📄</div>
        <div style="color:#4ECDC4;font-weight:600;margin-top:8px;">Your Documents</div>
        <div style="color:#666;font-size:0.8rem;">PDFs, text, web pages</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#45B7D122,#45B7D111);
                border:2px solid #45B7D1;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">✂️</div>
        <div style="color:#45B7D1;font-weight:600;margin-top:8px;">Chunk</div>
        <div style="color:#666;font-size:0.8rem;">Split into pieces</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#96CEB422,#96CEB411);
                border:2px solid #96CEB4;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">🔢</div>
        <div style="color:#96CEB4;font-weight:600;margin-top:8px;">Embed</div>
        <div style="color:#666;font-size:0.8rem;">Text → Vectors</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#FFEAA722,#FFEAA711);
                border:2px solid #FFEAA7;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">📦</div>
        <div style="color:#FFEAA7;font-weight:600;margin-top:8px;">Store</div>
        <div style="color:#666;font-size:0.8rem;">Vector database</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#DDA0DD22,#DDA0DD11);
                border:2px solid #DDA0DD;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">🔍</div>
        <div style="color:#DDA0DD;font-weight:600;margin-top:8px;">Retrieve</div>
        <div style="color:#666;font-size:0.8rem;">Find relevant chunks</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#98D8C822,#98D8C811);
                border:2px solid #98D8C8;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">🤖</div>
        <div style="color:#98D8C8;font-weight:600;margin-top:8px;">Generate</div>
        <div style="color:#666;font-size:0.8rem;">LLM answers with context</div>
    </div>
</div>
"""
st.html(flow_html)

# ── Quick Start Guide ───────────────────────────────────────────────────
st.markdown("## Quick Start")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🎓 New to RAG?
    Start with **Learn the Basics** and work through each module in order.
    Each module builds on the previous one, with interactive demos at every step.
    """)

with col2:
    st.markdown("""
    #### 🛠️ Ready to Build?
    Jump straight to the **Playground** and assemble a RAG pipeline
    using drag-and-drop. Configure LLM providers in the sidebar.
    """)

st.markdown("")

with st.expander("⚙️ Setup Guide — Configure your providers"):
    st.markdown("""
    **Option 1: OpenAI** (Recommended for best results)
    - Get an API key from [platform.openai.com](https://platform.openai.com)
    - Set `OPENAI_API_KEY` in your `.env` file or enter it in the sidebar

    **Option 2: OpenRouter** (Access many models with one key)
    - Get a key from [openrouter.ai](https://openrouter.ai)
    - Includes free models like Llama 3.1 and Gemma 2

    **Option 3: Ollama** (100% local, no API key needed)
    - Install from [ollama.com](https://ollama.com)
    - Run `ollama pull llama3.2` and `ollama pull nomic-embed-text`

    **No API key?** No problem! All educational content works without one.
    Interactive demos use local TF-IDF embeddings and mock responses.
    """)

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#555;padding:1rem;">'
    'Built with Streamlit | '
    '<span style="color:#6C63FF;">RAG Learning Lab</span> — Learn by doing'
    '</div>',
    unsafe_allow_html=True,
)
