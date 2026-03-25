"""Home page — AI Learning Lab landing."""

import streamlit as st

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
    <h1>🧠 AI Learning Lab</h1>
    <p>Master RAG, Agents, MCP, A2A, Embeddings & Prompting — from concepts to building your own systems</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="number">44</div><div class="label">Interactive Modules</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="number">6</div><div class="label">Learning Tracks</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="number">3</div><div class="label">LLM Providers</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="number">∞</div><div class="label">Hands-on Learning</div></div>', unsafe_allow_html=True)

st.markdown("")

# ── Learning Path ───────────────────────────────────────────────────────
st.markdown("## Learning Path")
st.markdown("Navigate through modules in order, or jump to any topic from the sidebar.")
st.markdown("")

modules = [
    ("📖", "Learn the Basics", "What is RAG? Why does it matter? Understand the big picture.", "beginner"),
    ("🧩", "Embeddings", "Turn text into numbers. Explore how machines understand meaning.", "beginner"),
    ("📦", "Vector Stores", "Store and search through embeddings efficiently.", "beginner"),
    ("🔍", "Retrieval", "Find the most relevant information for any query.", "intermediate"),
    ("🤖", "Generation", "Generate answers grounded in retrieved context.", "intermediate"),
    ("🔬", "Full Pipeline", "See all components work together end-to-end.", "intermediate"),
    ("📊", "Evaluation", "Measure pipeline quality with enterprise-grade metrics.", "advanced"),
    ("🎮", "Playground", "Build your own RAG pipeline visually!", "advanced"),
    ("❓", "RAG Help", "How every RAG Playground feature works, with resources.", "advanced"),
    ("🤖", "Agent Basics", "What are AI Agents? How do they differ from RAG?", "beginner"),
    ("🔧", "Tools", "Function calling, tool creation, and interactive demos.", "beginner"),
    ("🔄", "Agent Patterns", "ReAct, Plan-Execute, Reflection — see them run live.", "intermediate"),
    ("🌐", "Multi-Agent", "Router, orchestrator, and debate patterns for complex tasks.", "intermediate"),
    ("🎮", "Agent Playground", "Build your own agent with tools, RAG, and chat!", "advanced"),
    ("❓", "Agent Help", "Reference for all agent features, patterns, and resources.", "advanced"),
    ("🔌", "MCP Basics", "What is the Model Context Protocol? The USB-C for AI.", "beginner"),
    ("🏗️", "MCP Architecture", "Hosts, Clients, Servers, and Transports deep dive.", "intermediate"),
    ("🧱", "MCP Primitives", "Resources, Tools, and Prompts — interactive explorer.", "intermediate"),
    ("🛠️", "MCP Server Builder", "Build your own MCP server step by step.", "advanced"),
    ("🎮", "MCP Playground", "Multi-server environment with live protocol log.", "advanced"),
    ("❓", "MCP Help", "Comprehensive MCP reference and resources.", "advanced"),
    ("🤝", "A2A Basics", "Agent-to-Agent Protocol — how agents collaborate.", "beginner"),
    ("🪪", "Agent Cards", "How agents describe themselves and discover each other.", "intermediate"),
    ("📋", "A2A Tasks", "Task lifecycle, messages, artifacts — interactive demo.", "intermediate"),
    ("🌐", "A2A Collaboration", "Router, pipeline, and parallel multi-agent patterns.", "advanced"),
    ("🎮", "A2A Playground", "Full multi-agent environment with routing and pipelines.", "advanced"),
    ("❓", "A2A Help", "A2A reference, comparisons, and further reading.", "advanced"),
    ("📊", "Bag of Words", "The simplest text-to-numbers technique. Sparse vectors.", "beginner"),
    ("📈", "TF-IDF", "Term frequency meets inverse document frequency.", "beginner"),
    ("🧠", "Word2Vec", "Dense vectors from Google's word2vec-google-news-300.", "intermediate"),
    ("🌐", "GloVe", "Global vectors for word representation from Stanford.", "intermediate"),
    ("🤖", "Transformers", "Contextual embeddings with Sentence-BERT.", "intermediate"),
    ("🔬", "Grand Comparison", "BoW vs TF-IDF vs Transformers side by side.", "advanced"),
    ("🔍", "Semantic Search", "Keyword search vs semantic search showdown.", "advanced"),
    ("🎮", "Guess Embedding", "Mystery technique game — identify the algorithm.", "advanced"),
    ("🌌", "3D Universe", "Navigate word embeddings in 3D WebGL space.", "advanced"),
    ("🖼️", "Multimodal CLIP", "Image + text in a shared embedding space.", "advanced"),
    ("📐", "Vector Similarity", "Euclidean vs Cosine vs Dot Product — visual proof.", "beginner"),
    ("📚", "Embedding Resources", "Papers, MTEB leaderboard, and further reading.", "beginner"),
    ("📖", "Prompting Tutorial", "8 techniques from Zero-Shot to Tree-of-Thought.", "beginner"),
    ("🏟️", "Prompt Arena", "Compare two prompting techniques side by side.", "intermediate"),
    ("🔍", "Hallucination Detector", "Claim-by-claim analysis of LLM responses.", "intermediate"),
    ("🛠️", "Prompt Workbench", "Build custom prompts with template variables.", "advanced"),
    ("📊", "Prompt Analytics", "Track performance and history across sessions.", "advanced"),
]

# Page routes for each module card title.
module_paths = {
    "Learn the Basics": "pages/1_📖_Learn_Basics.py",
    "Embeddings": "pages/2_🧩_Embeddings.py",
    "Vector Stores": "pages/3_📦_Vector_Stores.py",
    "Retrieval": "pages/4_🔍_Retrieval.py",
    "Generation": "pages/5_🤖_Generation.py",
    "Full Pipeline": "pages/6_🔬_Full_Pipeline.py",
    "Evaluation": "pages/7_📊_Evaluation.py",
    "Playground": "pages/8_🎮_Playground.py",
    "RAG Help": "pages/9_❓_Help.py",
    "Agent Basics": "pages/10_🤖_Agent_Basics.py",
    "Tools": "pages/11_🔧_Tools.py",
    "Agent Patterns": "pages/12_🔄_Agent_Patterns.py",
    "Multi-Agent": "pages/13_🌐_Multi_Agent.py",
    "Agent Playground": "pages/14_🎮_Agent_Playground.py",
    "Agent Help": "pages/15_❓_Agent_Help.py",
    "MCP Basics": "pages/16_🔌_MCP_Basics.py",
    "MCP Architecture": "pages/17_🏗️_MCP_Architecture.py",
    "MCP Primitives": "pages/18_🧱_MCP_Primitives.py",
    "MCP Server Builder": "pages/19_🛠️_MCP_Server_Builder.py",
    "MCP Playground": "pages/20_🎮_MCP_Playground.py",
    "MCP Help": "pages/21_❓_MCP_Help.py",
    "A2A Basics": "pages/22_🤝_A2A_Basics.py",
    "Agent Cards": "pages/23_🪪_Agent_Cards.py",
    "A2A Tasks": "pages/24_📋_A2A_Tasks.py",
    "A2A Collaboration": "pages/25_🌐_A2A_Collaboration.py",
    "A2A Playground": "pages/26_🎮_A2A_Playground.py",
    "A2A Help": "pages/27_❓_A2A_Help.py",
    "Bag of Words": "pages/28_📊_Bag_of_Words.py",
    "TF-IDF": "pages/29_📈_TF_IDF.py",
    "Word2Vec": "pages/30_🧠_Word2Vec.py",
    "GloVe": "pages/31_🌐_GloVe.py",
    "Transformers": "pages/32_🤖_Transformers.py",
    "Grand Comparison": "pages/33_🔬_Grand_Comparison.py",
    "Semantic Search": "pages/34_🔍_Semantic_Search.py",
    "Guess Embedding": "pages/35_🎮_Guess_Embedding.py",
    "3D Universe": "pages/36_🌌_3D_Universe.py",
    "Multimodal CLIP": "pages/37_🖼️_Multimodal.py",
    "Vector Similarity": "pages/38_📐_Vector_Similarity.py",
    "Embedding Resources": "pages/39_📚_Resources.py",
    "Prompting Tutorial": "pages/40_📖_Tutorial.py",
    "Prompt Arena": "pages/41_🏟️_Prompt_Arena.py",
    "Hallucination Detector": "pages/42_🔍_Hallucination_Detector.py",
    "Prompt Workbench": "pages/43_🛠️_Prompt_Workbench.py",
    "Prompt Analytics": "pages/44_📊_Analytics.py",
}

rows = [modules[i:i+4] for i in range(0, len(modules), 4)]
for row in rows:
    cols = st.columns(4)
    for i, (icon, title, desc, level) in enumerate(row):
        with cols[i]:
            st.markdown(f"""
            <div class="module-card">
                <div class="icon">{icon}</div>
                <span class="badge badge-{level}">{level.upper()}</span>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            page_path = module_paths.get(title)
            if page_path:
                st.page_link(
                    page_path,
                    label=f"Open: {title}",
                    icon="🔗",
                    use_container_width=True,
                )

st.markdown("")

# ── How it Works ────────────────────────────────────────────────────────
st.markdown("## How RAG Works")
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

st.markdown("## How AI Agents Work")

agent_flow_html = """
<div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:30px 0;flex-wrap:wrap;">
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#FF8C9422,#FF8C9411);
                border:2px solid #FF8C94;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">❓</div>
        <div style="color:#FF8C94;font-weight:600;margin-top:8px;">User Question</div>
        <div style="color:#666;font-size:0.8rem;">Complex task</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#6C63FF22,#6C63FF11);
                border:2px solid #6C63FF;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">🧠</div>
        <div style="color:#6C63FF;font-weight:600;margin-top:8px;">Think</div>
        <div style="color:#666;font-size:0.8rem;">LLM reasons</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#00CC9622,#00CC9611);
                border:2px solid #00CC96;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">🔧</div>
        <div style="color:#00CC96;font-weight:600;margin-top:8px;">Use Tools</div>
        <div style="color:#666;font-size:0.8rem;">Calculator, search, RAG</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#FFAA0022,#FFAA0011);
                border:2px solid #FFAA00;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">👁️</div>
        <div style="color:#FFAA00;font-weight:600;margin-top:8px;">Observe</div>
        <div style="color:#666;font-size:0.8rem;">See tool results</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#00D2FF22,#00D2FF11);
                border:2px solid #00D2FF;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">🔄</div>
        <div style="color:#00D2FF;font-weight:600;margin-top:8px;">Iterate</div>
        <div style="color:#666;font-size:0.8rem;">Repeat until done</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#98D8C822,#98D8C811);
                border:2px solid #98D8C8;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">✅</div>
        <div style="color:#98D8C8;font-weight:600;margin-top:8px;">Answer</div>
        <div style="color:#666;font-size:0.8rem;">Grounded response</div>
    </div>
</div>
"""
st.html(agent_flow_html)

st.markdown("## How MCP Connects AI to Tools")

mcp_flow_html = """
<div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:30px 0;flex-wrap:wrap;">
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#9B59B622,#9B59B611);
                border:2px solid #9B59B6;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">🖥️</div>
        <div style="color:#9B59B6;font-weight:600;margin-top:8px;">MCP Host</div>
        <div style="color:#666;font-size:0.8rem;">AI app (Claude, IDE)</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#3498DB22,#3498DB11);
                border:2px solid #3498DB;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">🔗</div>
        <div style="color:#3498DB;font-weight:600;margin-top:8px;">MCP Client</div>
        <div style="color:#666;font-size:0.8rem;">JSON-RPC session</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#E67E2222,#E67E2211);
                border:2px solid #E67E22;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">⚙️</div>
        <div style="color:#E67E22;font-weight:600;margin-top:8px;">MCP Server</div>
        <div style="color:#666;font-size:0.8rem;">Exposes capabilities</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#1ABC9C22,#1ABC9C11);
                border:2px solid #1ABC9C;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">📄🔧💬</div>
        <div style="color:#1ABC9C;font-weight:600;margin-top:8px;">Primitives</div>
        <div style="color:#666;font-size:0.8rem;">Resources, Tools, Prompts</div>
    </div>
</div>
"""
st.html(mcp_flow_html)

st.markdown("## How A2A Enables Agent Collaboration")

a2a_flow_html = """
<div style="display:flex;align-items:center;justify-content:center;gap:12px;padding:30px 0;flex-wrap:wrap;">
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#E74C3C22,#E74C3C11);
                border:2px solid #E74C3C;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">🤝</div>
        <div style="color:#E74C3C;font-weight:600;margin-top:8px;">Discover</div>
        <div style="color:#666;font-size:0.8rem;">Agent Cards</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#9B59B622,#9B59B611);
                border:2px solid #9B59B6;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">📋</div>
        <div style="color:#9B59B6;font-weight:600;margin-top:8px;">Send Task</div>
        <div style="color:#666;font-size:0.8rem;">Delegate work</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#F39C1222,#F39C1211);
                border:2px solid #F39C12;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">💬</div>
        <div style="color:#F39C12;font-weight:600;margin-top:8px;">Communicate</div>
        <div style="color:#666;font-size:0.8rem;">Messages & Parts</div>
    </div>
    <div style="font-size:28px;color:#555;">→</div>
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#2ECC7122,#2ECC7111);
                border:2px solid #2ECC71;border-radius:16px;min-width:140px;">
        <div style="font-size:32px;">📦</div>
        <div style="color:#2ECC71;font-weight:600;margin-top:8px;">Artifacts</div>
        <div style="color:#666;font-size:0.8rem;">Deliverables</div>
    </div>
</div>
"""
st.html(a2a_flow_html)

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
    Jump to the **RAG Playground**, **Agent Playground**, **MCP Playground**, or
    **A2A Playground** to build hands-on. Configure LLM providers in the sidebar.
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
    '<span style="color:#6C63FF;">AI Learning Lab</span> — Learn by doing'
    '</div>',
    unsafe_allow_html=True,
)
