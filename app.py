"""
RAG & AI Learning Lab - Interactive educational app for understanding
RAG, AI Agents, MCP, and A2A Protocol from basics to advanced.
"""

import streamlit as st

st.set_page_config(
    page_title="AI Learning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Navigation with Module Sections ────────────────────────────
home = st.Page("home.py", title="Home", icon="🏠", default=True)

rag_pages = [
    st.Page("pages/1_📖_Learn_Basics.py", title="Learn Basics", icon="📖"),
    st.Page("pages/2_🧩_Embeddings.py", title="Embeddings", icon="🧩"),
    st.Page("pages/3_📦_Vector_Stores.py", title="Vector Stores", icon="📦"),
    st.Page("pages/4_🔍_Retrieval.py", title="Retrieval", icon="🔍"),
    st.Page("pages/5_🤖_Generation.py", title="Generation", icon="🤖"),
    st.Page("pages/6_🔬_Full_Pipeline.py", title="Full Pipeline", icon="🔬"),
    st.Page("pages/7_📊_Evaluation.py", title="Evaluation", icon="📊"),
    st.Page("pages/8_🎮_Playground.py", title="RAG Playground", icon="🎮"),
    st.Page("pages/9_❓_Help.py", title="RAG Help", icon="❓"),
]

agent_pages = [
    st.Page("pages/10_🤖_Agent_Basics.py", title="Agent Basics", icon="🤖"),
    st.Page("pages/11_🔧_Tools.py", title="Tools", icon="🔧"),
    st.Page("pages/12_🔄_Agent_Patterns.py", title="Agent Patterns", icon="🔄"),
    st.Page("pages/13_🌐_Multi_Agent.py", title="Multi-Agent", icon="🌐"),
    st.Page("pages/14_🎮_Agent_Playground.py", title="Agent Playground", icon="🎮"),
    st.Page("pages/15_❓_Agent_Help.py", title="Agent Help", icon="❓"),
]

mcp_pages = [
    st.Page("pages/16_🔌_MCP_Basics.py", title="MCP Basics", icon="🔌"),
    st.Page("pages/17_🏗️_MCP_Architecture.py", title="MCP Architecture", icon="🏗️"),
    st.Page("pages/18_🧱_MCP_Primitives.py", title="MCP Primitives", icon="🧱"),
    st.Page("pages/19_🛠️_MCP_Server_Builder.py", title="Server Builder", icon="🛠️"),
    st.Page("pages/20_🎮_MCP_Playground.py", title="MCP Playground", icon="🎮"),
    st.Page("pages/21_❓_MCP_Help.py", title="MCP Help", icon="❓"),
]

a2a_pages = [
    st.Page("pages/22_🤝_A2A_Basics.py", title="A2A Basics", icon="🤝"),
    st.Page("pages/23_🪪_Agent_Cards.py", title="Agent Cards", icon="🪪"),
    st.Page("pages/24_📋_A2A_Tasks.py", title="A2A Tasks", icon="📋"),
    st.Page("pages/25_🌐_A2A_Collaboration.py", title="A2A Collaboration", icon="🌐"),
    st.Page("pages/26_🎮_A2A_Playground.py", title="A2A Playground", icon="🎮"),
    st.Page("pages/27_❓_A2A_Help.py", title="A2A Help", icon="❓"),
]

embedding_pages = [
    st.Page("pages/28_📊_Bag_of_Words.py", title="Bag of Words", icon="📊"),
    st.Page("pages/29_📈_TF_IDF.py", title="TF-IDF", icon="📈"),
    st.Page("pages/30_🧠_Word2Vec.py", title="Word2Vec", icon="🧠"),
    st.Page("pages/31_🌐_GloVe.py", title="GloVe", icon="🌐"),
    st.Page("pages/32_🤖_Transformers.py", title="Transformers", icon="🤖"),
    st.Page("pages/33_🔬_Grand_Comparison.py", title="Grand Comparison", icon="🔬"),
    st.Page("pages/34_🔍_Semantic_Search.py", title="Semantic Search", icon="🔍"),
    st.Page("pages/35_🎮_Guess_Embedding.py", title="Guess the Embedding", icon="🎮"),
    st.Page("pages/36_🌌_3D_Universe.py", title="3D Universe", icon="🌌"),
    st.Page("pages/37_🖼️_Multimodal.py", title="Multimodal (CLIP)", icon="🖼️"),
    st.Page("pages/38_📐_Vector_Similarity.py", title="Vector Similarity", icon="📐"),
    st.Page("pages/39_📚_Resources.py", title="Embedding Resources", icon="📚"),
]

prompting_pages = [
    st.Page("pages/40_📖_Tutorial.py", title="Prompting Tutorial", icon="📖"),
    st.Page("pages/41_🏟️_Prompt_Arena.py", title="Prompt Arena", icon="🏟️"),
    st.Page("pages/42_🔍_Hallucination_Detector.py", title="Hallucination Detector", icon="🔍"),
    st.Page("pages/43_🛠️_Prompt_Workbench.py", title="Prompt Workbench", icon="🛠️"),
    st.Page("pages/44_📊_Analytics.py", title="Prompt Analytics", icon="📊"),
]

pg = st.navigation({
    "": [home],
    "Prompting": prompting_pages,
    "Embeddings": embedding_pages,
    "RAG": rag_pages,
    "Agents": agent_pages,
    "MCP": mcp_pages,
    "A2A": a2a_pages,
})

pg.run()
