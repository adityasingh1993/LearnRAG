"""
Module 1: Learn the Basics of RAG
Interactive introduction with visual explanations and quizzes.
"""

import streamlit as st

st.set_page_config(page_title="Learn the Basics | RAG Lab", page_icon="📖", layout="wide")

st.markdown("""
<style>
    .concept-box {
        background: linear-gradient(145deg, #1a1d29, #22263a);
        border-left: 4px solid #6C63FF;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .concept-box h4 { color: #6C63FF; margin-top: 0; }
    .analogy-box {
        background: #2a1f3d;
        border: 1px solid #6C63FF44;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .problem-box {
        background: #3d1f1f;
        border-left: 4px solid #FF5555;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 0.5rem 0;
    }
    .solution-box {
        background: #1f3d2a;
        border-left: 4px solid #00CC96;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📖 Learn the Basics")
st.markdown("*Understand what RAG is, why it exists, and how it works — no code required.*")
st.markdown("---")

# ── What is RAG? ─────────────────────────────────────────────────────────
st.header("What is RAG?")

st.markdown("""
<div class="concept-box">
<h4>Retrieval-Augmented Generation (RAG)</h4>
A technique that <strong>enhances LLMs</strong> by giving them access to external knowledge
at the time they generate a response. Instead of relying only on training data,
the model receives relevant documents as context.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="analogy-box">
<strong>🎓 Analogy: Open-Book Exam</strong><br><br>
Think of a regular LLM as a student taking a <strong>closed-book exam</strong> — they can only use
what they memorized (training data). RAG is like switching to an <strong>open-book exam</strong> —
the student can look up information in their notes (retrieved documents) before answering.
</div>
""", unsafe_allow_html=True)

# ── Why do we need RAG? ──────────────────────────────────────────────────
st.header("Why Do We Need RAG?")
st.markdown("LLMs are powerful, but they have real limitations:")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="problem-box">
    <strong>🕐 Knowledge Cutoff</strong><br>
    LLMs only know what was in their training data.
    They can't answer about recent events.
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="problem-box">
    <strong>👻 Hallucinations</strong><br>
    LLMs can confidently generate incorrect information
    that sounds perfectly plausible.
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="problem-box">
    <strong>🏢 No Private Data</strong><br>
    LLMs don't know your company docs,
    internal wikis, or personal files.
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="solution-box">
    <strong>✅ RAG Fix</strong><br>
    Connect to live/updated knowledge bases that are always current.
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="solution-box">
    <strong>✅ RAG Fix</strong><br>
    Ground responses in actual source documents with citations.
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="solution-box">
    <strong>✅ RAG Fix</strong><br>
    Index your own data and let the LLM use it contextually.
    </div>
    """, unsafe_allow_html=True)

# ── How Does RAG Work? ───────────────────────────────────────────────────
st.markdown("---")
st.header("How Does RAG Work?")
st.markdown("RAG has two phases: **Indexing** (prepare documents) and **Querying** (answer questions).")

tab1, tab2 = st.tabs(["📥 Phase 1: Indexing", "❓ Phase 2: Querying"])

with tab1:
    st.markdown("#### Preparing your documents for search")
    steps = [
        ("1️⃣", "Load Documents", "Gather your text data — PDFs, web pages, databases, etc."),
        ("2️⃣", "Chunk Text", "Split documents into smaller, meaningful pieces (typically 200-1000 characters)."),
        ("3️⃣", "Create Embeddings", "Convert each chunk into a numerical vector that captures its meaning."),
        ("4️⃣", "Store in Vector DB", "Save the vectors in a specialized database for fast similarity search."),
    ]
    for emoji, title, desc in steps:
        st.markdown(f"**{emoji} {title}** — {desc}")

    st.markdown("")
    st.info("💡 **This only happens once** (or when you update your documents). "
            "After indexing, querying is fast!")

with tab2:
    st.markdown("#### Answering a user's question")
    steps = [
        ("1️⃣", "Embed the Query", "Convert the user's question into a vector using the same embedding model."),
        ("2️⃣", "Search Vector DB", "Find the document chunks most similar to the query vector."),
        ("3️⃣", "Build Prompt", "Combine the retrieved chunks with the question into a structured prompt."),
        ("4️⃣", "Generate Answer", "The LLM reads the context and generates a grounded, accurate response."),
    ]
    for emoji, title, desc in steps:
        st.markdown(f"**{emoji} {title}** — {desc}")

    st.markdown("")
    st.success("🎯 **Result:** The LLM gives an answer based on YOUR data, not just its training.")

# ── RAG vs Alternatives ──────────────────────────────────────────────────
st.markdown("---")
st.header("RAG vs. Alternatives")

comparison_data = {
    "Approach": ["Fine-tuning", "RAG", "Long Context", "Prompt Engineering"],
    "Cost": ["$$$$ (GPU training)", "$ (API calls)", "$$ (large prompts)", "$ (clever prompting)"],
    "Fresh Data": ["❌ Needs retraining", "✅ Always current", "✅ If provided", "❌ Static knowledge"],
    "Private Data": ["⚠️ Risk of leaking", "✅ Data stays local", "⚠️ Sent to API", "❌ Not available"],
    "Accuracy": ["🟡 Can overfit", "🟢 Grounded", "🟢 Good if in context", "🔴 May hallucinate"],
    "Setup Effort": ["High", "Medium", "Low", "Low"],
}
st.table(comparison_data)

# ── Interactive Quiz ──────────────────────────────────────────────────────
st.markdown("---")
st.header("🧪 Quick Knowledge Check")

with st.form("quiz_basics"):
    q1 = st.radio(
        "**Q1:** What does the 'R' in RAG stand for?",
        ["Recursive", "Retrieval", "Reinforced", "Responsive"],
        index=None,
    )
    q2 = st.radio(
        "**Q2:** Why do LLMs hallucinate?",
        [
            "They are broken",
            "They generate statistically likely text even without factual grounding",
            "They always lie",
            "They don't understand language",
        ],
        index=None,
    )
    q3 = st.radio(
        "**Q3:** What is stored in a vector database?",
        ["Raw text files", "SQL tables", "Numerical representations (embeddings) of text", "Images only"],
        index=None,
    )
    submitted = st.form_submit_button("Check Answers", type="primary")

if submitted:
    score = 0
    if q1 == "Retrieval":
        score += 1
        st.success("Q1: Correct! RAG = **Retrieval**-Augmented Generation")
    elif q1:
        st.error("Q1: Not quite. RAG = **Retrieval**-Augmented Generation")

    if q2 == "They generate statistically likely text even without factual grounding":
        score += 1
        st.success("Q2: Correct! LLMs predict likely next tokens, not necessarily true ones.")
    elif q2:
        st.error("Q2: LLMs hallucinate because they predict statistically likely text, not verified facts.")

    if q3 == "Numerical representations (embeddings) of text":
        score += 1
        st.success("Q3: Correct! Vector databases store embeddings for similarity search.")
    elif q3:
        st.error("Q3: Vector databases store **embeddings** — numerical vectors representing text meaning.")

    if score == 3:
        st.balloons()
        st.success(f"🎉 Perfect score! {score}/3 — You're ready for the next module!")
    elif score > 0:
        st.info(f"Score: {score}/3 — Good start! Review the sections above and try again.")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("app.py", label="← Home", icon="🏠")
with col2:
    st.page_link("pages/1_📚_RAG/02_🧩_Embeddings.py", label="Next: Embeddings →", icon="🧩")
