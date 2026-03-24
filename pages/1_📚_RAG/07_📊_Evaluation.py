"""
Module 7: RAG Evaluation
Enterprise-grade evaluation of retrieval and generation quality.
Interactive metrics, LLM-as-judge, and side-by-side comparison.
"""

import streamlit as st
import numpy as np

st.set_page_config(page_title="Evaluation | RAG Lab", page_icon="📊", layout="wide")

from components.sidebar import render_provider_config, get_llm_provider, get_embedding_provider

render_provider_config()

st.title("📊 RAG Evaluation")
st.markdown("*Measure, compare, and improve your RAG pipeline with enterprise-grade evaluation.*")
st.markdown("---")

section = st.selectbox(
    "Choose an evaluation module:",
    [
        "Retrieval Metrics",
        "Generation Quality (LLM-as-Judge)",
        "End-to-End Pipeline Evaluation",
        "A/B Strategy Comparison",
    ],
    key="eval_section",
)

# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ═══════════════════════════════════════════════════════════════════════════
if section == "Retrieval Metrics":
    st.header("🎯 Retrieval Metrics")

    st.markdown("""
    Retrieval is the foundation of RAG — if you retrieve the wrong chunks, the LLM
    can't generate a good answer. These metrics quantify how well your retrieval is working.
    """)

    tab_learn, tab_calc, tab_live = st.tabs(["Learn", "Interactive Calculator", "Live Evaluation"])

    # ── Learn tab ──
    with tab_learn:
        st.markdown("""
        | Metric | What It Measures | Range | Formula |
        |--------|-----------------|-------|---------|
        | **Precision@K** | What fraction of retrieved docs are relevant? | 0–1 | `relevant_in_K / K` |
        | **Recall@K** | What fraction of all relevant docs did we retrieve? | 0–1 | `relevant_in_K / total_relevant` |
        | **MRR** | How high is the first relevant result? | 0–1 | `1 / rank_of_first_relevant` |
        | **NDCG@K** | Are relevant docs ranked higher than irrelevant ones? | 0–1 | Normalized Discounted Cumulative Gain |
        | **Hit Rate** | Did we find at least one relevant doc? | 0 or 1 | Binary |
        | **MAP** | Average precision across all relevant docs | 0–1 | Mean of precision at each relevant rank |

        ---
        #### When to use which metric

        - **Precision@K** — when showing K results to a user (like search results). Higher = less noise.
        - **Recall@K** — when you need to find *all* relevant information (legal, compliance). Higher = fewer misses.
        - **MRR** — when only the top result matters (chatbot, single-answer Q&A).
        - **NDCG@K** — when rank order matters (the best doc should be first).
        - **MAP** — single-number summary of overall retrieval quality across multiple queries.
        """)

    # ── Interactive Calculator tab ──
    with tab_calc:
        st.markdown("#### Mark which retrieved results are relevant")
        st.caption("Simulate a retrieval of K documents and see how metrics change.")

        k_val = st.slider("Number of retrieved results (K):", 3, 10, 5, key="eval_k")
        total_relevant = st.number_input(
            "Total relevant docs in the full corpus:",
            min_value=1, max_value=100, value=4, key="eval_total_rel",
        )

        st.markdown("**Is each result relevant?**")
        relevance = []
        cols = st.columns(k_val)
        for i in range(k_val):
            with cols[i]:
                relevance.append(st.checkbox(f"#{i+1}", key=f"eval_rel_{i}"))

        if any(relevance):
            relevant_count = sum(relevance)
            precision = relevant_count / k_val
            recall = min(relevant_count / total_relevant, 1.0)

            first_rel = next((i + 1 for i, r in enumerate(relevance) if r), None)
            mrr = 1.0 / first_rel if first_rel else 0.0

            hit_rate = 1.0 if any(relevance) else 0.0

            # NDCG
            dcg = sum((1.0 if relevance[i] else 0.0) / np.log2(i + 2) for i in range(k_val))
            ideal_rels = sorted(relevance, reverse=True)
            idcg = sum((1.0 if ideal_rels[i] else 0.0) / np.log2(i + 2) for i in range(k_val))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            # MAP (average precision)
            running_relevant = 0
            precision_sum = 0.0
            for i, r in enumerate(relevance):
                if r:
                    running_relevant += 1
                    precision_sum += running_relevant / (i + 1)
            avg_precision = precision_sum / total_relevant if total_relevant > 0 else 0.0

            st.markdown("---")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric(f"Precision@{k_val}", f"{precision:.1%}")
            m2.metric(f"Recall@{k_val}", f"{recall:.1%}")
            m3.metric("MRR", f"{mrr:.3f}")
            m4.metric(f"NDCG@{k_val}", f"{ndcg:.3f}")
            m5.metric("Hit Rate", f"{hit_rate:.0f}")
            m6.metric("Avg Precision", f"{avg_precision:.3f}")

            st.markdown("---")
            st.markdown("#### Interpretation")
            issues = []
            if precision < 0.5:
                issues.append("**Low Precision** — too many irrelevant results. Try reranking or a more precise retrieval strategy.")
            if recall < 0.5:
                issues.append("**Low Recall** — missing relevant docs. Increase K, use hybrid search, or try multi-query retrieval.")
            if mrr < 0.5:
                issues.append("**Low MRR** — the first relevant result is buried. Reranking would help push it to the top.")
            if ndcg < 0.7:
                issues.append("**Low NDCG** — relevant results aren't ranked optimally. Consider MMR or reranking.")
            if not issues:
                st.success("All metrics look healthy!")
            else:
                for issue in issues:
                    st.warning(issue)
        else:
            st.info("Check at least one result as relevant to see metrics.")

    # ── Live Evaluation tab ──
    with tab_live:
        st.markdown("""
        Run a real retrieval against your own documents and judge the results.
        """)

        eval_docs = st.text_area(
            "Knowledge base (one fact per line):",
            value=(
                "RAG stands for Retrieval-Augmented Generation.\n"
                "RAG combines retrieval systems with generative language models.\n"
                "Vector stores like ChromaDB enable fast similarity search.\n"
                "Embeddings are numerical representations of text.\n"
                "HNSW is an algorithm for approximate nearest neighbor search.\n"
                "Cosine similarity measures the angle between two vectors.\n"
                "Chunking splits documents into smaller pieces for embedding.\n"
                "Fine-tuning changes model weights; RAG adds knowledge at inference time.\n"
                "BM25 is a keyword-based ranking algorithm.\n"
                "Hybrid search combines semantic and keyword approaches."
            ),
            height=200,
            key="eval_live_docs",
        )

        eval_query = st.text_input("Query:", "How does RAG work?", key="eval_live_query")
        eval_k = st.slider("Retrieve top K:", 1, 10, 5, key="eval_live_k")

        if eval_query and st.button("🔍 Retrieve & Evaluate", type="primary", key="eval_live_btn"):
            docs = [d.strip() for d in eval_docs.strip().split("\n") if d.strip()]

            with st.spinner("Embedding and retrieving..."):
                try:
                    embed = get_embedding_provider()
                    doc_embs = embed.embed(docs)
                    q_emb = embed.embed_query(eval_query)

                    from core.vector_store import NumpyVectorStore
                    vs = NumpyVectorStore()
                    vs.add(docs, doc_embs)
                    results = vs.search(q_emb, k=eval_k)

                    st.subheader("Retrieved Results — Judge Relevance")
                    st.caption("Mark each result as relevant or not. Metrics update automatically.")

                    live_rels = []
                    for i, r in enumerate(results):
                        col_check, col_text = st.columns([0.1, 0.9])
                        with col_check:
                            live_rels.append(st.checkbox("Rel", key=f"eval_live_rel_{i}", label_visibility="collapsed"))
                        with col_text:
                            color = "#2ecc71" if live_rels[-1] else "#e74c3c" if len(live_rels) > 0 and not live_rels[-1] else "#888"
                            st.markdown(
                                f'<div style="border-left:3px solid {color};padding:6px 12px;margin:2px 0;">'
                                f'<small style="color:#888;">#{i+1} | score: {r.score:.4f}</small><br>{r.text}</div>',
                                unsafe_allow_html=True,
                            )

                    if any(live_rels):
                        rel_count = sum(live_rels)
                        st.markdown("---")
                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"Precision@{eval_k}", f"{rel_count/eval_k:.1%}")
                        c2.metric("MRR", f"{1/(next(i+1 for i,r in enumerate(live_rels) if r)):.3f}")
                        c3.metric("Hit Rate", "1")

                except Exception as e:
                    st.error(f"Retrieval error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# GENERATION QUALITY (LLM-AS-JUDGE)
# ═══════════════════════════════════════════════════════════════════════════
elif section == "Generation Quality (LLM-as-Judge)":
    st.header("🧑‍⚖️ Generation Quality — LLM-as-Judge")

    st.markdown("""
    Use an LLM to automatically evaluate the quality of RAG-generated answers.
    This is the standard enterprise approach — the same LLM (or a stronger one) acts as a judge.
    """)

    tab_learn, tab_eval = st.tabs(["Learn", "Run Evaluation"])

    with tab_learn:
        st.markdown("""
        #### Evaluation Dimensions

        | Dimension | What It Measures | Why It Matters |
        |-----------|-----------------|----------------|
        | **Faithfulness** | Is every claim in the answer supported by the context? | Catches hallucinations |
        | **Answer Relevance** | Does the answer actually address the question? | Catches off-topic responses |
        | **Completeness** | Does the answer cover all aspects of the question? | Catches partial answers |
        | **Context Utilisation** | Does the answer use the provided context effectively? | Catches context ignoring |
        | **Coherence** | Is the answer well-structured and easy to understand? | Quality of presentation |

        ---
        #### How LLM-as-Judge Works

        1. The judge LLM receives the **question**, **context**, and **generated answer**
        2. For each dimension, it scores on a scale of 1–5 with a justification
        3. Scores are aggregated into an overall quality score
        4. Flagged issues (score ≤ 2) are highlighted for review

        This is the same pattern used by RAGAS, DeepEval, and enterprise RAG evaluation frameworks.
        """)

    with tab_eval:
        st.markdown("Provide a question, context, and answer to evaluate.")

        eval_context = st.text_area(
            "Context (the chunks retrieved by your pipeline):",
            value=(
                "RAG stands for Retrieval-Augmented Generation. It combines information retrieval "
                "with text generation to produce grounded answers. The retrieval step finds relevant "
                "documents from a knowledge base, and the generation step uses those documents as "
                "context for the language model."
            ),
            height=120,
            key="judge_context",
        )
        eval_question = st.text_input(
            "Question:", "What is RAG and how does it work?", key="judge_question",
        )
        eval_answer = st.text_area(
            "Generated Answer (paste or type the answer to evaluate):",
            value=(
                "RAG, or Retrieval-Augmented Generation, is a technique that combines retrieval systems "
                "with language models. It works by first retrieving relevant documents from a knowledge "
                "base, then passing those documents as context to the LLM to generate a grounded answer."
            ),
            height=120,
            key="judge_answer",
        )

        if st.button("🧑‍⚖️ Run LLM-as-Judge Evaluation", type="primary", key="judge_run"):
            if not eval_context or not eval_question or not eval_answer:
                st.warning("Please fill in all three fields.")
            else:
                with st.spinner("LLM is judging the answer..."):
                    try:
                        llm = get_llm_provider()

                        dimensions = {
                            "Faithfulness": (
                                "Is every claim in the ANSWER supported by evidence in the CONTEXT? "
                                "Score 1 if heavily hallucinated, 5 if fully grounded."
                            ),
                            "Answer Relevance": (
                                "Does the ANSWER directly address the QUESTION? "
                                "Score 1 if completely off-topic, 5 if perfectly on-topic."
                            ),
                            "Completeness": (
                                "Does the ANSWER cover all aspects of the QUESTION given the CONTEXT? "
                                "Score 1 if it only addresses a tiny part, 5 if fully comprehensive."
                            ),
                            "Context Utilisation": (
                                "Does the ANSWER make effective use of the information in the CONTEXT? "
                                "Score 1 if context is ignored, 5 if thoroughly used."
                            ),
                            "Coherence": (
                                "Is the ANSWER well-structured, clear, and easy to understand? "
                                "Score 1 if confusing or incoherent, 5 if excellently written."
                            ),
                        }

                        scores = {}
                        justifications = {}

                        for dim_name, criteria in dimensions.items():
                            prompt = (
                                f"You are an expert evaluator for RAG systems. Evaluate the following on the dimension: {dim_name}.\n\n"
                                f"Criteria: {criteria}\n\n"
                                f"CONTEXT:\n{eval_context}\n\n"
                                f"QUESTION: {eval_question}\n\n"
                                f"ANSWER:\n{eval_answer}\n\n"
                                f"Respond in EXACTLY this format (no other text):\n"
                                f"SCORE: <number 1-5>\n"
                                f"JUSTIFICATION: <one sentence>"
                            )
                            response = llm.generate(prompt, temperature=0.0, max_tokens=100)
                            text = response.text.strip()

                            score = 3
                            justification = text
                            for line in text.split("\n"):
                                line_stripped = line.strip()
                                if line_stripped.upper().startswith("SCORE:"):
                                    try:
                                        score = int("".join(c for c in line_stripped.split(":", 1)[1] if c.isdigit())[:1])
                                        score = max(1, min(5, score))
                                    except (ValueError, IndexError):
                                        pass
                                elif line_stripped.upper().startswith("JUSTIFICATION:"):
                                    justification = line_stripped.split(":", 1)[1].strip()

                            scores[dim_name] = score
                            justifications[dim_name] = justification

                        overall = sum(scores.values()) / len(scores)

                        st.markdown("---")
                        st.subheader("Evaluation Results")

                        # Overall score with colour
                        if overall >= 4.0:
                            st.success(f"**Overall Score: {overall:.1f} / 5.0** — Excellent")
                        elif overall >= 3.0:
                            st.warning(f"**Overall Score: {overall:.1f} / 5.0** — Good, room for improvement")
                        else:
                            st.error(f"**Overall Score: {overall:.1f} / 5.0** — Needs significant improvement")

                        # Dimension breakdown
                        score_cols = st.columns(len(dimensions))
                        for i, (dim_name, score) in enumerate(scores.items()):
                            with score_cols[i]:
                                delta_color = "normal" if score >= 3 else "inverse"
                                st.metric(dim_name, f"{score}/5", delta=f"{'Good' if score >= 4 else 'OK' if score >= 3 else 'Low'}", delta_color=delta_color)

                        # Detailed justifications
                        st.markdown("---")
                        st.markdown("#### Detailed Justifications")
                        for dim_name in dimensions:
                            icon = "✅" if scores[dim_name] >= 4 else "⚠️" if scores[dim_name] >= 3 else "❌"
                            st.markdown(f"{icon} **{dim_name}** ({scores[dim_name]}/5): {justifications[dim_name]}")

                        # Improvement suggestions
                        low_dims = [d for d, s in scores.items() if s <= 2]
                        if low_dims:
                            st.markdown("---")
                            st.subheader("🔧 Suggested Improvements")
                            suggestions = {
                                "Faithfulness": "Enable the **Hallucination Check** guardrail, or use **Self-Reflect** reasoning mode to catch unsupported claims.",
                                "Answer Relevance": "Try **Chain-of-Thought** reasoning to keep the LLM focused, or enable the **Relevance Check** output guardrail.",
                                "Completeness": "Increase retrieval K to provide more context, or use **Multi-Query** retrieval to capture different aspects.",
                                "Context Utilisation": "Switch to **Analysis** or **CoT + Analysis** reasoning mode to force the LLM to examine each chunk.",
                                "Coherence": "Lower the temperature setting, or use **Self-Reflect** mode for iterative refinement.",
                            }
                            for dim in low_dims:
                                st.info(f"**{dim}:** {suggestions.get(dim, 'Review pipeline configuration.')}")

                    except Exception as e:
                        st.error(f"Evaluation error: {e}")
                        st.info("Make sure you have an LLM provider configured in the sidebar.")

# ═══════════════════════════════════════════════════════════════════════════
# END-TO-END PIPELINE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "End-to-End Pipeline Evaluation":
    st.header("🔬 End-to-End Pipeline Evaluation")

    st.markdown("""
    Evaluate your full RAG pipeline against a test suite — a set of questions
    with known correct answers. This is the gold standard for measuring pipeline quality.
    """)

    tab_learn, tab_run = st.tabs(["Learn", "Run Test Suite"])

    with tab_learn:
        st.markdown("""
        #### Test Suite Structure

        A test suite consists of **evaluation samples**, each with:

        | Field | Description |
        |-------|-------------|
        | **Question** | The user's query |
        | **Expected Answer** | The ground-truth answer (what a perfect response would say) |
        | **Relevant Chunks** *(optional)* | Which documents/chunks should be retrieved |

        #### Automated Scoring

        For each sample, the pipeline runs end-to-end and the judge LLM scores:
        1. **Answer Correctness** — does the generated answer match the expected answer?
        2. **Faithfulness** — is the answer grounded in retrieved context?
        3. **Retrieval Quality** — were the right chunks retrieved?

        #### Enterprise Evaluation Workflow

        ```
        Test Suite (N questions + expected answers)
              │
              ▼
        Run Pipeline on each question
              │
              ▼
        LLM Judge scores each answer
              │
              ▼
        Aggregate metrics + failure analysis
              │
              ▼
        Identify weak spots → tune pipeline
        ```
        """)

    with tab_run:
        st.markdown("Define your test cases below, then run the evaluation.")

        default_suite = (
            "What is RAG? | RAG is Retrieval-Augmented Generation, combining retrieval with LLM generation.\n"
            "How do embeddings work? | Embeddings convert text into numerical vectors that capture semantic meaning.\n"
            "What is cosine similarity? | Cosine similarity measures the angle between two vectors, used to compare embeddings."
        )

        suite_text = st.text_area(
            "Test suite (one per line: `question | expected answer`):",
            value=default_suite,
            height=150,
            key="e2e_suite",
        )

        knowledge = st.text_area(
            "Knowledge base (one fact per line):",
            value=(
                "RAG stands for Retrieval-Augmented Generation.\n"
                "RAG combines retrieval systems with generative language models.\n"
                "Embeddings are numerical vector representations of text that capture semantic meaning.\n"
                "Cosine similarity measures the angle between two vectors and is used to compare embeddings.\n"
                "Vector stores enable fast similarity search over embeddings.\n"
                "Chunking splits documents into smaller pieces for processing."
            ),
            height=150,
            key="e2e_knowledge",
        )

        e2e_col1, e2e_col2 = st.columns(2)
        with e2e_col1:
            e2e_strategy = st.selectbox("Retrieval strategy:", ["similarity", "mmr", "hybrid"], key="e2e_strat")
        with e2e_col2:
            e2e_reasoning = st.selectbox("Reasoning mode:", ["standard", "cot", "analysis"], key="e2e_reason")

        if st.button("🚀 Run End-to-End Evaluation", type="primary", key="e2e_run"):
            test_cases = []
            for line in suite_text.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 1)
                    test_cases.append({"question": parts[0].strip(), "expected": parts[1].strip()})

            if not test_cases:
                st.warning("Add at least one test case in `question | expected answer` format.")
            else:
                docs = [d.strip() for d in knowledge.strip().split("\n") if d.strip()]

                with st.spinner(f"Running {len(test_cases)} test cases..."):
                    try:
                        from core.embeddings import create_embeddings
                        from core.vector_store import NumpyVectorStore
                        from core.rag_pipeline import RAGPipeline

                        cfg = st.session_state.get("provider_config", {})
                        embed = get_embedding_provider()
                        llm = get_llm_provider()

                        pipeline = RAGPipeline(
                            embedding_provider=embed,
                            vector_store=NumpyVectorStore(),
                            llm_provider=llm,
                            chunk_strategy="sentence",
                            chunk_kwargs={"max_sentences": 2, "overlap_sentences": 0},
                            retrieval_k=3,
                            retrieval_strategy=e2e_strategy,
                            reasoning_mode=e2e_reasoning,
                        )
                        pipeline.ingest("\n".join(docs))

                        results_data = []
                        progress = st.progress(0.0, text="Evaluating...")

                        for idx, tc in enumerate(test_cases):
                            progress.progress((idx + 1) / len(test_cases), text=f"Evaluating {idx+1}/{len(test_cases)}: {tc['question'][:40]}...")

                            result = pipeline.query(tc["question"])

                            judge_prompt = (
                                f"You are evaluating a RAG system. Compare the GENERATED answer to the EXPECTED answer.\n\n"
                                f"QUESTION: {tc['question']}\n\n"
                                f"EXPECTED ANSWER: {tc['expected']}\n\n"
                                f"GENERATED ANSWER: {result.answer}\n\n"
                                f"CONTEXT PROVIDED TO THE LLM:\n{chr(10).join(c.text for c in result.retrieved_chunks)}\n\n"
                                f"Score each dimension 1-5. Respond in EXACTLY this format:\n"
                                f"CORRECTNESS: <1-5>\n"
                                f"FAITHFULNESS: <1-5>\n"
                                f"REASONING: <one sentence summary>"
                            )
                            judge_resp = llm.generate(judge_prompt, temperature=0.0, max_tokens=150)

                            correctness, faithfulness = 3, 3
                            reasoning = judge_resp.text.strip()
                            for line in judge_resp.text.strip().split("\n"):
                                ls = line.strip().upper()
                                if ls.startswith("CORRECTNESS:"):
                                    try:
                                        correctness = max(1, min(5, int("".join(c for c in ls.split(":", 1)[1] if c.isdigit())[:1])))
                                    except (ValueError, IndexError):
                                        pass
                                elif ls.startswith("FAITHFULNESS:"):
                                    try:
                                        faithfulness = max(1, min(5, int("".join(c for c in ls.split(":", 1)[1] if c.isdigit())[:1])))
                                    except (ValueError, IndexError):
                                        pass
                                elif ls.startswith("REASONING:"):
                                    reasoning = line.strip().split(":", 1)[1].strip()

                            results_data.append({
                                "question": tc["question"],
                                "expected": tc["expected"],
                                "generated": result.answer,
                                "correctness": correctness,
                                "faithfulness": faithfulness,
                                "reasoning": reasoning,
                                "chunks": [c.text[:80] for c in result.retrieved_chunks],
                                "duration_ms": result.total_duration_ms,
                            })

                        progress.empty()

                        # Aggregate
                        avg_correct = np.mean([r["correctness"] for r in results_data])
                        avg_faithful = np.mean([r["faithfulness"] for r in results_data])
                        avg_overall = (avg_correct + avg_faithful) / 2
                        pass_rate = sum(1 for r in results_data if r["correctness"] >= 4 and r["faithfulness"] >= 4) / len(results_data)
                        avg_latency = np.mean([r["duration_ms"] for r in results_data])

                        st.markdown("---")
                        st.subheader("Aggregate Results")

                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Avg Correctness", f"{avg_correct:.1f}/5")
                        m2.metric("Avg Faithfulness", f"{avg_faithful:.1f}/5")
                        m3.metric("Overall Score", f"{avg_overall:.1f}/5")
                        m4.metric("Pass Rate (≥4/5)", f"{pass_rate:.0%}")
                        m5.metric("Avg Latency", f"{avg_latency:.0f}ms")

                        if avg_overall >= 4.0:
                            st.success(f"Pipeline is performing well with **{e2e_strategy}** retrieval and **{e2e_reasoning}** reasoning.")
                        elif avg_overall >= 3.0:
                            st.warning("Pipeline is decent but has room for improvement. Try different strategies in the A/B comparison.")
                        else:
                            st.error("Pipeline needs significant tuning. Check individual failures below.")

                        # Per-question results
                        st.markdown("---")
                        st.subheader("Per-Question Breakdown")

                        for i, r in enumerate(results_data):
                            icon = "✅" if r["correctness"] >= 4 and r["faithfulness"] >= 4 else "⚠️" if r["correctness"] >= 3 else "❌"
                            with st.expander(f"{icon} Q{i+1}: {r['question']} — Correctness: {r['correctness']}/5, Faithfulness: {r['faithfulness']}/5"):
                                st.markdown(f"**Expected:** {r['expected']}")
                                st.markdown(f"**Generated:** {r['generated']}")
                                st.markdown(f"**Judge:** {r['reasoning']}")
                                st.caption(f"Latency: {r['duration_ms']:.0f}ms | Chunks: {len(r['chunks'])}")

                    except Exception as e:
                        st.error(f"Evaluation error: {e}")
                        st.info("Make sure you have an LLM and embedding provider configured in the sidebar.")

# ═══════════════════════════════════════════════════════════════════════════
# A/B STRATEGY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
elif section == "A/B Strategy Comparison":
    st.header("⚔️ A/B Strategy Comparison")

    st.markdown("""
    Compare two different pipeline configurations side by side on the same query.
    See how retrieval strategy, reasoning mode, and other settings affect answer quality.
    """)

    knowledge_ab = st.text_area(
        "Knowledge base (shared for both pipelines):",
        value=(
            "RAG stands for Retrieval-Augmented Generation.\n"
            "RAG combines retrieval systems with generative language models.\n"
            "The retrieval step finds relevant documents from a knowledge base.\n"
            "The generation step uses retrieved documents as context for the LLM.\n"
            "Embeddings convert text into numerical vectors that capture meaning.\n"
            "Vector stores like ChromaDB enable fast similarity search.\n"
            "Hybrid search combines keyword (BM25) and semantic approaches.\n"
            "Chain-of-thought prompting makes LLMs reason step by step.\n"
            "Reranking uses a second model to re-score retrieved chunks.\n"
            "MMR balances relevance with diversity in retrieval results."
        ),
        height=160,
        key="ab_knowledge",
    )

    ab_query = st.text_input("Query (same for both):", "How does RAG improve answer quality?", key="ab_query")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Pipeline A")
        a_retrieval = st.selectbox("Retrieval:", ["similarity", "mmr", "hybrid"], key="ab_a_ret")
        a_reasoning = st.selectbox("Reasoning:", ["standard", "cot", "analysis", "cot_analysis", "step_back", "self_reflect"], key="ab_a_reas")
        a_k = st.slider("Top K:", 1, 10, 3, key="ab_a_k")

    with col_b:
        st.markdown("### Pipeline B")
        b_retrieval = st.selectbox("Retrieval:", ["similarity", "mmr", "hybrid"], index=2, key="ab_b_ret")
        b_reasoning = st.selectbox("Reasoning:", ["standard", "cot", "analysis", "cot_analysis", "step_back", "self_reflect"], index=1, key="ab_b_reas")
        b_k = st.slider("Top K:", 1, 10, 5, key="ab_b_k")

    if ab_query and st.button("⚔️ Run A/B Comparison", type="primary", key="ab_run"):
        docs_ab = [d.strip() for d in knowledge_ab.strip().split("\n") if d.strip()]

        with st.spinner("Running both pipelines..."):
            try:
                from core.rag_pipeline import RAGPipeline
                from core.vector_store import NumpyVectorStore

                embed = get_embedding_provider()
                llm = get_llm_provider()

                configs = [
                    {"label": "A", "retrieval": a_retrieval, "reasoning": a_reasoning, "k": a_k},
                    {"label": "B", "retrieval": b_retrieval, "reasoning": b_reasoning, "k": b_k},
                ]

                ab_results = []
                for c in configs:
                    p = RAGPipeline(
                        embedding_provider=embed,
                        vector_store=NumpyVectorStore(),
                        llm_provider=llm,
                        chunk_strategy="sentence",
                        chunk_kwargs={"max_sentences": 2, "overlap_sentences": 0},
                        retrieval_k=c["k"],
                        retrieval_strategy=c["retrieval"],
                        reasoning_mode=c["reasoning"],
                    )
                    p.ingest("\n".join(docs_ab))
                    result = p.query(ab_query)
                    ab_results.append({"config": c, "result": result})

                # LLM judge comparison
                judge_prompt = (
                    f"You are comparing two RAG pipeline answers to the same question.\n\n"
                    f"QUESTION: {ab_query}\n\n"
                    f"ANSWER A ({a_retrieval} + {a_reasoning}):\n{ab_results[0]['result'].answer}\n\n"
                    f"ANSWER B ({b_retrieval} + {b_reasoning}):\n{ab_results[1]['result'].answer}\n\n"
                    f"Compare the two answers on: accuracy, completeness, clarity, and groundedness.\n"
                    f"Respond in EXACTLY this format:\n"
                    f"WINNER: A or B or TIE\n"
                    f"SCORE_A: <1-5>\n"
                    f"SCORE_B: <1-5>\n"
                    f"REASONING: <2-3 sentences explaining the comparison>"
                )
                judge_resp = llm.generate(judge_prompt, temperature=0.0, max_tokens=200)

                winner = "TIE"
                score_a, score_b = 3, 3
                judge_reasoning = judge_resp.text.strip()

                for line in judge_resp.text.strip().split("\n"):
                    ls = line.strip().upper()
                    if ls.startswith("WINNER:"):
                        w = ls.split(":", 1)[1].strip()
                        if "A" in w and "B" not in w:
                            winner = "A"
                        elif "B" in w and "A" not in w:
                            winner = "B"
                        else:
                            winner = "TIE"
                    elif ls.startswith("SCORE_A:"):
                        try:
                            score_a = max(1, min(5, int("".join(c for c in ls.split(":", 1)[1] if c.isdigit())[:1])))
                        except (ValueError, IndexError):
                            pass
                    elif ls.startswith("SCORE_B:"):
                        try:
                            score_b = max(1, min(5, int("".join(c for c in ls.split(":", 1)[1] if c.isdigit())[:1])))
                        except (ValueError, IndexError):
                            pass
                    elif ls.startswith("REASONING:"):
                        judge_reasoning = line.strip().split(":", 1)[1].strip()

                # Display results
                st.markdown("---")

                if winner == "A":
                    st.success(f"**Winner: Pipeline A** ({a_retrieval} + {a_reasoning})")
                elif winner == "B":
                    st.success(f"**Winner: Pipeline B** ({b_retrieval} + {b_reasoning})")
                else:
                    st.info("**Result: Tie** — both pipelines produced comparable answers.")

                st.markdown(f"*Judge:* {judge_reasoning}")

                st.markdown("---")

                col_ra, col_rb = st.columns(2)

                with col_ra:
                    st.markdown(f"### Pipeline A {'🏆' if winner == 'A' else ''}")
                    st.metric("Score", f"{score_a}/5")
                    st.caption(f"{a_retrieval} retrieval | {a_reasoning} reasoning | K={a_k}")
                    st.markdown(f"**Answer:**\n\n{ab_results[0]['result'].answer}")
                    st.caption(f"Latency: {ab_results[0]['result'].total_duration_ms:.0f}ms")

                    with st.expander("Retrieved chunks"):
                        for i, c in enumerate(ab_results[0]["result"].retrieved_chunks):
                            st.markdown(f"#{i+1} ({c.score:.3f}): {c.text[:120]}...")

                with col_rb:
                    st.markdown(f"### Pipeline B {'🏆' if winner == 'B' else ''}")
                    st.metric("Score", f"{score_b}/5")
                    st.caption(f"{b_retrieval} retrieval | {b_reasoning} reasoning | K={b_k}")
                    st.markdown(f"**Answer:**\n\n{ab_results[1]['result'].answer}")
                    st.caption(f"Latency: {ab_results[1]['result'].total_duration_ms:.0f}ms")

                    with st.expander("Retrieved chunks"):
                        for i, c in enumerate(ab_results[1]["result"].retrieved_chunks):
                            st.markdown(f"#{i+1} ({c.score:.3f}): {c.text[:120]}...")

            except Exception as e:
                st.error(f"Comparison error: {e}")
                st.info("Make sure you have an LLM and embedding provider configured in the sidebar.")

# ── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/1_📚_RAG/06_🔬_Full_Pipeline.py", label="← Full Pipeline", icon="🔬")
with col2:
    st.page_link("pages/1_📚_RAG/08_🎮_Playground.py", label="Next: Playground →", icon="🎮")
