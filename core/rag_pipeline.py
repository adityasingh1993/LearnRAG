"""
End-to-end RAG pipeline that connects all components.
Tracks each step for educational visibility.

Supports:
  - 8 chunking strategies
  - 5 retrieval strategies (similarity, MMR, hybrid, multi-query, HyDE)
  - 6 reasoning modes  (standard, CoT, analysis, CoT+analysis, step-back, self-reflect)
  - LLM reranking
  - Optional input / output guardrails
  - Conversation context management (6 strategies)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
import time
import json
import re

from core.chunking import Chunk, chunk_text
from core.embeddings import EmbeddingProvider
from core.vector_store import VectorStore, SearchResult
from core.llm_providers import LLMProvider, LLMResponse

if TYPE_CHECKING:
    from core.context import ContextManager


# ═════════════════════════════════════════════════════════════════════════
#  Reasoning modes & prompt templates
# ═════════════════════════════════════════════════════════════════════════

REASONING_MODES = {
    "standard":      "Direct answer from retrieved context",
    "cot":           "Chain-of-Thought: step-by-step reasoning before the final answer",
    "analysis":      "Analyse each chunk's relevance first, then synthesise an answer",
    "cot_analysis":  "CoT + Analysis combined for maximum depth",
    "step_back":     "Step-back prompting: answer a broader question first for context",
    "self_reflect":  "Generate, critique, and refine the answer in one pass",
}

_HISTORY_BLOCK = (
    "{history}\n\n"
)

_PROMPT_STANDARD = (
    "You are a helpful assistant. Answer the user's question based ONLY on the "
    "provided context. If the context doesn't contain the answer, say so clearly.\n\n"
    "{history}"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

_PROMPT_COT = (
    "You are a helpful assistant. Answer the user's question based ONLY on the "
    "provided context.\n\n"
    "{history}"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step before answering:\n"
    "1. First, identify which pieces of context are relevant.\n"
    "2. Then, reason through the information logically.\n"
    "3. Finally, provide a clear, well-structured answer.\n\n"
    "## Step-by-step reasoning"
)

_PROMPT_ANALYSIS = (
    "You are a helpful assistant. You will first analyse the relevance of each "
    "retrieved chunk, then synthesise a final answer.\n\n"
    "{history}"
    "Context chunks:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. For EACH chunk above, write a brief relevance verdict "
    "(Highly Relevant / Somewhat Relevant / Not Relevant) and why.\n"
    "2. After the analysis, write a section titled '## Answer' with "
    "your final answer using only the relevant chunks.\n\n"
    "## Chunk Analysis"
)

_PROMPT_COT_ANALYSIS = (
    "You are a helpful assistant. You will analyse, reason step by step, "
    "and then answer.\n\n"
    "{history}"
    "Context chunks:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. **Chunk Analysis** — For each chunk, assess its relevance to the question.\n"
    "2. **Chain of Thought** — Using only the relevant chunks, reason step by step "
    "toward the answer.\n"
    "3. **Answer** — Provide a clear, final answer.\n\n"
    "## Chunk Analysis"
)

_PROMPT_STEP_BACK = (
    "You are a helpful assistant using step-back prompting.\n\n"
    "{history}"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. **Step Back** — First, identify a broader, more general question that "
    "provides useful background for answering the specific question.\n"
    "2. **Background** — Answer the broader question using the context.\n"
    "3. **Final Answer** — Now answer the original specific question, informed by "
    "the background you just established.\n\n"
    "## Step-back question"
)

_PROMPT_SELF_REFLECT = (
    "You are a helpful assistant using self-reflection.\n\n"
    "{history}"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. **Draft Answer** — Write an initial answer based on the context.\n"
    "2. **Critique** — Review your draft. Identify any weaknesses: "
    "unsupported claims, missing nuances, or logical gaps.\n"
    "3. **Refined Answer** — Write an improved final answer that addresses "
    "the critique. Mark this section '## Final Answer'.\n\n"
    "## Draft Answer"
)

_RERANK_PROMPT = (
    "You are a relevance judge. Score each document chunk on how well it answers "
    "the given question. Return ONLY valid JSON — an array of objects with "
    '"index" (int) and "score" (float 0-10).\n\n'
    "Question: {question}\n\n"
    "Chunks:\n{chunks}\n\n"
    "JSON scores:"
)

PROMPT_TEMPLATES = {
    "standard":     _PROMPT_STANDARD,
    "cot":          _PROMPT_COT,
    "analysis":     _PROMPT_ANALYSIS,
    "cot_analysis": _PROMPT_COT_ANALYSIS,
    "step_back":    _PROMPT_STEP_BACK,
    "self_reflect": _PROMPT_SELF_REFLECT,
}


# ═════════════════════════════════════════════════════════════════════════
#  Data classes
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineStep:
    name: str
    input_data: Any
    output_data: Any
    duration_ms: float
    details: dict = field(default_factory=dict)


@dataclass
class GuardrailReport:
    passed: bool
    results: list  # list[GuardrailResult]


@dataclass
class RAGResult:
    answer: str
    query: str
    retrieved_chunks: list[SearchResult]
    steps: list[PipelineStep]
    total_duration_ms: float
    reasoning_mode: str = "standard"
    input_guardrails: GuardrailReport | None = None
    output_guardrails: GuardrailReport | None = None
    token_usage: Any = None  # TurnTokenUsage


# ═════════════════════════════════════════════════════════════════════════
#  Pipeline
# ═════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Configurable RAG pipeline with step-by-step tracking.
    Each step records inputs, outputs, and timing for educational purposes.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        chunk_strategy: str = "recursive",
        chunk_kwargs: dict | None = None,
        retrieval_k: int = 3,
        retrieval_strategy: str = "similarity",
        system_prompt: str | None = None,
        reasoning_mode: str = "standard",
        use_reranking: bool = False,
        rerank_top_n: int | None = None,
        input_guardrails: list | None = None,
        output_guardrails: list | None = None,
        guardrail_mode: str = "warn",
        context_manager: "ContextManager | None" = None,
        # legacy compat
        use_mmr: bool = False,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.chunk_strategy = chunk_strategy
        self.chunk_kwargs = chunk_kwargs or {}
        self.retrieval_k = retrieval_k

        if use_mmr and retrieval_strategy == "similarity":
            retrieval_strategy = "mmr"
        self.retrieval_strategy = retrieval_strategy

        self.reasoning_mode = reasoning_mode if reasoning_mode in PROMPT_TEMPLATES else "standard"
        self.use_reranking = use_reranking
        self.rerank_top_n = rerank_top_n
        self.system_prompt = system_prompt or PROMPT_TEMPLATES[self.reasoning_mode]
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.guardrail_mode = guardrail_mode  # "warn" or "block"
        self.context_manager = context_manager
        self._steps: list[PipelineStep] = []

    # ── helpers ────────────────────────────────────────────────────────

    def _track(self, name: str, input_data: Any, func, **details) -> Any:
        start = time.perf_counter()
        result = func()
        duration = (time.perf_counter() - start) * 1000
        self._steps.append(PipelineStep(
            name=name, input_data=input_data, output_data=result,
            duration_ms=round(duration, 2), details=details,
        ))
        return result

    # ── ingest ─────────────────────────────────────────────────────────

    def ingest(self, text: str, source: str = "uploaded") -> list[Chunk]:
        self._steps = []

        chunks = self._track(
            "Chunking", f"{len(text)} chars of text",
            lambda: chunk_text(text, self.chunk_strategy, **self.chunk_kwargs),
            strategy=self.chunk_strategy,
        )

        chunk_texts = [c.text for c in chunks]
        embeddings = self._track(
            "Embedding", f"{len(chunks)} chunks",
            lambda: self.embedding_provider.embed(chunk_texts),
            provider=self.embedding_provider.name(),
            dimension=self.embedding_provider.dimension(),
        )

        metadatas = [{"source": source, "chunk_index": c.index, **(c.metadata or {})} for c in chunks]
        self._track(
            "Storing", f"{len(chunks)} embeddings (dim={embeddings.shape[1]})",
            lambda: self.vector_store.add(chunk_texts, embeddings, metadatas),
            store_size_after=self.vector_store.count() + len(chunks),
        )
        return chunks

    # ── retrieval dispatch ─────────────────────────────────────────────

    def _retrieve(self, question: str, query_embedding):
        from core.retrieval import hybrid_search, multi_query_search, hyde_search

        fetch_k = self.retrieval_k
        if self.use_reranking:
            fetch_k = max(self.retrieval_k * 3, 10)

        strategy = self.retrieval_strategy

        if strategy == "mmr" and hasattr(self.vector_store, "search_mmr"):
            return self._track(
                "Retrieval (MMR)", f"k={fetch_k}",
                lambda: self.vector_store.search_mmr(query_embedding, k=fetch_k),
                method="MMR",
            )
        elif strategy == "hybrid":
            return self._track(
                "Retrieval (Hybrid)", f"k={fetch_k}",
                lambda: hybrid_search(question, query_embedding, self.vector_store, k=fetch_k),
                method="hybrid_rrf",
            )
        elif strategy == "multi_query":
            return self._track(
                "Retrieval (Multi-Query)", f"k={fetch_k}",
                lambda: multi_query_search(
                    question, self.embedding_provider, self.vector_store,
                    self.llm_provider, k=fetch_k,
                ),
                method="multi_query_rrf",
            )
        elif strategy == "hyde":
            return self._track(
                "Retrieval (HyDE)", f"k={fetch_k}",
                lambda: hyde_search(
                    question, self.embedding_provider, self.vector_store,
                    self.llm_provider, k=fetch_k,
                ),
                method="hyde",
            )
        else:
            return self._track(
                "Retrieval", f"k={fetch_k}",
                lambda: self.vector_store.search(query_embedding, k=fetch_k),
                method="cosine_similarity",
            )

    # ── reranking ──────────────────────────────────────────────────────

    def _rerank_with_llm(self, question: str, results: list[SearchResult]) -> list[SearchResult]:
        chunks_text = "\n\n".join(
            f"[Chunk {i}] {r.text[:500]}" for i, r in enumerate(results)
        )
        prompt = _RERANK_PROMPT.format(question=question, chunks=chunks_text)
        response = self.llm_provider.generate(prompt, temperature=0.0, max_tokens=1024)

        try:
            raw = response.text.strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            scores = json.loads(match.group()) if match else json.loads(raw)
            score_map = {item["index"]: float(item["score"]) for item in scores}
            for r_idx, r in enumerate(results):
                r.score = score_map.get(r_idx, r.score)
            results.sort(key=lambda r: r.score, reverse=True)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return results

    # ── guardrails ─────────────────────────────────────────────────────

    def _run_input_guardrails(self, query: str) -> GuardrailReport | None:
        if not self.input_guardrails:
            return None
        from core.guardrails import run_input_guardrails
        results = run_input_guardrails(
            self.input_guardrails, query, llm_provider=self.llm_provider,
        )
        self._steps.append(PipelineStep(
            name="Input Guardrails",
            input_data=query,
            output_data=results,
            duration_ms=0,
            details={"count": len(results), "all_passed": all(r.passed for r in results)},
        ))
        return GuardrailReport(passed=all(r.passed for r in results), results=results)

    def _run_output_guardrails(self, query: str, context: str, answer: str) -> GuardrailReport | None:
        if not self.output_guardrails:
            return None
        from core.guardrails import run_output_guardrails
        results = run_output_guardrails(
            self.output_guardrails, query, context, answer, llm_provider=self.llm_provider,
        )
        self._steps.append(PipelineStep(
            name="Output Guardrails",
            input_data=f"answer ({len(answer)} chars)",
            output_data=results,
            duration_ms=0,
            details={"count": len(results), "all_passed": all(r.passed for r in results)},
        ))
        return GuardrailReport(passed=all(r.passed for r in results), results=results)

    # ── main query ─────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResult:
        from core.token_tracker import TurnTokenUsage, count_tokens

        self._steps = []
        turn_tokens = TurnTokenUsage()
        total_start = time.perf_counter()

        # Input guardrails
        input_report = self._run_input_guardrails(question)
        if input_report and not input_report.passed and self.guardrail_mode == "block":
            failed = [r for r in input_report.results if not r.passed]
            reasons = "; ".join(r.reason for r in failed)
            return RAGResult(
                answer=f"**Blocked by input guardrails:** {reasons}",
                query=question, retrieved_chunks=[], steps=self._steps,
                total_duration_ms=round((time.perf_counter() - total_start) * 1000, 2),
                reasoning_mode=self.reasoning_mode,
                input_guardrails=input_report,
                token_usage=turn_tokens,
            )

        # Embed query
        q_tok = count_tokens(question)
        query_embedding = self._track(
            "Query Embedding", question,
            lambda: self.embedding_provider.embed_query(question),
            provider=self.embedding_provider.name(),
            tokens=q_tok,
        )
        turn_tokens.add("Query Embedding", embedding=q_tok)

        # Retrieve
        results = self._retrieve(question, query_embedding)

        # Rerank
        if self.use_reranking and results:
            rerank_input_tok = count_tokens(question) + sum(count_tokens(r.text[:500]) for r in results)
            results = self._track(
                "LLM Reranking", f"Reranking {len(results)} chunks",
                lambda: self._rerank_with_llm(question, results),
                model=self.llm_provider.name(),
                tokens=rerank_input_tok,
            )
            rerank_resp = self._steps[-1].output_data
            rerank_usage = getattr(rerank_resp, "usage", None) if hasattr(rerank_resp, "usage") else None
            if isinstance(rerank_usage, dict):
                turn_tokens.add("LLM Reranking",
                                prompt=rerank_usage.get("prompt_tokens", rerank_input_tok),
                                completion=rerank_usage.get("completion_tokens", 0))
            else:
                turn_tokens.add("LLM Reranking", prompt=rerank_input_tok, completion=50)
            results = results[:self.rerank_top_n or self.retrieval_k]

        # Build context
        context = "\n\n---\n\n".join(
            f"[Chunk {i+1} | Score: {r.score:.3f}]\n{r.text}"
            for i, r in enumerate(results)
        )
        context_tok = count_tokens(context)

        # Build conversation history
        history_str = ""
        history_tok = 0
        if self.context_manager is not None:
            from core.context import RelevantHistoryContext
            if isinstance(self.context_manager, RelevantHistoryContext):
                history_str = self.context_manager.get_context_string(query=question)
            else:
                history_str = self.context_manager.get_context_string()
            if history_str:
                history_tok = count_tokens(history_str)
                history_str = f"Conversation history:\n{history_str}\n\n"
                self._steps.append(PipelineStep(
                    name="Context (History)",
                    input_data=f"{self.context_manager.turn_count} turns",
                    output_data=history_str,
                    duration_ms=0,
                    details={"strategy": self.context_manager.strategy_name,
                             "history_chars": len(history_str),
                             "tokens": history_tok},
                ))
                turn_tokens.add("Context (History)", prompt=history_tok)

        # Generate
        active_prompt = self.system_prompt
        if "{context}" not in active_prompt:
            active_prompt = PROMPT_TEMPLATES.get(self.reasoning_mode, _PROMPT_STANDARD)
        prompt = active_prompt.format(context=context, question=question, history=history_str)
        prompt_tok = count_tokens(prompt)

        gen_label = f"Generation ({self.reasoning_mode.upper()})"
        response = self._track(
            gen_label, f"Prompt with {len(results)} chunks | mode={self.reasoning_mode}",
            lambda: self.llm_provider.generate(prompt),
            model=self.llm_provider.name(),
            context_length=len(context),
            reasoning_mode=self.reasoning_mode,
            prompt_tokens=prompt_tok,
        )

        # Extract actual usage from LLM response if available
        gen_prompt = prompt_tok
        gen_completion = count_tokens(response.text)
        if response.usage:
            gen_prompt = response.usage.get("prompt_tokens", prompt_tok)
            gen_completion = response.usage.get("completion_tokens", gen_completion)
        self._steps[-1].details["prompt_tokens"] = gen_prompt
        self._steps[-1].details["completion_tokens"] = gen_completion
        self._steps[-1].details["total_tokens"] = gen_prompt + gen_completion
        turn_tokens.add(gen_label, prompt=gen_prompt, completion=gen_completion)

        # Output guardrails
        output_report = self._run_output_guardrails(question, context, response.text)
        answer_text = response.text
        if output_report and not output_report.passed and self.guardrail_mode == "block":
            failed = [r for r in output_report.results if not r.passed]
            reasons = "; ".join(r.reason for r in failed)
            answer_text = f"**Blocked by output guardrails:** {reasons}"

        # Record turn in context manager
        if self.context_manager is not None:
            self.context_manager.add_turn("user", question)
            self.context_manager.add_turn("assistant", answer_text)

        total_duration = (time.perf_counter() - total_start) * 1000

        return RAGResult(
            answer=answer_text,
            query=question,
            retrieved_chunks=results,
            steps=self._steps,
            total_duration_ms=round(total_duration, 2),
            reasoning_mode=self.reasoning_mode,
            input_guardrails=input_report,
            output_guardrails=output_report,
            token_usage=turn_tokens,
        )

    def query_stream(self, question: str):
        query_embedding = self.embedding_provider.embed_query(question)

        if self.retrieval_strategy == "mmr" and hasattr(self.vector_store, "search_mmr"):
            results = self.vector_store.search_mmr(query_embedding, k=self.retrieval_k)
        else:
            results = self.vector_store.search(query_embedding, k=self.retrieval_k)

        context = "\n\n---\n\n".join(
            f"[Chunk {i+1} | Score: {r.score:.3f}]\n{r.text}"
            for i, r in enumerate(results)
        )

        history_str = ""
        if self.context_manager is not None:
            from core.context import RelevantHistoryContext
            if isinstance(self.context_manager, RelevantHistoryContext):
                history_str = self.context_manager.get_context_string(query=question)
            else:
                history_str = self.context_manager.get_context_string()
            if history_str:
                history_str = f"Conversation history:\n{history_str}\n\n"

        prompt = self.system_prompt.format(context=context, question=question, history=history_str)
        if self.context_manager is not None:
            self.context_manager.add_turn("user", question)
        return self.llm_provider.generate_stream(prompt), results

    @property
    def last_steps(self) -> list[PipelineStep]:
        return list(self._steps)
