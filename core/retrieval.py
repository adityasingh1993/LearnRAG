"""
Advanced retrieval strategies that wrap around a base vector store.
  - hybrid     : BM25 (keyword) + semantic search with Reciprocal Rank Fusion
  - multi_query: LLM generates query variations; results merged via RRF
  - hyde       : Hypothetical Document Embeddings — LLM generates a hypothetical
                 answer, embeds it, and uses that for retrieval
"""

from __future__ import annotations

import numpy as np
from core.vector_store import VectorStore, SearchResult
from core.embeddings import EmbeddingProvider
from core.llm_providers import LLMProvider


RETRIEVAL_STRATEGIES = {
    "similarity":   "Cosine similarity — standard nearest-neighbour search",
    "mmr":          "Maximum Marginal Relevance — balances relevance with diversity",
    "hybrid":       "BM25 keyword search + semantic search fused with RRF",
    "multi_query":  "LLM generates 3 query rewrites; results merged via RRF",
    "hyde":         "Hypothetical Document Embeddings — search with a generated answer",
}


# ─── Hybrid search ────────────────────────────────────────────────────────

def _bm25_search(query: str, texts: list[str], k: int = 10) -> list[tuple[int, float]]:
    """Lightweight BM25 approximation using TF-IDF cosine score."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not texts:
        return []
    corpus = list(texts) + [query]
    tfidf = TfidfVectorizer(stop_words="english")
    try:
        matrix = tfidf.fit_transform(corpus)
    except ValueError:
        return []
    scores = (matrix[:-1] @ matrix[-1].T).toarray().flatten()
    top = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top]


def _rrf_merge(*ranked_lists: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked result lists."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    vector_store: VectorStore,
    k: int = 5,
    fetch_k: int = 20,
) -> list[SearchResult]:
    """Semantic + BM25 keyword search fused with Reciprocal Rank Fusion."""
    sem_results = vector_store.search(query_embedding, k=fetch_k)
    texts, _ = vector_store.get_all_embeddings()

    sem_ranked = [(r.index, r.score) for r in sem_results]
    kw_ranked = _bm25_search(query, texts, k=fetch_k)

    fused = _rrf_merge(sem_ranked, kw_ranked)

    out: list[SearchResult] = []
    sem_map = {r.index: r for r in sem_results}
    for idx, rrf_score in fused[:k]:
        if idx in sem_map:
            r = sem_map[idx]
            out.append(SearchResult(text=r.text, score=rrf_score, index=idx, metadata=r.metadata))
        elif 0 <= idx < len(texts):
            out.append(SearchResult(text=texts[idx], score=rrf_score, index=idx))
    return out


# ─── Multi-query retrieval ────────────────────────────────────────────────

_MULTI_QUERY_PROMPT = (
    "You are a search query rewriter. Given the original question, generate exactly "
    "3 alternative search queries that might retrieve useful documents. "
    "Return ONLY the 3 queries, one per line, with no numbering or extra text.\n\n"
    "Original question: {question}\n\n"
    "Alternative queries:"
)


def multi_query_search(
    question: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    llm_provider: LLMProvider,
    k: int = 5,
    fetch_k: int = 10,
) -> list[SearchResult]:
    """Generate query variations with the LLM, run each, merge via RRF."""
    resp = llm_provider.generate(
        _MULTI_QUERY_PROMPT.format(question=question),
        temperature=0.7, max_tokens=256,
    )
    alt_queries = [q.strip().lstrip("0123456789.-) ") for q in resp.text.strip().split("\n") if q.strip()]
    queries = [question] + alt_queries[:3]

    ranked_lists: list[list[tuple[int, float]]] = []
    for q in queries:
        emb = embedding_provider.embed_query(q)
        results = vector_store.search(emb, k=fetch_k)
        ranked_lists.append([(r.index, r.score) for r in results])

    fused = _rrf_merge(*ranked_lists)
    all_results = vector_store.search(embedding_provider.embed_query(question), k=fetch_k)
    res_map = {r.index: r for r in all_results}
    texts, _ = vector_store.get_all_embeddings()

    out: list[SearchResult] = []
    for idx, rrf_score in fused[:k]:
        if idx in res_map:
            r = res_map[idx]
            out.append(SearchResult(text=r.text, score=rrf_score, index=idx, metadata=r.metadata))
        elif 0 <= idx < len(texts):
            out.append(SearchResult(text=texts[idx], score=rrf_score, index=idx))
    return out


# ─── HyDE — Hypothetical Document Embeddings ─────────────────────────────

_HYDE_PROMPT = (
    "Write a short paragraph that would be the ideal passage in a document "
    "to answer the following question. Do NOT say 'I don't know'. "
    "Just write the passage as if it exists.\n\n"
    "Question: {question}\n\n"
    "Ideal passage:"
)


def hyde_search(
    question: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    llm_provider: LLMProvider,
    k: int = 5,
) -> list[SearchResult]:
    """Generate a hypothetical answer, embed it, search with that embedding."""
    resp = llm_provider.generate(
        _HYDE_PROMPT.format(question=question),
        temperature=0.7, max_tokens=512,
    )
    hypothetical_doc = resp.text.strip()
    hyde_embedding = embedding_provider.embed_query(hypothetical_doc)
    return vector_store.search(hyde_embedding, k=k)
