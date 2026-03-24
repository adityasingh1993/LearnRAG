"""
Vector store implementations for storing and retrieving embeddings.
Includes a simple NumPy-based store for education and ChromaDB for production use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SearchResult:
    text: str
    score: float
    index: int
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def add(self, texts: list[str], embeddings: np.ndarray, metadatas: list[dict] | None = None):
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Returns (texts, embeddings_array) for visualization."""
        ...


class NumpyVectorStore(VectorStore):
    """
    Educational in-memory vector store using NumPy.
    Demonstrates how similarity search works under the hood.
    """

    def __init__(self):
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._metadatas: list[dict] = []

    def add(self, texts: list[str], embeddings: np.ndarray, metadatas: list[dict] | None = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        if self._embeddings is None or len(self._texts) == 0:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = doc_norms @ query_norm

        k = min(k, len(self._texts))
        top_indices = np.argsort(similarities)[::-1][:k]

        return [
            SearchResult(
                text=self._texts[i],
                score=float(similarities[i]),
                index=i,
                metadata=self._metadatas[i],
            )
            for i in top_indices
        ]

    def search_mmr(
        self, query_embedding: np.ndarray, k: int = 5, lambda_mult: float = 0.5, fetch_k: int = 20,
    ) -> list[SearchResult]:
        """
        Maximal Marginal Relevance search.
        Balances relevance to query with diversity among results.
        """
        if self._embeddings is None or len(self._texts) == 0:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = doc_norms @ query_norm

        fetch_k = min(fetch_k, len(self._texts))
        k = min(k, fetch_k)
        candidate_indices = np.argsort(similarities)[::-1][:fetch_k].tolist()

        selected = []
        while len(selected) < k and candidate_indices:
            best_score = -float("inf")
            best_idx = -1
            for idx in candidate_indices:
                relevance = float(similarities[idx])
                if selected:
                    selected_embs = doc_norms[selected]
                    max_sim_to_selected = float(np.max(selected_embs @ doc_norms[idx]))
                else:
                    max_sim_to_selected = 0.0
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected.append(best_idx)
            candidate_indices.remove(best_idx)

        return [
            SearchResult(
                text=self._texts[i],
                score=float(similarities[i]),
                index=i,
                metadata=self._metadatas[i],
            )
            for i in selected
        ]

    def count(self) -> int:
        return len(self._texts)

    def clear(self):
        self._texts = []
        self._embeddings = None
        self._metadatas = []

    def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        if self._embeddings is None:
            return [], np.array([])
        return self._texts, self._embeddings


class ChromaVectorStore(VectorStore):
    """Production-ready vector store using ChromaDB."""

    def __init__(self, collection_name: str = "rag_collection", persist_dir: str | None = None):
        import chromadb
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts: list[str], embeddings: np.ndarray, metadatas: list[dict] | None = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        clean_metadatas = []
        for m in metadatas:
            clean = {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
            clean_metadatas.append(clean if clean else {"source": "uploaded"})
        start_id = self._collection.count()
        ids = [f"doc_{start_id + i}" for i in range(len(texts))]
        self._collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=clean_metadatas,
            ids=ids,
        )

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        k = min(k, max(self._collection.count(), 1))
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
        )
        search_results = []
        for i in range(len(results["documents"][0])):
            score = 1 - results["distances"][0][i] if results["distances"] else 0.0
            search_results.append(SearchResult(
                text=results["documents"][0][i],
                score=score,
                index=i,
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            ))
        return search_results

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        if self._collection.count() == 0:
            return [], np.array([])
        data = self._collection.get(include=["documents", "embeddings"])
        return data["documents"], np.array(data["embeddings"])


def create_vector_store(store_type: str = "numpy", **kwargs) -> VectorStore:
    """Factory function for vector stores."""
    stores = {
        "numpy": NumpyVectorStore,
        "chroma": ChromaVectorStore,
    }
    if store_type not in stores:
        raise ValueError(f"Unknown store type: {store_type}. Choose from {list(stores.keys())}")
    return stores[store_type](**kwargs)
