"""
Embedding providers for converting text to vectors.
Includes a local TF-IDF fallback that works without any API keys.
"""

from abc import ABC, abstractmethod
import numpy as np
import requests


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns (n_texts, dim) array of embeddings."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Returns (dim,) array for a single query."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._dim = 1536 if "ada" in model else 1536

    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        embeddings = [item.embedding for item in response.data]
        result = np.array(embeddings)
        self._dim = result.shape[1]
        return result

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class OpenRouterEmbeddings(EmbeddingProvider):
    """
    Embeddings via OpenRouter's /api/v1/embeddings endpoint.
    OpenAI-compatible API — supports models like openai/text-embedding-3-small,
    google/text-embedding-004, etc.
    """

    MODELS = [
        "openai/text-embedding-3-small",
        "openai/text-embedding-3-large",
        "openai/text-embedding-ada-002",
        "google/text-embedding-004",
    ]

    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self._dim = 1536

    def name(self) -> str:
        return f"OpenRouter ({self.model.split('/')[-1]})"

    def dimension(self) -> int:
        return self._dim

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "RAG Learning App",
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        result = np.array(embeddings)
        self._dim = result.shape[1]
        return result

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class OllamaEmbeddings(EmbeddingProvider):
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dim = 768

    def name(self) -> str:
        return f"Ollama ({self.model})"

    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        result = np.array(embeddings)
        self._dim = result.shape[1]
        return result

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class TFIDFEmbeddings(EmbeddingProvider):
    """
    Local TF-IDF based embeddings. No API key needed.
    Great for educational purposes - shows how text vectorization works.
    Uses SVD to reduce to a fixed dimensionality.
    """

    def __init__(self, dim: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._target_dim = dim
        self._dim = dim
        self._vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self._svd = TruncatedSVD(n_components=dim)
        self._is_fitted = False
        self._corpus: list[str] = []

    def name(self) -> str:
        return f"TF-IDF (local, dim={self._dim})"

    def dimension(self) -> int:
        return self._dim

    def _fit(self, texts: list[str]):
        self._corpus = list(texts)
        if len(self._corpus) < 2:
            self._corpus.append("placeholder document for fitting")
        tfidf_matrix = self._vectorizer.fit_transform(self._corpus)
        n_components = min(self._target_dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if n_components < 1:
            n_components = 1
        self._svd = __import__("sklearn.decomposition", fromlist=["TruncatedSVD"]).TruncatedSVD(
            n_components=n_components
        )
        self._svd.fit(tfidf_matrix)
        self._dim = n_components
        self._is_fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        all_texts = list(texts)
        if self._corpus:
            all_texts = self._corpus + [t for t in texts if t not in self._corpus]
        self._fit(all_texts)
        tfidf = self._vectorizer.transform(texts)
        result = self._svd.transform(tfidf)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return result / norms

    def embed_query(self, text: str) -> np.ndarray:
        if not self._is_fitted:
            self._fit([text, "default context document"])
        tfidf = self._vectorizer.transform([text])
        result = self._svd.transform(tfidf)
        norm = np.linalg.norm(result)
        return (result[0] / norm) if norm > 0 else result[0]


class SimpleHashEmbeddings(EmbeddingProvider):
    """
    Deterministic hash-based embeddings. Always works, no fitting needed.
    Useful as a quick demo - shows the concept of mapping text to vectors.
    Not semantically meaningful but demonstrates the interface.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    def name(self) -> str:
        return f"Hash Embeddings (demo, dim={self._dim})"

    def dimension(self) -> int:
        return self._dim

    def _hash_text(self, text: str) -> np.ndarray:
        import hashlib
        words = text.lower().split()
        vec = np.zeros(self._dim)
        for i, word in enumerate(words):
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            for d in range(self._dim):
                bit = (h >> d) & 1
                vec[d] += (1 if bit else -1) * (1.0 / (i + 1))
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._hash_text(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._hash_text(text)


def create_embeddings(provider: str, **kwargs) -> EmbeddingProvider:
    """Factory function to create an embedding provider."""
    providers = {
        "openai": OpenAIEmbeddings,
        "openrouter": OpenRouterEmbeddings,
        "ollama": OllamaEmbeddings,
        "tfidf": TFIDFEmbeddings,
        "hash": SimpleHashEmbeddings,
    }
    if provider not in providers:
        raise ValueError(f"Unknown embedding provider: {provider}. Choose from {list(providers.keys())}")
    return providers[provider](**kwargs)
