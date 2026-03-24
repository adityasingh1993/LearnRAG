"""
Enterprise-grade conversation context management for RAG pipelines.

Strategies:
  - none             : Stateless — each query is independent
  - full_history     : Pass the entire conversation to every query
  - sliding_window   : Keep only the last N turns
  - summary_buffer   : Summarise older turns via LLM, keep recent ones verbatim
  - token_budget     : Keep as many recent turns as fit within a token limit
  - relevant_history : Embed past turns and retrieve only the most relevant ones
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_providers import LLMProvider
    from core.embeddings import EmbeddingProvider


@dataclass
class Turn:
    role: str        # "user" or "assistant"
    content: str
    turn_number: int = 0


CONTEXT_STRATEGIES = {
    "none":             "Stateless — every query is independent",
    "full_history":     "Pass the entire conversation history each time",
    "sliding_window":   "Keep the last N conversation turns",
    "summary_buffer":   "Summarise older turns (LLM), keep recent turns verbatim",
    "token_budget":     "Keep as many recent turns as fit in a token budget",
    "relevant_history": "Embed past turns, retrieve only the relevant ones",
}


class ContextManager(ABC):
    """Base class for all context strategies."""

    @abstractmethod
    def add_turn(self, role: str, content: str) -> None: ...

    @abstractmethod
    def get_context_string(self) -> str:
        """Return formatted conversation history to inject into the prompt."""
        ...

    def get_history(self) -> list[Turn]:
        """Return raw turn list."""
        return []

    def clear(self) -> None: ...

    @property
    def strategy_name(self) -> str:
        return self.__class__.__name__

    @property
    def turn_count(self) -> int:
        return len(self.get_history())


# ─── Stateless ────────────────────────────────────────────────────────────

class NoContext(ContextManager):
    """No history — every query is independent."""

    def add_turn(self, role: str, content: str) -> None:
        pass

    def get_context_string(self) -> str:
        return ""

    def clear(self) -> None:
        pass


# ─── Full history ─────────────────────────────────────────────────────────

class FullHistoryContext(ContextManager):
    """Keep and return every turn of the conversation."""

    def __init__(self):
        self._history: list[Turn] = []
        self._counter = 0

    def add_turn(self, role: str, content: str) -> None:
        self._counter += 1
        self._history.append(Turn(role=role, content=content, turn_number=self._counter))

    def get_context_string(self) -> str:
        if not self._history:
            return ""
        lines = []
        for t in self._history:
            tag = "User" if t.role == "user" else "Assistant"
            lines.append(f"[{tag}]: {t.content}")
        return "\n".join(lines)

    def get_history(self) -> list[Turn]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self._counter = 0


# ─── Sliding window ──────────────────────────────────────────────────────

class SlidingWindowContext(ContextManager):
    """Keep the last N turns (a 'turn' is one user + one assistant message)."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._history: list[Turn] = []
        self._counter = 0

    def add_turn(self, role: str, content: str) -> None:
        self._counter += 1
        self._history.append(Turn(role=role, content=content, turn_number=self._counter))

    def _windowed(self) -> list[Turn]:
        max_msgs = self.window_size * 2  # user+assistant per turn
        return self._history[-max_msgs:]

    def get_context_string(self) -> str:
        window = self._windowed()
        if not window:
            return ""
        lines = []
        for t in window:
            tag = "User" if t.role == "user" else "Assistant"
            lines.append(f"[{tag}]: {t.content}")
        return "\n".join(lines)

    def get_history(self) -> list[Turn]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self._counter = 0


# ─── Summary buffer ──────────────────────────────────────────────────────

_SUMMARISE_PROMPT = (
    "You are a conversation summariser. Condense the following conversation history "
    "into a brief summary (2-4 sentences) that preserves all key facts, questions asked, "
    "and answers given. Do NOT lose any factual details.\n\n"
    "Conversation:\n{conversation}\n\n"
    "Summary:"
)


class SummaryBufferContext(ContextManager):
    """
    Keep the last `buffer_size` turns verbatim.
    Older turns get summarised by the LLM into a rolling summary.
    """

    def __init__(self, llm_provider: "LLMProvider | None" = None, buffer_size: int = 3):
        self.llm_provider = llm_provider
        self.buffer_size = buffer_size
        self._history: list[Turn] = []
        self._summary: str = ""
        self._counter = 0

    def add_turn(self, role: str, content: str) -> None:
        self._counter += 1
        self._history.append(Turn(role=role, content=content, turn_number=self._counter))
        self._maybe_summarise()

    def _maybe_summarise(self) -> None:
        max_buffer = self.buffer_size * 2
        if len(self._history) <= max_buffer:
            return
        if self.llm_provider is None:
            self._history = self._history[-max_buffer:]
            return

        overflow = self._history[:-max_buffer]
        to_summarise = self._format_turns(overflow)
        if self._summary:
            to_summarise = f"Previous summary: {self._summary}\n\n{to_summarise}"

        resp = self.llm_provider.generate(
            _SUMMARISE_PROMPT.format(conversation=to_summarise),
            temperature=0.0, max_tokens=300,
        )
        self._summary = resp.text.strip()
        self._history = self._history[-max_buffer:]

    @staticmethod
    def _format_turns(turns: list[Turn]) -> str:
        return "\n".join(
            f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}" for t in turns
        )

    def get_context_string(self) -> str:
        parts = []
        if self._summary:
            parts.append(f"[Conversation summary]: {self._summary}")
        for t in self._history:
            tag = "User" if t.role == "user" else "Assistant"
            parts.append(f"[{tag}]: {t.content}")
        return "\n".join(parts) if parts else ""

    def get_history(self) -> list[Turn]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self._summary = ""
        self._counter = 0


# ─── Token budget ─────────────────────────────────────────────────────────

class TokenBudgetContext(ContextManager):
    """
    Keep as many recent turns as fit within a token budget.
    Uses tiktoken if available, otherwise estimates ~4 chars per token.
    """

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self._history: list[Turn] = []
        self._counter = 0
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass

    def _count_tokens(self, text: str) -> int:
        if self._encoder:
            return len(self._encoder.encode(text))
        return max(1, len(text) // 4)

    def add_turn(self, role: str, content: str) -> None:
        self._counter += 1
        self._history.append(Turn(role=role, content=content, turn_number=self._counter))

    def _budgeted(self) -> list[Turn]:
        budget = self.max_tokens
        result: list[Turn] = []
        for t in reversed(self._history):
            cost = self._count_tokens(t.content) + 10  # overhead for role tags
            if budget - cost < 0 and result:
                break
            budget -= cost
            result.append(t)
        result.reverse()
        return result

    def get_context_string(self) -> str:
        window = self._budgeted()
        if not window:
            return ""
        lines = []
        for t in window:
            tag = "User" if t.role == "user" else "Assistant"
            lines.append(f"[{tag}]: {t.content}")
        return "\n".join(lines)

    def get_history(self) -> list[Turn]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self._counter = 0


# ─── Relevant history (embedding-based) ──────────────────────────────────

class RelevantHistoryContext(ContextManager):
    """
    Embed all past turns. At query time, retrieve the top-K most relevant
    past exchanges based on cosine similarity to the new question.
    """

    def __init__(self, embedding_provider: "EmbeddingProvider | None" = None, top_k: int = 3):
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self._history: list[Turn] = []
        self._counter = 0

    def add_turn(self, role: str, content: str) -> None:
        self._counter += 1
        self._history.append(Turn(role=role, content=content, turn_number=self._counter))

    def get_context_string(self, query: str = "") -> str:
        if not self._history or not query:
            return self._fallback()
        if self.embedding_provider is None:
            return self._fallback()

        import numpy as np

        turn_texts = [t.content for t in self._history]
        try:
            turn_embeds = self.embedding_provider.embed(turn_texts)
            query_embed = self.embedding_provider.embed_query(query)

            sims = np.dot(turn_embeds, query_embed)
            norms = np.linalg.norm(turn_embeds, axis=1) * np.linalg.norm(query_embed)
            norms[norms == 0] = 1e-10
            scores = sims / norms

            top_indices = np.argsort(scores)[::-1][:self.top_k * 2]
            top_indices = sorted(top_indices)
        except Exception:
            return self._fallback()

        lines = []
        for idx in top_indices:
            t = self._history[idx]
            tag = "User" if t.role == "user" else "Assistant"
            lines.append(f"[{tag} (turn {t.turn_number})]: {t.content}")
        return "\n".join(lines) if lines else ""

    def _fallback(self) -> str:
        recent = self._history[-6:]
        if not recent:
            return ""
        return "\n".join(
            f"[{'User' if t.role == 'user' else 'Assistant'}]: {t.content}" for t in recent
        )

    def get_history(self) -> list[Turn]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self._counter = 0


# ═════════════════════════════════════════════════════════════════════════
#  Factory
# ═════════════════════════════════════════════════════════════════════════

def create_context_manager(
    strategy: str = "none",
    llm_provider: "LLMProvider | None" = None,
    embedding_provider: "EmbeddingProvider | None" = None,
    **kwargs,
) -> ContextManager:
    """Create a context manager by strategy name."""
    if strategy == "none":
        return NoContext()
    elif strategy == "full_history":
        return FullHistoryContext()
    elif strategy == "sliding_window":
        return SlidingWindowContext(window_size=kwargs.get("window_size", 5))
    elif strategy == "summary_buffer":
        return SummaryBufferContext(
            llm_provider=llm_provider,
            buffer_size=kwargs.get("buffer_size", 3),
        )
    elif strategy == "token_budget":
        return TokenBudgetContext(max_tokens=kwargs.get("max_tokens", 2000))
    elif strategy == "relevant_history":
        return RelevantHistoryContext(
            embedding_provider=embedding_provider,
            top_k=kwargs.get("top_k", 3),
        )
    else:
        raise ValueError(f"Unknown context strategy: {strategy}. Choose from {list(CONTEXT_STRATEGIES.keys())}")
