"""
Token counting and session-level usage tracking.
Helps users understand and optimise the cost of their RAG pipelines.

Uses tiktoken (cl100k_base) when available, falls back to a ~4 chars/token estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _encoder = False  # sentinel: not available
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


@dataclass
class StepTokenUsage:
    """Token usage for a single pipeline step."""
    step_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens + self.embedding_tokens


@dataclass
class TurnTokenUsage:
    """Aggregated token usage for a single query turn."""
    steps: list[StepTokenUsage] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def embedding_tokens(self) -> int:
        return sum(s.embedding_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    def add(self, step_name: str, prompt: int = 0, completion: int = 0,
            embedding: int = 0, total: int = 0) -> None:
        self.steps.append(StepTokenUsage(
            step_name=step_name,
            prompt_tokens=prompt,
            completion_tokens=completion,
            embedding_tokens=embedding,
            total_tokens=total or (prompt + completion + embedding),
        ))


@dataclass
class SessionTokenTracker:
    """Tracks cumulative token usage across an entire chat session."""
    turns: list[TurnTokenUsage] = field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def total_completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.turns)

    @property
    def total_embedding_tokens(self) -> int:
        return sum(t.embedding_tokens for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens for t in self.turns)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def add_turn(self, turn: TurnTokenUsage) -> None:
        self.turns.append(turn)

    def estimate_cost(
        self,
        prompt_price_per_1k: float = 0.00015,
        completion_price_per_1k: float = 0.0006,
        embedding_price_per_1k: float = 0.00002,
    ) -> dict:
        """
        Estimate cost in USD. Default prices are for GPT-4o-mini.
        Returns per-category and total cost.
        """
        prompt_cost = (self.total_prompt_tokens / 1000) * prompt_price_per_1k
        completion_cost = (self.total_completion_tokens / 1000) * completion_price_per_1k
        embedding_cost = (self.total_embedding_tokens / 1000) * embedding_price_per_1k
        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "embedding_cost": embedding_cost,
            "total_cost": prompt_cost + completion_cost + embedding_cost,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "embedding_tokens": self.total_embedding_tokens,
            "total_tokens": self.total_tokens,
        }


# Approximate pricing per 1K tokens for common models
MODEL_PRICING = {
    "gpt-4o-mini":        {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o":             {"prompt": 0.0025,  "completion": 0.01},
    "gpt-4":              {"prompt": 0.03,    "completion": 0.06},
    "gpt-3.5-turbo":      {"prompt": 0.0005,  "completion": 0.0015},
    "claude-3.5-sonnet":  {"prompt": 0.003,   "completion": 0.015},
    "llama-3.1-8b":       {"prompt": 0.0,     "completion": 0.0},
    "gemma-2-9b":         {"prompt": 0.0,     "completion": 0.0},
    "mistral-7b":         {"prompt": 0.0,     "completion": 0.0},
    "text-embedding-3-small": {"embedding": 0.00002},
    "text-embedding-3-large": {"embedding": 0.00013},
    "text-embedding-ada-002": {"embedding": 0.0001},
}


def get_model_pricing(model_name: str) -> dict:
    """Return pricing for a model name (fuzzy match)."""
    lower = model_name.lower()
    for key, prices in MODEL_PRICING.items():
        if key in lower:
            return prices
    return {"prompt": 0.00015, "completion": 0.0006}
