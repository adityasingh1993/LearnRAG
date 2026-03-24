"""
PromptCraft — Scoring Utilities
Composite scoring, color mapping, and analysis helpers.
"""

from prompting_core import config


def get_score_color(score: float) -> str:
    """Get a color for a score value (0-100)."""
    if score >= 85:
        return config.SCORE_COLORS["excellent"]
    elif score >= 70:
        return config.SCORE_COLORS["good"]
    elif score >= 50:
        return config.SCORE_COLORS["moderate"]
    elif score >= 30:
        return config.SCORE_COLORS["poor"]
    else:
        return config.SCORE_COLORS["bad"]


def get_score_label(score: float) -> str:
    """Get a label for a score value (0-100)."""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Moderate"
    elif score >= 30:
        return "Poor"
    else:
        return "Bad"


def get_verdict_emoji(verdict: str) -> str:
    """Get an emoji for a claim verdict."""
    mapping = {
        "supported": "✅",
        "uncertain": "⚠️",
        "unsupported": "❌",
    }
    return mapping.get(verdict.lower(), "❓")


def composite_score(quality_scores: dict) -> float:
    """Calculate a weighted composite score from quality dimensions."""
    weights = {
        "accuracy": 0.30,
        "completeness": 0.20,
        "clarity": 0.15,
        "reasoning": 0.25,
        "conciseness": 0.10,
    }
    total = 0
    weight_sum = 0
    for key, weight in weights.items():
        if key in quality_scores:
            total += quality_scores[key] * weight
            weight_sum += weight

    return round(total / weight_sum, 1) if weight_sum > 0 else 0


def token_efficiency(tokens: int, quality_score: float) -> float:
    """Score token efficiency (quality per token). Higher is better."""
    if tokens == 0:
        return 0
    return round(quality_score / (tokens / 100), 1)


def format_latency(ms: int) -> str:
    """Format latency in a human-readable way."""
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


def improvement_percentage(baseline_score: float, improved_score: float) -> float:
    """Calculate improvement percentage."""
    if baseline_score == 0:
        return 100.0 if improved_score > 0 else 0.0
    return round(((improved_score - baseline_score) / baseline_score) * 100, 1)
