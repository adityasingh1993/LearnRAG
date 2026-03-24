"""
Enterprise-grade guardrails for RAG pipelines.
All guardrails are optional and can be mixed-and-matched.

Input guardrails  — run BEFORE retrieval/generation.
Output guardrails — run AFTER generation.

Each guardrail returns a GuardrailResult with pass/fail and a reason.
The pipeline can be configured to block or warn on failures.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_providers import LLMProvider


@dataclass
class GuardrailResult:
    passed: bool
    name: str
    reason: str = ""
    details: dict = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════
#  Input guardrails
# ═════════════════════════════════════════════════════════════════════════

class InputGuardrail(ABC):
    @abstractmethod
    def check(self, query: str, **kwargs) -> GuardrailResult:
        ...


class InputLengthGuardrail(InputGuardrail):
    """Reject queries that are too short or too long."""

    def __init__(self, min_chars: int = 3, max_chars: int = 5000):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def check(self, query: str, **kwargs) -> GuardrailResult:
        n = len(query.strip())
        if n < self.min_chars:
            return GuardrailResult(False, "Input Length", f"Query too short ({n} chars, min {self.min_chars})")
        if n > self.max_chars:
            return GuardrailResult(False, "Input Length", f"Query too long ({n} chars, max {self.max_chars})")
        return GuardrailResult(True, "Input Length", f"{n} chars — OK")


class PIIDetectionGuardrail(InputGuardrail):
    """Detect common PII patterns (email, phone, SSN, credit card) in the query."""

    PATTERNS = {
        "email": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:\d[ -]*?){13,19}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }

    def check(self, query: str, **kwargs) -> GuardrailResult:
        found = {}
        for name, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, query)
            if matches:
                found[name] = len(matches)
        if found:
            return GuardrailResult(False, "PII Detection",
                                   f"PII detected: {found}", details={"pii_types": found})
        return GuardrailResult(True, "PII Detection", "No PII found")


class PromptInjectionGuardrail(InputGuardrail):
    """
    Heuristic detection of prompt-injection attempts.
    Catches common jailbreak patterns, system-prompt overrides, and role switches.
    """

    SUSPECT_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions',
        r'ignore\s+(all\s+)?above',
        r'disregard\s+(all\s+)?(previous|prior|above)',
        r'you\s+are\s+now\s+',
        r'pretend\s+(you\s+are|to\s+be)',
        r'act\s+as\s+(if|a|an)',
        r'new\s+instructions?\s*:',
        r'system\s*:\s*',
        r'<\|?(system|im_start|endoftext)',
        r'forget\s+(everything|all|your)',
        r'do\s+not\s+follow',
        r'override\s+(the\s+)?(system|instructions)',
        r'jailbreak',
        r'DAN\s+mode',
    ]

    def check(self, query: str, **kwargs) -> GuardrailResult:
        lower = query.lower()
        for pattern in self.SUSPECT_PATTERNS:
            if re.search(pattern, lower):
                return GuardrailResult(False, "Prompt Injection",
                                       f"Possible injection detected: matched '{pattern}'",
                                       details={"pattern": pattern})
        return GuardrailResult(True, "Prompt Injection", "No injection patterns found")


class TopicGuardrail(InputGuardrail):
    """
    LLM-based topic filter. Checks whether the query is on-topic for the
    configured allowed topics. Requires an LLM provider.
    """

    def __init__(self, allowed_topics: str = "any topic related to the provided documents",
                 llm_provider: "LLMProvider | None" = None):
        self.allowed_topics = allowed_topics
        self.llm_provider = llm_provider

    def check(self, query: str, **kwargs) -> GuardrailResult:
        llm = kwargs.get("llm_provider") or self.llm_provider
        if llm is None:
            return GuardrailResult(True, "Topic Filter", "Skipped — no LLM provider")

        prompt = (
            f"Determine if the following user query is on-topic.\n"
            f"Allowed topics: {self.allowed_topics}\n\n"
            f"Query: {query}\n\n"
            f"Respond with ONLY 'on-topic' or 'off-topic' followed by a brief reason."
        )
        resp = llm.generate(prompt, temperature=0.0, max_tokens=100)
        text = resp.text.strip().lower()
        passed = text.startswith("on-topic") or "on-topic" in text[:30]
        return GuardrailResult(passed, "Topic Filter", resp.text.strip())


class ToxicityInputGuardrail(InputGuardrail):
    """
    Keyword + LLM-based toxicity check on the user query.
    The keyword list catches obvious cases; the LLM handles nuance.
    """

    TOXIC_KEYWORDS = [
        "kill", "murder", "bomb", "attack", "suicide", "terrorist",
        "hack into", "steal", "exploit vulnerability",
    ]

    def __init__(self, llm_provider: "LLMProvider | None" = None, use_llm: bool = True):
        self.llm_provider = llm_provider
        self.use_llm = use_llm

    def check(self, query: str, **kwargs) -> GuardrailResult:
        lower = query.lower()
        for kw in self.TOXIC_KEYWORDS:
            if kw in lower:
                return GuardrailResult(False, "Toxicity (Input)",
                                       f"Blocked — contains '{kw}'",
                                       details={"keyword": kw})

        llm = kwargs.get("llm_provider") or self.llm_provider
        if self.use_llm and llm is not None:
            prompt = (
                "Is the following user query harmful, toxic, or requesting dangerous/illegal information? "
                "Respond ONLY with 'safe' or 'unsafe' followed by a brief reason.\n\n"
                f"Query: {query}"
            )
            resp = llm.generate(prompt, temperature=0.0, max_tokens=80)
            text = resp.text.strip().lower()
            if text.startswith("unsafe") or "unsafe" in text[:20]:
                return GuardrailResult(False, "Toxicity (Input)", resp.text.strip())

        return GuardrailResult(True, "Toxicity (Input)", "Safe")


# ═════════════════════════════════════════════════════════════════════════
#  Output guardrails
# ═════════════════════════════════════════════════════════════════════════

class OutputGuardrail(ABC):
    @abstractmethod
    def check(self, query: str, context: str, answer: str, **kwargs) -> GuardrailResult:
        ...


class HallucinationGuardrail(OutputGuardrail):
    """
    LLM-based faithfulness check.
    Asks the LLM whether every claim in the answer is supported by the context.
    """

    def __init__(self, llm_provider: "LLMProvider | None" = None):
        self.llm_provider = llm_provider

    def check(self, query: str, context: str, answer: str, **kwargs) -> GuardrailResult:
        llm = kwargs.get("llm_provider") or self.llm_provider
        if llm is None:
            return GuardrailResult(True, "Hallucination Check", "Skipped — no LLM")

        prompt = (
            "You are a fact-checker. Given the CONTEXT and the ANSWER, determine if "
            "every claim in the answer is supported by the context.\n\n"
            f"CONTEXT:\n{context[:3000]}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Respond with a JSON object: {\"faithful\": true/false, \"unsupported_claims\": [\"...\"]}"
        )
        resp = llm.generate(prompt, temperature=0.0, max_tokens=512)
        text = resp.text.strip().lower()
        passed = '"faithful": true' in text or '"faithful":true' in text
        return GuardrailResult(passed, "Hallucination Check", resp.text.strip())


class RelevanceGuardrail(OutputGuardrail):
    """LLM-based relevance check — is the answer actually answering the question?"""

    def __init__(self, llm_provider: "LLMProvider | None" = None):
        self.llm_provider = llm_provider

    def check(self, query: str, context: str, answer: str, **kwargs) -> GuardrailResult:
        llm = kwargs.get("llm_provider") or self.llm_provider
        if llm is None:
            return GuardrailResult(True, "Relevance Check", "Skipped — no LLM")

        prompt = (
            "Does the following answer actually address the user's question? "
            "Respond with ONLY 'relevant' or 'irrelevant' followed by a brief reason.\n\n"
            f"Question: {query}\n\n"
            f"Answer: {answer[:2000]}"
        )
        resp = llm.generate(prompt, temperature=0.0, max_tokens=100)
        text = resp.text.strip().lower()
        passed = text.startswith("relevant") or "relevant" in text[:20]
        return GuardrailResult(passed, "Relevance Check", resp.text.strip())


class PIIOutputGuardrail(OutputGuardrail):
    """Scan the generated answer for PII and flag/redact if found."""

    PATTERNS = PIIDetectionGuardrail.PATTERNS

    def check(self, query: str, context: str, answer: str, **kwargs) -> GuardrailResult:
        found = {}
        for name, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, answer)
            if matches:
                found[name] = len(matches)
        if found:
            return GuardrailResult(False, "PII in Output",
                                   f"PII detected in answer: {found}",
                                   details={"pii_types": found})
        return GuardrailResult(True, "PII in Output", "No PII in answer")


class ToxicityOutputGuardrail(OutputGuardrail):
    """Check generated answer for toxic/harmful content."""

    def __init__(self, llm_provider: "LLMProvider | None" = None):
        self.llm_provider = llm_provider

    def check(self, query: str, context: str, answer: str, **kwargs) -> GuardrailResult:
        llm = kwargs.get("llm_provider") or self.llm_provider
        if llm is None:
            return GuardrailResult(True, "Toxicity (Output)", "Skipped — no LLM")

        prompt = (
            "Is the following AI-generated answer harmful, toxic, or inappropriate? "
            "Respond ONLY with 'safe' or 'unsafe' followed by a brief reason.\n\n"
            f"Answer: {answer[:2000]}"
        )
        resp = llm.generate(prompt, temperature=0.0, max_tokens=80)
        text = resp.text.strip().lower()
        if text.startswith("unsafe") or "unsafe" in text[:20]:
            return GuardrailResult(False, "Toxicity (Output)", resp.text.strip())
        return GuardrailResult(True, "Toxicity (Output)", "Safe")


# ═════════════════════════════════════════════════════════════════════════
#  Guardrail runner
# ═════════════════════════════════════════════════════════════════════════

AVAILABLE_INPUT_GUARDRAILS = {
    "input_length":      ("Input Length", "Reject too-short or too-long queries", False),
    "pii_detection":     ("PII Detection", "Block queries containing personal data", False),
    "prompt_injection":  ("Prompt Injection", "Detect jailbreak / override attempts", False),
    "topic_filter":      ("Topic Filter", "LLM checks if query is on-topic", True),
    "toxicity_input":    ("Toxicity (Input)", "Block harmful / toxic queries", True),
}

AVAILABLE_OUTPUT_GUARDRAILS = {
    "hallucination":     ("Hallucination Check", "Verify answer is grounded in context", True),
    "relevance":         ("Relevance Check", "Verify answer addresses the question", True),
    "pii_output":        ("PII in Output", "Flag PII leaking into the answer", False),
    "toxicity_output":   ("Toxicity (Output)", "Block harmful generated content", True),
}


def create_input_guardrails(
    enabled: list[str],
    llm_provider: "LLMProvider | None" = None,
    **kwargs,
) -> list[InputGuardrail]:
    """Factory: build a list of input guardrails from string names."""
    mapping: dict[str, type[InputGuardrail]] = {
        "input_length": InputLengthGuardrail,
        "pii_detection": PIIDetectionGuardrail,
        "prompt_injection": PromptInjectionGuardrail,
        "topic_filter": TopicGuardrail,
        "toxicity_input": ToxicityInputGuardrail,
    }
    result = []
    for name in enabled:
        cls = mapping.get(name)
        if cls is None:
            continue
        if name in ("topic_filter", "toxicity_input"):
            result.append(cls(llm_provider=llm_provider, **{k: v for k, v in kwargs.items() if k in cls.__init__.__code__.co_varnames}))
        else:
            result.append(cls())
    return result


def create_output_guardrails(
    enabled: list[str],
    llm_provider: "LLMProvider | None" = None,
) -> list[OutputGuardrail]:
    """Factory: build a list of output guardrails from string names."""
    mapping: dict[str, type[OutputGuardrail]] = {
        "hallucination": HallucinationGuardrail,
        "relevance": RelevanceGuardrail,
        "pii_output": PIIOutputGuardrail,
        "toxicity_output": ToxicityOutputGuardrail,
    }
    result = []
    for name in enabled:
        cls = mapping.get(name)
        if cls is None:
            continue
        if name in ("hallucination", "relevance", "toxicity_output"):
            result.append(cls(llm_provider=llm_provider))
        else:
            result.append(cls())
    return result


def run_input_guardrails(
    guardrails: list[InputGuardrail], query: str, **kwargs,
) -> list[GuardrailResult]:
    return [g.check(query, **kwargs) for g in guardrails]


def run_output_guardrails(
    guardrails: list[OutputGuardrail], query: str, context: str, answer: str, **kwargs,
) -> list[GuardrailResult]:
    return [g.check(query, context, answer, **kwargs) for g in guardrails]
