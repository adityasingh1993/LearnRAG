"""
PromptCraft — Evaluator
LLM-as-Judge evaluation for hallucination detection, faithfulness,
and response quality scoring.
"""

import json
import re
from prompting_core import llm_client
from prompting_core import config


CLAIM_EXTRACTION_PROMPT = """You are a precise claim extractor. Given a text response, extract every distinct factual claim made.

Response to analyze:
\"\"\"
{response}
\"\"\"

Extract each factual claim as a separate item. Return ONLY a valid JSON array of strings.
Example: ["The Earth orbits the Sun", "Water boils at 100°C at sea level"]

Claims:"""


HALLUCINATION_JUDGE_PROMPT = """You are an expert fact-checker and hallucination detector. Your job is to evaluate each claim for factual accuracy.

{context_section}

Claims to evaluate:
{claims_json}

For EACH claim, evaluate:
1. Is this claim factually accurate based on widely-known facts{context_note}?
2. Rate your confidence.

Return ONLY a valid JSON array with objects for each claim:
[
    {{
        "claim": "the claim text",
        "verdict": "supported" | "uncertain" | "unsupported",
        "reason": "brief explanation",
        "confidence": 0.0 to 1.0
    }}
]

Evaluation:"""


QUALITY_JUDGE_PROMPT = """You are an expert response quality evaluator. Rate the following LLM response on multiple dimensions.

Question asked: "{question}"
Response: 
\"\"\"
{response}
\"\"\"

Rate each dimension from 0-100:
1. **Accuracy**: Are the facts correct?
2. **Completeness**: Does it fully answer the question?
3. **Clarity**: Is it well-structured and easy to understand?
4. **Reasoning**: Does it show sound logical reasoning?
5. **Conciseness**: Is it appropriately concise without unnecessary padding?

Return ONLY a valid JSON object:
{{
    "accuracy": 0-100,
    "completeness": 0-100,
    "clarity": 0-100,
    "reasoning": 0-100,
    "conciseness": 0-100,
    "overall_score": 0-100,
    "summary": "One-line summary of the quality"
}}

Evaluation:"""


def _parse_json(text: str):
    """Try to extract JSON from LLM response text."""
    # Try to find JSON in the text
    # First try direct parse
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array or object in text
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    return None


def extract_claims(response_text: str) -> list[str]:
    """Extract factual claims from a response."""
    prompt = CLAIM_EXTRACTION_PROMPT.format(response=response_text)
    result = llm_client.generate(
        prompt=prompt,
        temperature=config.JUDGE_TEMPERATURE,
        model=config.JUDGE_MODEL,
    )

    claims = _parse_json(result["response"])
    if isinstance(claims, list):
        return [str(c) for c in claims]

    # Fallback: split by sentences
    sentences = [s.strip() for s in response_text.split('.') if len(s.strip()) > 15]
    return sentences[:10]


def evaluate_hallucination(
    response_text: str,
    context: str = "",
) -> dict:
    """
    Evaluate a response for hallucinations.

    Returns:
        dict with: claims, evaluations, hallucination_score (0-100, lower=better),
                   faithfulness_score (0-100, higher=better)
    """
    claims = extract_claims(response_text)

    if not claims:
        return {
            "claims": [],
            "evaluations": [],
            "hallucination_score": 0,
            "faithfulness_score": 100,
        }

    # Build context section
    if context:
        context_section = f'Reference context (ground truth):\n"""\n{context}\n"""'
        context_note = " and the provided context"
    else:
        context_section = "No reference context provided. Evaluate based on widely-known factual knowledge."
        context_note = ""

    prompt = HALLUCINATION_JUDGE_PROMPT.format(
        context_section=context_section,
        context_note=context_note,
        claims_json=json.dumps(claims, indent=2),
    )

    result = llm_client.generate(
        prompt=prompt,
        temperature=config.JUDGE_TEMPERATURE,
        model=config.JUDGE_MODEL,
    )

    evaluations = _parse_json(result["response"])
    if not isinstance(evaluations, list):
        # Fallback
        evaluations = [
            {"claim": c, "verdict": "uncertain", "reason": "Could not evaluate", "confidence": 0.5}
            for c in claims
        ]

    # Calculate scores
    supported = sum(1 for e in evaluations if e.get("verdict") == "supported")
    unsupported = sum(1 for e in evaluations if e.get("verdict") == "unsupported")
    total = len(evaluations)

    faithfulness_score = round((supported / total) * 100) if total > 0 else 0
    hallucination_score = round((unsupported / total) * 100) if total > 0 else 0

    return {
        "claims": claims,
        "evaluations": evaluations,
        "hallucination_score": hallucination_score,
        "faithfulness_score": faithfulness_score,
    }


def evaluate_quality(question: str, response_text: str) -> dict:
    """
    Evaluate overall response quality.

    Returns:
        dict with: accuracy, completeness, clarity, reasoning, conciseness,
                   overall_score, summary
    """
    prompt = QUALITY_JUDGE_PROMPT.format(
        question=question,
        response=response_text,
    )

    result = llm_client.generate(
        prompt=prompt,
        temperature=config.JUDGE_TEMPERATURE,
        model=config.JUDGE_MODEL,
    )

    scores = _parse_json(result["response"])
    if not isinstance(scores, dict):
        scores = {
            "accuracy": 50,
            "completeness": 50,
            "clarity": 50,
            "reasoning": 50,
            "conciseness": 50,
            "overall_score": 50,
            "summary": "Could not evaluate quality",
        }

    return scores


def compare_responses(question: str, response_a: str, response_b: str) -> dict:
    """
    Compare two responses side-by-side.

    Returns:
        dict with quality scores for both and a winner determination.
    """
    quality_a = evaluate_quality(question, response_a)
    quality_b = evaluate_quality(question, response_b)

    winner = "A" if quality_a.get("overall_score", 0) > quality_b.get("overall_score", 0) else "B"
    if quality_a.get("overall_score", 0) == quality_b.get("overall_score", 0):
        winner = "Tie"

    return {
        "quality_a": quality_a,
        "quality_b": quality_b,
        "winner": winner,
    }
