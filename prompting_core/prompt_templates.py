"""
PromptCraft — Prompt Templates
Templates for 8 prompting techniques, each parameterized for any question.
"""

from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    """A prompting technique template."""
    key: str
    name: str
    icon: str
    description: str
    how_it_works: str
    why_it_helps: str
    system_prompt: str
    user_prompt_template: str
    example_boost: str  # Describes what improvement to expect
    recommended_temperature: float = 0.7  # Per-technique temperature


# ── Templates ────────────────────────────────────────────────────────────

ZERO_SHOT = PromptTemplate(
    key="zero_shot",
    name="Zero-Shot",
    icon="🎯",
    description="Ask the question directly, with no examples or special instructions.",
    how_it_works="Reset if any previous approach is set. The question is sent to the LLM exactly as-is, relying entirely on the model's pre-trained knowledge.",
    why_it_helps="This is the **baseline**. It shows what the LLM does with no guidance — useful for simple questions but often produces vague or hallucinated answers for complex ones.",
    example_boost="Baseline — no boost applied",
    system_prompt="",
    user_prompt_template="{question}",
    recommended_temperature=0.7,
)

ROLE_PROMPTING = PromptTemplate(
    key="role_prompting",
    name="Role Prompting",
    icon="🎭",
    description="Assign a specific expert persona to the LLM before asking the question.",
    how_it_works="Reset if any previous approach is set. A system prompt assigns a role. This primes the model to respond with domain-specific vocabulary, depth, and accuracy. If it does not belong to domain, model should not give any vague response",
    why_it_helps="Role assignment activates domain-relevant knowledge patterns in the model. It reduces generic answers and increases the use of **accurate, specialized terminology**. ",
    example_boost="Expect highly technical, domain-specific, authoritative answers",
    system_prompt="You are an expert medical science, medicines. You have to answer the question based on the context provided. First classify if the question belongs to domain of medical science, medicines or not. If not, then respond that this question does not belong to the domain of medical science, medicines.",
    user_prompt_template="{question}",
    recommended_temperature=0.5,
)

FEW_SHOT = PromptTemplate(
    key="few_shot",
    name="Few-Shot",
    icon="📝",
    description="Provide 2-3 example Q&A pairs before the actual question.",
    how_it_works="By showing the model examples of desired input-output pairs, it learns the expected format, depth of detail, and reasoning pattern to follow.",
    why_it_helps="Examples act as **implicit instructions**. The model mimics the demonstrated format and quality, producing more consistent, well-structured responses with fewer hallucinations.",
    example_boost="Expect more consistent format and fewer off-topic responses",
    system_prompt="You are a helpful and accurate assistant. Follow the same format and level of detail as shown in the examples below.",
    user_prompt_template="""Here are some examples of high-quality answers:

Example 1:
Q: What is the capital of France?
A: The capital of France is **Paris**.

Example 2:
Q: What causes rain?
A: Rain is caused by the **water cycle**

Now answer this question with the same level of detail and accuracy:
Q: {question}
A:""",
    recommended_temperature=0.5,
)

CHAIN_OF_THOUGHT = PromptTemplate(
    key="chain_of_thought",
    name="Chain-of-Thought (CoT)",
    icon="🔗",
    description="Ask the LLM to think step-by-step before giving its final answer.",
    how_it_works="Reset if any previous approach is set. Adding 'Let's think step by step' or similar instructions forces the model to decompose the problem into logical steps, making each reasoning stage explicit and verifiable.",
    why_it_helps="Step-by-step reasoning **dramatically reduces errors** in math, logic, and multi-step problems. It makes the model's reasoning transparent so you can spot where mistakes occur.",
    example_boost="Expect significantly better accuracy on reasoning tasks",
    system_prompt="You are a careful analytical thinker. Always show your step-by-step reasoning before giving a final answer. Never skip steps.",
    user_prompt_template="""Question: {question}

Let's approach this step-by-step:
1. First, let me understand what's being asked.
2. Then, I'll work through the reasoning carefully.
3. Finally, I'll provide a clear answer.

Step-by-step reasoning:""",
    recommended_temperature=0.3,
)

STRUCTURED_OUTPUT = PromptTemplate(
    key="structured_output",
    name="Structured Output",
    icon="📐",
    description="Constrain the LLM to respond in a specific format (JSON, table, checklist).",
    how_it_works="Reset if any previous approach is set. By specifying the exact output structure, the model focuses on filling in accurate content rather than generating free-form text. This reduces rambling and speculative elaboration.",
    why_it_helps="Format constraints **reduce hallucination by limiting scope**. The model can't pad its response with fabricated details when it must fit a rigid structure. Each field demands a specific, verifiable piece of information.",
    example_boost="Expect more concise, verifiable, and consistent responses",
    system_prompt="You are a precise assistant that always responds in the exact format requested. Never deviate from the requested structure.",
    user_prompt_template="""Answer the following question in a structured JSON format.

Question: {question}

Respond ONLY with a valid JSON object containing these fields:
{{
    "answer": "Your direct answer here",
    "confidence": "high/medium/low",
    "key_facts": ["fact 1", "fact 2", "fact 3"],
    "sources_to_verify": ["where to verify this information"],
    "potential_caveats": ["any limitations or nuances"]
}}""",
    recommended_temperature=0.2,
)

SELF_CONSISTENCY = PromptTemplate(
    key="self_consistency",
    name="Self-Consistency",
    icon="🔄",
    description="Generate multiple reasoning paths and pick the most common answer.",
    how_it_works="Reset if any previous approach is set. The same question is sent to the LLM multiple times (with slight temperature variation). The answers are compared, and the most frequent conclusion is selected. This is like asking multiple experts and going with the majority opinion.",
    why_it_helps="Random hallucinations are unlikely to be consistent across runs. If 3 out of 3 runs agree, the answer is likely correct. **Disagreement signals uncertainty** and potential hallucination.",
    example_boost="Expect higher reliability through majority-vote consensus",
    system_prompt="You are a careful, analytical thinker. Reason through the problem step by step and provide a clear final answer.",
    user_prompt_template="""Question: {question}

Think through this carefully and provide your reasoning followed by a clear final answer.

Reasoning:""",
    recommended_temperature=0.7,  # Varies per pass in generate_multiple
)

CHAIN_OF_VERIFICATION = PromptTemplate(
    key="chain_of_verification",
    name="Chain-of-Verification (CoVe)",
    icon="✅",
    description="Ask the LLM to generate an answer, then verify each claim it made.",
    how_it_works="Reset if any previous approach is set. This is a two-phase approach: First, the LLM generates an answer. Then, it extracts claims from its own answer and evaluates each one for accuracy, flagging anything it's uncertain about.",
    why_it_helps="Self-verification catches **hallucinated facts** that the model generated confidently. By forcing the model to re-examine its claims, it often corrects errors it would otherwise leave unchecked.",
    example_boost="Expect self-corrected answers with flagged uncertainties",
    system_prompt="You are a meticulous fact-checker and analyst. Your job is to provide accurate answers and then rigorously verify every claim you make.",
    user_prompt_template="""Question: {question}

Please follow this exact process:

## PHASE 1: Initial Answer
Provide your best answer to the question.

## PHASE 2: Claim Extraction
List every factual claim you made in your answer as bullet points.

## PHASE 3: Verification
For each claim, evaluate:
- ✅ VERIFIED: I'm confident this is accurate
- ⚠️ UNCERTAIN: I'm not fully sure about this
- ❌ LIKELY INCORRECT: This might be wrong

## PHASE 4: Corrected Answer
Based on your verification, provide a revised answer, removing or correcting any uncertain/incorrect claims.""",
    recommended_temperature=0.3,
)

TREE_OF_THOUGHT = PromptTemplate(
    key="tree_of_thought",
    name="Tree-of-Thought (ToT)",
    icon="🌳",
    description="Explore multiple reasoning branches in parallel, evaluate each, and pick the best.",
    how_it_works="Instead of a single reasoning chain, the model explores 3 different approaches simultaneously. Each branch is evaluated for soundness, and the best path is selected for the final answer.",
    why_it_helps="Complex problems often have **multiple valid approaches**. By exploring several paths and comparing them, the model avoids getting locked into a flawed reasoning chain. This is the most robust technique for difficult problems.",
    example_boost="Expect the most thorough analysis for complex problems",
    system_prompt="You are an expert problem solver who explores multiple approaches before settling on the best one. You think broadly and deeply.",
    user_prompt_template="""Question: {question}

Explore this problem using three different reasoning approaches:

## Approach 1: [Name this approach]
- Reasoning: ...
- Conclusion: ...
- Confidence: high/medium/low

## Approach 2: [Name this approach]
- Reasoning: ...
- Conclusion: ...
- Confidence: high/medium/low

## Approach 3: [Name this approach]
- Reasoning: ...
- Conclusion: ...
- Confidence: high/medium/low

## Evaluation
Compare the three approaches. Which reasoning is most sound and why?

## Final Answer
Based on the best approach, provide your definitive answer.""",
    recommended_temperature=0.5,
)


# ── Template Registry ────────────────────────────────────────────────

ALL_TEMPLATES: dict[str, PromptTemplate] = {
    "zero_shot": ZERO_SHOT,
    "role_prompting": ROLE_PROMPTING,
    "few_shot": FEW_SHOT,
    "chain_of_thought": CHAIN_OF_THOUGHT,
    "structured_output": STRUCTURED_OUTPUT,
    "self_consistency": SELF_CONSISTENCY,
    "chain_of_verification": CHAIN_OF_VERIFICATION,
    "tree_of_thought": TREE_OF_THOUGHT,
}


def get_template(technique_key: str) -> PromptTemplate:
    """Get a template by its key."""
    if technique_key not in ALL_TEMPLATES:
        raise ValueError(f"Unknown technique: {technique_key}. Available: {list(ALL_TEMPLATES.keys())}")
    return ALL_TEMPLATES[technique_key]


def format_prompt(technique_key: str, question: str, **kwargs) -> tuple[str, str]:
    """
    Format a prompt for the given technique and question.

    Returns:
        (system_prompt, user_prompt) tuple
    """
    template = get_template(technique_key)
    user_prompt = template.user_prompt_template.format(question=question, **kwargs)
    return template.system_prompt, user_prompt


def list_techniques() -> list[dict]:
    """List all available techniques with their metadata."""
    return [
        {
            "key": t.key,
            "name": t.name,
            "icon": t.icon,
            "description": t.description,
        }
        for t in ALL_TEMPLATES.values()
    ]
