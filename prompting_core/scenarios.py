"""
PromptCraft — Scenarios
Pre-built challenge scenarios with known ground truth answers.
"""

from dataclasses import dataclass


@dataclass
class Scenario:
    """A challenge scenario for testing prompting techniques."""
    id: str
    title: str
    question: str
    ground_truth: str
    category: str
    difficulty: str
    why_interesting: str
    best_technique: str  # Which technique is expected to shine


ALL_SCENARIOS: list[Scenario] = [

    # ── Factual Q&A ─────────────────────────────────────────────
    Scenario(
        id="factual_01",
        title="African Lakes",
        question="What are the 5 largest lakes in Africa by surface area? List them with their approximate areas.",
        ground_truth="1. Lake Victoria (~68,870 km²), 2. Lake Tanganyika (~32,600 km²), 3. Lake Malawi/Nyasa (~29,600 km²), 4. Lake Turkana (~6,405 km²), 5. Lake Albert (~5,300 km²)",
        category="Factual Q&A",
        difficulty="Medium",
        why_interesting="LLMs often hallucinate exact numbers and rankings. This tests whether the model gets the correct order and approximate areas.",
        best_technique="chain_of_verification",
    ),
    Scenario(
        id="factual_02",
        title="Nobel Prize History",
        question="Who won the Nobel Prize in Physics in 2020 and what was it awarded for?",
        ground_truth="The 2020 Nobel Prize in Physics was awarded to Roger Penrose (for discovering that black hole formation is a robust prediction of general relativity), and jointly to Reinhard Genzel and Andrea Ghez (for the discovery of a supermassive compact object at the centre of our galaxy).",
        category="Factual Q&A",
        difficulty="Medium",
        why_interesting="Historical factual questions are prone to hallucination — models might confuse years or mix up laureates.",
        best_technique="chain_of_verification",
    ),
    Scenario(
        id="factual_03",
        title="Programming Language Origins",
        question="Who created the Python programming language and in what year was it first released?",
        ground_truth="Python was created by Guido van Rossum. It was first released in 1991 (version 0.9.0 in February 1991).",
        category="Factual Q&A",
        difficulty="Easy",
        why_interesting="Common knowledge question — good baseline to show that even simple facts can vary in specificity across prompting styles.",
        best_technique="zero_shot",
    ),

    # ── Math & Logic ────────────────────────────────────────────
    Scenario(
        id="math_01",
        title="The Sheep Riddle",
        question="A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        ground_truth="9 sheep are left. The phrase 'all but 9' means 9 survive.",
        category="Math & Logic",
        difficulty="Easy",
        why_interesting="A classic trick question. Zero-shot LLMs often calculate 17-9=8 instead of recognizing the wordplay. Chain-of-thought forces careful reading.",
        best_technique="chain_of_thought",
    ),
    Scenario(
        id="math_02",
        title="Crossing the River",
        question="A man needs to cross a river with a fox, a chicken, and a bag of grain. He can only carry one item at a time. If left alone, the fox will eat the chicken, and the chicken will eat the grain. How does he get everything across?",
        ground_truth="1) Take the chicken across. 2) Go back alone. 3) Take the fox across. 4) Bring the chicken back. 5) Take the grain across. 6) Go back alone. 7) Take the chicken across.",
        category="Math & Logic",
        difficulty="Medium",
        why_interesting="Requires multi-step logical planning. Models without CoT often miss the crucial step of bringing the chicken back.",
        best_technique="tree_of_thought",
    ),
    Scenario(
        id="math_03",
        title="The Bat and Ball",
        question="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        ground_truth="The ball costs $0.05. If the ball is $0.05, the bat is $1.05, totaling $1.10.",
        category="Math & Logic",
        difficulty="Easy",
        why_interesting="The classic cognitive bias trap. Most people (and LLMs) instinctively say $0.10. Chain-of-thought reasoning catches this.",
        best_technique="chain_of_thought",
    ),

    # ── Code Generation ─────────────────────────────────────────
    Scenario(
        id="code_01",
        title="Fibonacci Function",
        question="Write a Python function that returns the nth Fibonacci number. Handle edge cases like negative numbers and n=0.",
        ground_truth="A correct implementation should: handle n<0 (raise error or return appropriate value), return 0 for n=0, return 1 for n=1, and correctly compute fib(n) for larger values. Should avoid stack overflow for large n.",
        category="Code Generation",
        difficulty="Easy",
        why_interesting="Tests whether the LLM handles edge cases. Zero-shot often produces correct base logic but misses edge cases.",
        best_technique="structured_output",
    ),
    Scenario(
        id="code_02",
        title="Merge Sorted Arrays",
        question="Write a Python function to merge two sorted arrays into a single sorted array without using the built-in sort function. What is the time complexity?",
        ground_truth="Two-pointer approach with O(n+m) time complexity where n and m are array lengths. Should handle empty arrays and arrays of different lengths.",
        category="Code Generation",
        difficulty="Medium",
        why_interesting="Tests algorithmic understanding. LLMs sometimes hallucinate wrong time complexities or produce subtly buggy merge logic.",
        best_technique="chain_of_thought",
    ),

    # ── Medical / Legal ─────────────────────────────────────────
    Scenario(
        id="medical_01",
        title="Ibuprofen Side Effects",
        question="What are the most common side effects of ibuprofen, and when should someone avoid taking it?",
        ground_truth="Common side effects: stomach pain, nausea, vomiting, headache, dizziness, mild heartburn. Avoid if: history of stomach ulcers, kidney disease, heart disease, third trimester of pregnancy, allergy to NSAIDs, or taking blood thinners.",
        category="Medical / Legal",
        difficulty="Medium",
        why_interesting="High-stakes domain where hallucination is dangerous. Models might invent rare side effects or miss important contraindications.",
        best_technique="chain_of_verification",
    ),
    Scenario(
        id="medical_02",
        title="Symptoms of Stroke",
        question="What are the warning signs of a stroke, and what should a bystander do?",
        ground_truth="Warning signs (FAST): Face drooping, Arm weakness, Speech difficulty, Time to call emergency. Other signs: sudden numbness, confusion, trouble seeing, walking, severe headache. Bystander should call emergency services immediately and note the time symptoms started.",
        category="Medical / Legal",
        difficulty="Easy",
        why_interesting="Tests whether the model provides actionable, accurate medical information without adding fabricated advice.",
        best_technique="role_prompting",
    ),

    # ── Creative Writing ────────────────────────────────────────
    Scenario(
        id="creative_01",
        title="Quantum Haiku",
        question="Write a haiku (5-7-5 syllable structure) about quantum computing.",
        ground_truth="Any haiku that follows the 5-7-5 syllable structure and relates to quantum computing.",
        category="Creative Writing",
        difficulty="Easy",
        why_interesting="Tests constraint adherence. LLMs frequently get syllable counts wrong without structured prompting.",
        best_technique="structured_output",
    ),
    Scenario(
        id="creative_02",
        title="Explain Like I'm 5",
        question="Explain how the internet works to a 5-year-old child.",
        ground_truth="A good answer uses simple analogies (e.g., mail delivery, roads connecting houses) and avoids technical jargon.",
        category="Creative Writing",
        difficulty="Easy",
        why_interesting="Role prompting shines here — assigning the persona of a kindergarten teacher produces much better simplified explanations.",
        best_technique="role_prompting",
    ),

    # ── Multi-step Reasoning ────────────────────────────────────
    Scenario(
        id="reasoning_01",
        title="Transitive Logic",
        question="Alice is taller than Bob. Bob is taller than Charlie. Charlie is taller than Diana. Is Alice taller than Diana? Explain your reasoning.",
        ground_truth="Yes, Alice is taller than Diana. By transitive property: Alice > Bob > Charlie > Diana.",
        category="Multi-step Reasoning",
        difficulty="Easy",
        why_interesting="Tests basic transitive reasoning. Simple for humans but LLMs sometimes fumble without explicit step-by-step reasoning.",
        best_technique="chain_of_thought",
    ),
    Scenario(
        id="reasoning_02",
        title="Birthday Paradox",
        question="In a room of 23 people, what's the approximate probability that at least two people share the same birthday? Explain why this is counterintuitive.",
        ground_truth="About 50.7%. It's counterintuitive because we compare all pairs (253 pairs for 23 people), not just one person against the rest. The probability of NO shared birthday decreases multiplicatively: 364/365 × 363/365 × ... which drops to ~0.493.",
        category="Multi-step Reasoning",
        difficulty="Hard",
        why_interesting="Requires mathematical reasoning AND intuitive explanation. Tree-of-Thought helps explore both the math and the intuition.",
        best_technique="tree_of_thought",
    ),
    Scenario(
        id="reasoning_03",
        title="The Monty Hall Problem",
        question="In the Monty Hall problem, you pick door 1. The host opens door 3 (showing a goat). Should you switch to door 2 or stick with door 1? Why?",
        ground_truth="You should switch. Switching gives a 2/3 probability of winning, while sticking gives 1/3. When you first chose, you had a 1/3 chance. The host's reveal doesn't change your door's probability but concentrates the remaining 2/3 on the other unopened door.",
        category="Multi-step Reasoning",
        difficulty="Hard",
        why_interesting="A famously counterintuitive probability problem. LLMs often give the wrong answer or correct answer with wrong reasoning.",
        best_technique="tree_of_thought",
    ),
]


def get_scenario(scenario_id: str) -> Scenario:
    """Get a scenario by ID."""
    for s in ALL_SCENARIOS:
        if s.id == scenario_id:
            return s
    raise ValueError(f"Scenario '{scenario_id}' not found.")


def get_scenarios_by_category(category: str) -> list[Scenario]:
    """Get all scenarios in a category."""
    return [s for s in ALL_SCENARIOS if s.category == category]


def get_scenarios_by_difficulty(difficulty: str) -> list[Scenario]:
    """Get all scenarios by difficulty."""
    return [s for s in ALL_SCENARIOS if s.difficulty == difficulty]


def get_categories() -> list[str]:
    """Get unique categories."""
    return list(dict.fromkeys(s.category for s in ALL_SCENARIOS))


def get_recommended_scenario(technique_key: str) -> Scenario | None:
    """Get the best showcase scenario for a given technique."""
    for s in ALL_SCENARIOS:
        if s.best_technique == technique_key:
            return s
    return ALL_SCENARIOS[0] if ALL_SCENARIOS else None
