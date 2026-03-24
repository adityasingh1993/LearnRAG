"""
PromptCraft — Application Configuration
"""

# ─── LLM Settings ───────────────────────────────────────────────
MODEL_NAME = "google/gemini-2.0-flash-001"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0            # Nucleus sampling parameter
DEFAULT_TOP_K = None           # Top-k sampling parameter
LOW_TEMPERATURE = 0.2          # For factual / deterministic tasks
HIGH_TEMPERATURE = 1.0         # For creative tasks
MAX_OUTPUT_TOKENS = 2048
SELF_CONSISTENCY_RUNS = 3      # Number of passes for self-consistency

AVAILABLE_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash-preview",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-4-scout:free",
    "mistralai/mistral-7b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen-2.5-72b-instruct:free",
]

# ─── Evaluation Settings ────────────────────────────────────────
JUDGE_MODEL = "google/gemini-2.0-flash-001"
JUDGE_TEMPERATURE = 0.1        # Low temp for consistent evaluation

# ─── Prompting Techniques ───────────────────────────────────────
TECHNIQUE_ORDER = [
    "zero_shot",
    "role_prompting",
    "few_shot",
    "chain_of_thought",
    "structured_output",
    "self_consistency",
    "chain_of_verification",
    "tree_of_thought",
]

TECHNIQUE_DISPLAY_NAMES = {
    "zero_shot": "Zero-Shot",
    "role_prompting": "Role Prompting",
    "few_shot": "Few-Shot",
    "chain_of_thought": "Chain-of-Thought (CoT)",
    "structured_output": "Structured Output",
    "self_consistency": "Self-Consistency",
    "chain_of_verification": "Chain-of-Verification (CoVe)",
    "tree_of_thought": "Tree-of-Thought (ToT)",
}

TECHNIQUE_ICONS = {
    "zero_shot": "🎯",
    "role_prompting": "🎭",
    "few_shot": "📝",
    "chain_of_thought": "🔗",
    "structured_output": "📐",
    "self_consistency": "🔄",
    "chain_of_verification": "✅",
    "tree_of_thought": "🌳",
}

# ─── Scenario Categories ────────────────────────────────────────
SCENARIO_CATEGORIES = [
    "Factual Q&A",
    "Math & Logic",
    "Code Generation",
    "Medical / Legal",
    "Creative Writing",
    "Multi-step Reasoning",
]

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

# ─── UI Settings ────────────────────────────────────────────────
APP_TITLE = "PromptCraft"
APP_ICON = "🧠"
APP_DESCRIPTION = "Master LLM Prompting — See How Better Prompts Eliminate Hallucinations"

SCORE_COLORS = {
    "excellent": "#00E676",   # Green
    "good": "#76FF03",        # Light green
    "moderate": "#FFD600",    # Yellow
    "poor": "#FF9100",        # Orange
    "bad": "#FF1744",         # Red
}
