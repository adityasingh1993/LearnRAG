"""
Tool system for AI Agents.
Defines the Tool interface, a registry, and built-in tools.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    function: Callable[..., str] | None = None
    category: str = "general"

    def run(self, **kwargs) -> str:
        if self.function is None:
            return f"Tool '{self.name}' has no implementation."
        try:
            return str(self.function(**kwargs))
        except Exception as e:
            return f"Error running tool '{self.name}': {e}"

    def schema_for_prompt(self) -> str:
        params = ", ".join(
            f"{p.name}: {p.type}" + ("" if p.required else " (optional)")
            for p in self.parameters
        )
        return f"{self.name}({params}) — {self.description}"


class ToolRegistry:
    """Central registry for available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def format_for_prompt(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"  - {t.schema_for_prompt()}")
        return "\n".join(lines)

    def run_tool(self, name: str, **kwargs) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Unknown tool: '{name}'. Available: {', '.join(self.tool_names())}"
        return tool.run(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  Built-in Tools
# ═══════════════════════════════════════════════════════════════════════════

def _calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/().% ")
    cleaned = expression.strip()

    safe_funcs = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "abs": abs, "round": round, "pow": pow,
        "pi": math.pi, "e": math.e,
    }

    for name in safe_funcs:
        cleaned = cleaned.replace(name, f"__{name}__")

    if not all(c in allowed or c.isalpha() or c == '_' for c in cleaned):
        return f"Invalid expression: {expression}"

    try:
        ns = {f"__{k}__": v for k, v in safe_funcs.items()}
        ns["__builtins__"] = {}
        result = eval(cleaned, ns)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def _datetime_now(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(format)


def _text_stats(text: str) -> str:
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return json.dumps({
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 1),
        "unique_words": len(set(w.lower() for w in words)),
    }, indent=2)


def _json_parse(data: str) -> str:
    try:
        parsed = json.loads(data)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


def _web_search(query: str) -> str:
    """Simulated web search — returns plausible results for educational purposes."""
    responses = {
        "weather": "Current weather: 22°C, partly cloudy. Forecast: mild temperatures expected this week.",
        "python": "Python is a high-level programming language. Latest version: 3.12. Created by Guido van Rossum.",
        "rag": "RAG (Retrieval-Augmented Generation) combines retrieval with LLM generation for grounded answers.",
        "agent": "AI Agents are systems that use LLMs to reason, plan, and take actions using tools to accomplish goals.",
        "llm": "Large Language Models (LLMs) are neural networks trained on text data. Examples: GPT-4, Claude, Llama.",
    }
    query_lower = query.lower()
    for keyword, response in responses.items():
        if keyword in query_lower:
            return f"Search results for '{query}':\n{response}"
    return (
        f"Search results for '{query}':\n"
        f"Found multiple results. Key takeaway: {query} is a topic with extensive "
        f"documentation available online. Please refine your query for specific information."
    )


def _unit_convert(value: str, from_unit: str, to_unit: str) -> str:
    """Convert between common units."""
    conversions = {
        ("km", "miles"): 0.621371, ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462, ("lbs", "kg"): 0.453592,
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("meters", "feet"): 3.28084, ("feet", "meters"): 0.3048,
        ("liters", "gallons"): 0.264172, ("gallons", "liters"): 3.78541,
    }
    try:
        val = float(value)
    except ValueError:
        return f"Invalid number: {value}"

    key = (from_unit.lower(), to_unit.lower())
    factor = conversions.get(key)
    if factor is None:
        return f"Unsupported conversion: {from_unit} → {to_unit}"

    if callable(factor):
        result = factor(val)
    else:
        result = val * factor
    return f"{val} {from_unit} = {result:.4f} {to_unit}"


def _string_transform(text: str, operation: str) -> str:
    """Apply a string transformation."""
    ops = {
        "uppercase": text.upper(),
        "lowercase": text.lower(),
        "title": text.title(),
        "reverse": text[::-1],
        "word_count": str(len(text.split())),
        "char_count": str(len(text)),
        "strip": text.strip(),
    }
    result = ops.get(operation.lower())
    if result is None:
        return f"Unknown operation: {operation}. Available: {', '.join(ops.keys())}"
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Tool Definitions
# ═══════════════════════════════════════════════════════════════════════════

BUILTIN_TOOLS: dict[str, Tool] = {
    "calculator": Tool(
        name="calculator",
        description="Evaluate a mathematical expression. Supports +, -, *, /, sqrt, sin, cos, log, pi, etc.",
        parameters=[ToolParameter("expression", "string", "The math expression to evaluate")],
        function=lambda expression: _calculator(expression),
        category="math",
    ),
    "datetime": Tool(
        name="datetime",
        description="Get the current date and time.",
        parameters=[ToolParameter("format", "string", "Date format string (e.g. '%Y-%m-%d')", required=False)],
        function=lambda format="%Y-%m-%d %H:%M:%S": _datetime_now(format),
        category="utility",
    ),
    "text_stats": Tool(
        name="text_stats",
        description="Analyse text and return statistics: character count, word count, sentence count, etc.",
        parameters=[ToolParameter("text", "string", "The text to analyse")],
        function=lambda text: _text_stats(text),
        category="text",
    ),
    "web_search": Tool(
        name="web_search",
        description="Search the web for information on any topic. Returns a summary of search results.",
        parameters=[ToolParameter("query", "string", "The search query")],
        function=lambda query: _web_search(query),
        category="search",
    ),
    "json_parse": Tool(
        name="json_parse",
        description="Parse and pretty-print a JSON string. Validates the JSON and formats it.",
        parameters=[ToolParameter("data", "string", "The JSON string to parse")],
        function=lambda data: _json_parse(data),
        category="utility",
    ),
    "unit_convert": Tool(
        name="unit_convert",
        description="Convert a value between units (km/miles, kg/lbs, celsius/fahrenheit, meters/feet, liters/gallons).",
        parameters=[
            ToolParameter("value", "string", "The numeric value to convert"),
            ToolParameter("from_unit", "string", "Source unit"),
            ToolParameter("to_unit", "string", "Target unit"),
        ],
        function=lambda value, from_unit, to_unit: _unit_convert(value, from_unit, to_unit),
        category="math",
    ),
    "string_transform": Tool(
        name="string_transform",
        description="Transform a string: uppercase, lowercase, title, reverse, word_count, char_count, strip.",
        parameters=[
            ToolParameter("text", "string", "The text to transform"),
            ToolParameter("operation", "string", "The operation to apply"),
        ],
        function=lambda text, operation: _string_transform(text, operation),
        category="text",
    ),
}


def create_tool_registry(tool_names: list[str] | None = None) -> ToolRegistry:
    """Create a registry with selected (or all) built-in tools."""
    registry = ToolRegistry()
    names = tool_names or list(BUILTIN_TOOLS.keys())
    for name in names:
        if name in BUILTIN_TOOLS:
            registry.register(BUILTIN_TOOLS[name])
    return registry


def create_custom_tool(
    name: str,
    description: str,
    parameters: list[dict],
    code: str,
) -> Tool:
    """Create a tool from user-provided Python code (educational, sandboxed)."""
    params = [ToolParameter(**p) for p in parameters]

    def _run(**kwargs):
        ns = {"__builtins__": {"str": str, "int": int, "float": float, "len": len, "range": range, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round}}
        ns.update(kwargs)
        exec(code, ns)  # noqa: S102
        return str(ns.get("result", "No 'result' variable set in tool code."))

    return Tool(name=name, description=description, parameters=params, function=_run, category="custom")
