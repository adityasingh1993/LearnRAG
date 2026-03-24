"""
Agent execution engine.
Implements ReAct, Plan-and-Execute, Reflection, and ToolChoice patterns.
Each pattern uses LLM reasoning + tool calls in a loop.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.tools import ToolRegistry
from core.llm_providers import LLMProvider
from core.token_tracker import TurnTokenUsage, StepTokenUsage, count_tokens


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentStep:
    """One step in the agent's reasoning chain."""
    step_number: int
    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    is_final: bool = False
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """Complete result from an agent run."""
    answer: str
    steps: list[AgentStep]
    total_duration_ms: float
    pattern: str
    tools_used: list[str] = field(default_factory=list)
    token_usage: TurnTokenUsage | None = None


AGENT_PATTERNS = {
    "react": "ReAct — Reason + Act loop: think, pick a tool, observe, repeat until done",
    "plan_execute": "Plan-and-Execute — create a step-by-step plan first, then execute each step",
    "reflection": "Reflection — generate an answer, critique it, then refine",
    "tool_choice": "Tool Choice — pick the single best tool and use it once",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Templates
# ═══════════════════════════════════════════════════════════════════════════

_REACT_SYSTEM = """You are a helpful AI assistant with access to tools. Use them to answer the user's question.

Available tools:
{tools}

You MUST respond in EXACTLY this format for each step:

Thought: <your reasoning about what to do next>
Action: <tool name, or "finish" if you have the answer>
Action Input: {{"param": "value"}}

If Action is "finish", put your final answer in Action Input like:
Action Input: {{"answer": "your final answer here"}}

Important rules:
- Always start with a Thought
- Use tools when you need information you don't have
- When you have enough information, use Action: finish
- Keep Action Input as valid JSON"""

_REACT_USER = """Question: {question}

{history}Begin reasoning step by step."""

_PLAN_SYSTEM = """You are a planning AI assistant. Create a step-by-step plan to answer the user's question, then execute each step.

Available tools:
{tools}

Phase 1 — PLAN: List numbered steps needed to answer the question.
Respond with:
Plan:
1. <step description>
2. <step description>
...

Phase 2 — EXECUTE: For each step, respond with:
Step N:
Thought: <reasoning>
Action: <tool name or "think" for reasoning-only steps>
Action Input: {{"param": "value"}}

When all steps are done, respond with:
Final Answer: <your complete answer>"""

_REFLECTION_SYSTEM = """You are a thoughtful AI assistant. Answer the question, then critically evaluate your answer and improve it.

Available tools:
{tools}

Follow this process:
1. Initial Answer — give your best answer using available tools
2. Critique — identify weaknesses, gaps, or errors in your answer
3. Refined Answer — improve the answer based on your critique

Respond in this format:

INITIAL ANSWER:
<your first answer>

CRITIQUE:
<what's wrong or could be improved>

REFINED ANSWER:
<your improved final answer>"""

_TOOL_CHOICE_SYSTEM = """You are a helpful AI assistant. Pick the SINGLE best tool to answer the question, use it, then give your answer.

Available tools:
{tools}

Respond in this format:
Thought: <why you're choosing this tool>
Action: <tool name>
Action Input: {{"param": "value"}}

After seeing the tool result, give your final answer."""


# ═══════════════════════════════════════════════════════════════════════════
#  Agent Executor
# ═══════════════════════════════════════════════════════════════════════════

class AgentExecutor:
    """Runs an agent loop with the specified pattern."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        pattern: str = "react",
        max_steps: int = 8,
        verbose: bool = True,
    ):
        self.llm = llm_provider
        self.tools = tool_registry
        self.pattern = pattern
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, question: str, history: str = "") -> AgentResult:
        start = time.time()
        dispatch = {
            "react": self._run_react,
            "plan_execute": self._run_plan_execute,
            "reflection": self._run_reflection,
            "tool_choice": self._run_tool_choice,
        }
        runner = dispatch.get(self.pattern, self._run_react)
        result = runner(question, history)
        result.total_duration_ms = (time.time() - start) * 1000
        result.pattern = self.pattern
        return result

    # ── ReAct ─────────────────────────────────────────────────────────────

    def _run_react(self, question: str, history: str) -> AgentResult:
        system = _REACT_SYSTEM.format(tools=self.tools.format_for_prompt())
        user_msg = _REACT_USER.format(question=question, history=history)

        messages = [system, user_msg]
        steps: list[AgentStep] = []
        tools_used: list[str] = []
        conversation = user_msg

        for step_num in range(1, self.max_steps + 1):
            t0 = time.time()
            prompt = system + "\n\n" + conversation
            resp = self.llm.generate(prompt, temperature=0.1, max_tokens=400)
            text = resp.text.strip()
            duration = (time.time() - t0) * 1000

            thought = self._extract(text, "Thought")
            action = self._extract(text, "Action")
            action_input_raw = self._extract(text, "Action Input")

            action_input = self._parse_json(action_input_raw)

            if action and action.lower() == "finish":
                answer = action_input.get("answer", action_input_raw) if isinstance(action_input, dict) else action_input_raw
                steps.append(AgentStep(
                    step_number=step_num, thought=thought,
                    action="finish", action_input={"answer": answer},
                    is_final=True, duration_ms=duration,
                ))
                return AgentResult(answer=str(answer), steps=steps, total_duration_ms=0, pattern="react", tools_used=tools_used)

            observation = ""
            if action:
                kwargs = action_input if isinstance(action_input, dict) else {}
                observation = self.tools.run_tool(action, **kwargs)
                tools_used.append(action)

            steps.append(AgentStep(
                step_number=step_num, thought=thought,
                action=action, action_input=action_input,
                observation=observation, duration_ms=duration,
            ))

            conversation += f"\n{text}\nObservation: {observation}\n"

        final = steps[-1].thought if steps else "Could not determine an answer."
        return AgentResult(answer=final, steps=steps, total_duration_ms=0, pattern="react", tools_used=tools_used)

    # ── Plan-and-Execute ──────────────────────────────────────────────────

    def _run_plan_execute(self, question: str, history: str) -> AgentResult:
        system = _PLAN_SYSTEM.format(tools=self.tools.format_for_prompt())
        steps: list[AgentStep] = []
        tools_used: list[str] = []

        # Phase 1: Plan
        t0 = time.time()
        plan_prompt = f"{system}\n\nQuestion: {question}\n{history}\nCreate your plan:"
        plan_resp = self.llm.generate(plan_prompt, temperature=0.1, max_tokens=400)
        plan_text = plan_resp.text.strip()
        duration = (time.time() - t0) * 1000

        plan_steps = re.findall(r'\d+\.\s*(.+)', plan_text)
        if not plan_steps:
            plan_steps = [line.strip() for line in plan_text.split("\n") if line.strip() and not line.strip().startswith("Plan")]

        steps.append(AgentStep(
            step_number=0, thought=f"Created plan with {len(plan_steps)} steps",
            action="plan", action_input={"steps": plan_steps},
            observation=plan_text, duration_ms=duration,
        ))

        # Phase 2: Execute each step
        context = f"Question: {question}\nPlan:\n{plan_text}\n\nExecuting steps:\n"
        for i, plan_step in enumerate(plan_steps[:self.max_steps], 1):
            t0 = time.time()
            exec_prompt = (
                f"{system}\n\n{context}\n"
                f"Execute step {i}: {plan_step}\n"
                f"Respond with Thought, Action, and Action Input."
            )
            exec_resp = self.llm.generate(exec_prompt, temperature=0.1, max_tokens=300)
            exec_text = exec_resp.text.strip()
            duration = (time.time() - t0) * 1000

            thought = self._extract(exec_text, "Thought") or plan_step
            action = self._extract(exec_text, "Action")
            action_input = self._parse_json(self._extract(exec_text, "Action Input"))

            observation = ""
            if action and action.lower() not in ("think", "reason", "none", "finish"):
                kwargs = action_input if isinstance(action_input, dict) else {}
                observation = self.tools.run_tool(action, **kwargs)
                tools_used.append(action)
            elif action and action.lower() == "finish":
                observation = exec_text

            steps.append(AgentStep(
                step_number=i, thought=thought, action=action,
                action_input=action_input, observation=observation,
                duration_ms=duration,
            ))
            context += f"\nStep {i} result: {observation or thought}\n"

        # Final synthesis
        t0 = time.time()
        synthesis_prompt = f"{context}\n\nAll steps complete. Give a Final Answer to: {question}"
        synth_resp = self.llm.generate(synthesis_prompt, temperature=0.1, max_tokens=500)
        answer = synth_resp.text.strip()
        duration = (time.time() - t0) * 1000

        final_answer = answer
        if "Final Answer:" in answer:
            final_answer = answer.split("Final Answer:", 1)[1].strip()

        steps.append(AgentStep(
            step_number=len(steps), thought="Synthesising final answer",
            action="finish", action_input={"answer": final_answer},
            is_final=True, duration_ms=duration,
        ))

        return AgentResult(answer=final_answer, steps=steps, total_duration_ms=0, pattern="plan_execute", tools_used=tools_used)

    # ── Reflection ────────────────────────────────────────────────────────

    def _run_reflection(self, question: str, history: str) -> AgentResult:
        system = _REFLECTION_SYSTEM.format(tools=self.tools.format_for_prompt())
        steps: list[AgentStep] = []
        tools_used: list[str] = []

        # Step 1: Initial answer (optionally with tool use)
        t0 = time.time()
        tool_prompt = (
            f"{_REACT_SYSTEM.format(tools=self.tools.format_for_prompt())}\n\n"
            f"Question: {question}\n{history}\nUse a tool if helpful, then provide your answer."
        )
        tool_resp = self.llm.generate(tool_prompt, temperature=0.2, max_tokens=400)
        tool_text = tool_resp.text.strip()
        duration = (time.time() - t0) * 1000

        action = self._extract(tool_text, "Action")
        observation = ""
        if action and action.lower() not in ("finish", "none"):
            action_input = self._parse_json(self._extract(tool_text, "Action Input"))
            kwargs = action_input if isinstance(action_input, dict) else {}
            observation = self.tools.run_tool(action, **kwargs)
            tools_used.append(action)

        steps.append(AgentStep(
            step_number=1, thought="Gathering information",
            action=action, action_input=self._parse_json(self._extract(tool_text, "Action Input")),
            observation=observation, duration_ms=duration,
        ))

        # Step 2: Full reflection cycle
        t0 = time.time()
        reflection_prompt = (
            f"{system}\n\nQuestion: {question}\n{history}"
            + (f"\nTool output: {observation}" if observation else "")
            + "\n\nFollow the reflection process."
        )
        refl_resp = self.llm.generate(reflection_prompt, temperature=0.2, max_tokens=800)
        refl_text = refl_resp.text.strip()
        duration = (time.time() - t0) * 1000

        initial = self._extract_section(refl_text, "INITIAL ANSWER")
        critique = self._extract_section(refl_text, "CRITIQUE")
        refined = self._extract_section(refl_text, "REFINED ANSWER")

        if initial:
            steps.append(AgentStep(
                step_number=2, thought=f"Initial answer: {initial[:200]}",
                action="think", observation=initial, duration_ms=duration / 3,
            ))
        if critique:
            steps.append(AgentStep(
                step_number=3, thought=f"Critique: {critique[:200]}",
                action="critique", observation=critique, duration_ms=duration / 3,
            ))

        final_answer = refined or initial or refl_text
        steps.append(AgentStep(
            step_number=len(steps) + 1, thought="Refined answer ready",
            action="finish", action_input={"answer": final_answer},
            is_final=True, duration_ms=duration / 3,
        ))

        return AgentResult(answer=final_answer, steps=steps, total_duration_ms=0, pattern="reflection", tools_used=tools_used)

    # ── Tool Choice ───────────────────────────────────────────────────────

    def _run_tool_choice(self, question: str, history: str) -> AgentResult:
        system = _TOOL_CHOICE_SYSTEM.format(tools=self.tools.format_for_prompt())
        steps: list[AgentStep] = []
        tools_used: list[str] = []

        # Pick and use tool
        t0 = time.time()
        prompt = f"{system}\n\nQuestion: {question}\n{history}"
        resp = self.llm.generate(prompt, temperature=0.1, max_tokens=300)
        text = resp.text.strip()
        duration = (time.time() - t0) * 1000

        thought = self._extract(text, "Thought")
        action = self._extract(text, "Action")
        action_input = self._parse_json(self._extract(text, "Action Input"))

        observation = ""
        if action:
            kwargs = action_input if isinstance(action_input, dict) else {}
            observation = self.tools.run_tool(action, **kwargs)
            tools_used.append(action)

        steps.append(AgentStep(
            step_number=1, thought=thought, action=action,
            action_input=action_input, observation=observation,
            duration_ms=duration,
        ))

        # Generate final answer
        t0 = time.time()
        answer_prompt = (
            f"Question: {question}\n"
            f"Tool used: {action}\n"
            f"Tool result: {observation}\n\n"
            f"Give a clear, complete answer to the question based on the tool result."
        )
        answer_resp = self.llm.generate(answer_prompt, temperature=0.2, max_tokens=400)
        final_answer = answer_resp.text.strip()
        duration = (time.time() - t0) * 1000

        steps.append(AgentStep(
            step_number=2, thought="Formulating final answer from tool output",
            action="finish", action_input={"answer": final_answer},
            is_final=True, duration_ms=duration,
        ))

        return AgentResult(answer=final_answer, steps=steps, total_duration_ms=0, pattern="tool_choice", tools_used=tools_used)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract(text: str, field: str) -> str:
        pattern = rf"{field}:\s*(.+?)(?=\n(?:Thought|Action|Observation|$)|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_section(text: str, header: str) -> str:
        pattern = rf"{header}:\s*\n?(.*?)(?=\n[A-Z ]+:|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _parse_json(raw: str) -> dict | str:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return raw
