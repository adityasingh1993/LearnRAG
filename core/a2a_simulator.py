"""
Agent-to-Agent (A2A) Protocol educational simulator.
Simulates Agent Cards, Tasks, Messages, Artifacts, and multi-agent
communication for interactive learning.

Reference: https://google.github.io/A2A/
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
#  Agent Card
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentSkill:
    """A capability that an agent advertises."""
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    """
    JSON metadata describing an agent's capabilities.
    Published at /.well-known/agent.json
    """
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    skills: list[AgentSkill] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])
    supports_streaming: bool = False
    supports_push_notifications: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": {
                "streaming": self.supports_streaming,
                "pushNotifications": self.supports_push_notifications,
            },
            "skills": [
                {"id": s.id, "name": s.name, "description": s.description,
                 "tags": s.tags, "examples": s.examples}
                for s in self.skills
            ],
            "defaultInputModes": self.input_modes,
            "defaultOutputModes": self.output_modes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  Task Lifecycle
# ═══════════════════════════════════════════════════════════════════════════

class TaskState(Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class TextPart:
    text: str
    type: str = "text"

    def to_dict(self) -> dict:
        return {"type": self.type, "text": self.text}


@dataclass
class FilePart:
    name: str
    mime_type: str
    data: str
    type: str = "file"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "file": {"name": self.name, "mimeType": self.mime_type, "bytes": self.data},
        }


@dataclass
class DataPart:
    data: dict
    type: str = "data"

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data}


@dataclass
class Message:
    """A message in the A2A conversation."""
    role: str
    parts: list[TextPart | FilePart | DataPart] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def text_content(self) -> str:
        return " ".join(p.text for p in self.parts if isinstance(p, TextPart))

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
            "metadata": self.metadata,
        }


@dataclass
class Artifact:
    """An output produced by an agent during task execution."""
    name: str
    description: str
    parts: list[TextPart | FilePart | DataPart] = field(default_factory=list)
    index: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parts": [p.to_dict() for p in self.parts],
            "index": self.index,
        }


@dataclass
class Task:
    """A unit of work in the A2A protocol."""
    id: str = ""
    session_id: str = ""
    state: TaskState = TaskState.SUBMITTED
    messages: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.session_id:
            self.session_id = str(uuid.uuid4())[:12]

    def transition(self, new_state: TaskState, detail: str = ""):
        old = self.state
        self.state = new_state
        self.history.append({
            "from": old.value, "to": new_state.value,
            "detail": detail, "timestamp": time.time(),
        })

    def add_message(self, role: str, text: str, **metadata):
        self.messages.append(Message(
            role=role,
            parts=[TextPart(text=text)],
            metadata=metadata,
        ))

    def add_artifact(self, name: str, description: str, content: str):
        self.artifacts.append(Artifact(
            name=name,
            description=description,
            parts=[TextPart(text=content)],
            index=len(self.artifacts),
        ))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sessionId": self.session_id,
            "status": {"state": self.state.value},
            "messages": [m.to_dict() for m in self.messages],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": self.metadata,
            "history": self.history,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  A2A Agent
# ═══════════════════════════════════════════════════════════════════════════

class A2AAgent:
    """Simulated A2A-compatible agent."""

    def __init__(self, card: AgentCard, handler: Callable[[Task], Task] | None = None):
        self.card = card
        self._handler = handler or self._default_handler
        self._tasks: dict[str, Task] = {}

    def send_task(self, user_message: str, session_id: str = "") -> Task:
        task = Task(session_id=session_id or str(uuid.uuid4())[:12])
        task.add_message("user", user_message)
        task.transition(TaskState.WORKING, "Processing user request")
        self._tasks[task.id] = task

        try:
            task = self._handler(task)
            if task.state == TaskState.WORKING:
                task.transition(TaskState.COMPLETED, "Task finished successfully")
        except Exception as e:
            task.transition(TaskState.FAILED, str(e))
            task.add_message("agent", f"Error: {e}")

        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> Task | None:
        task = self._tasks.get(task_id)
        if task and task.state in (TaskState.SUBMITTED, TaskState.WORKING):
            task.transition(TaskState.CANCELED, "Canceled by user")
        return task

    @staticmethod
    def _default_handler(task: Task) -> Task:
        user_text = task.messages[-1].text_content() if task.messages else ""
        task.add_message("agent", f"Received: '{user_text}'. This is a default response.")
        return task


# ═══════════════════════════════════════════════════════════════════════════
#  Agent Registry (Discovery)
# ═══════════════════════════════════════════════════════════════════════════

class AgentRegistry:
    """Registry for discovering A2A agents by skill or name."""

    def __init__(self):
        self._agents: dict[str, A2AAgent] = {}

    def register(self, agent: A2AAgent):
        self._agents[agent.card.name] = agent

    def discover(self, query: str = "") -> list[AgentCard]:
        if not query:
            return [a.card for a in self._agents.values()]
        query_lower = query.lower()
        results = []
        for agent in self._agents.values():
            card = agent.card
            if (query_lower in card.name.lower()
                or query_lower in card.description.lower()
                or any(query_lower in s.name.lower() or any(query_lower in t for t in s.tags) for s in card.skills)):
                results.append(card)
        return results

    def get_agent(self, name: str) -> A2AAgent | None:
        return self._agents.get(name)

    def route_task(self, user_message: str) -> tuple[A2AAgent | None, str]:
        """Simple keyword-based routing to the best agent."""
        msg_lower = user_message.lower()
        best_agent = None
        best_score = 0
        reason = ""

        for agent in self._agents.values():
            score = 0
            for skill in agent.card.skills:
                for tag in skill.tags:
                    if tag.lower() in msg_lower:
                        score += 2
                if any(word in msg_lower for word in skill.name.lower().split()):
                    score += 1
            if any(word in msg_lower for word in agent.card.description.lower().split()):
                score += 1

            if score > best_score:
                best_score = score
                best_agent = agent
                reason = f"Matched {score} keywords from '{agent.card.name}' skills"

        if best_agent is None and self._agents:
            best_agent = list(self._agents.values())[0]
            reason = f"Default routing to '{best_agent.card.name}'"

        return best_agent, reason


# ═══════════════════════════════════════════════════════════════════════════
#  Pre-built Demo Agents
# ═══════════════════════════════════════════════════════════════════════════

def _math_handler(task: Task) -> Task:
    user_text = task.messages[-1].text_content() if task.messages else ""
    import re, math
    numbers = re.findall(r'[\d.]+', user_text)
    if "add" in user_text.lower() or "sum" in user_text.lower() or "+" in user_text:
        result = sum(float(n) for n in numbers)
        task.add_message("agent", f"The sum is: {result}")
        task.add_artifact("calculation", "Math result", f"Sum of {numbers} = {result}")
    elif "multiply" in user_text.lower() or "*" in user_text or "product" in user_text.lower():
        result = 1.0
        for n in numbers:
            result *= float(n)
        task.add_message("agent", f"The product is: {result}")
        task.add_artifact("calculation", "Math result", f"Product of {numbers} = {result}")
    elif "sqrt" in user_text.lower() and numbers:
        result = math.sqrt(float(numbers[0]))
        task.add_message("agent", f"The square root of {numbers[0]} is: {result:.4f}")
        task.add_artifact("calculation", "Math result", f"√{numbers[0]} = {result:.4f}")
    else:
        task.add_message("agent", f"I can help with math! I found numbers: {numbers}. Try asking me to add, multiply, or find square roots.")
    return task


def _writer_handler(task: Task) -> Task:
    user_text = task.messages[-1].text_content() if task.messages else ""
    if "email" in user_text.lower():
        task.add_message("agent", "Here's a draft email based on your request:")
        task.add_artifact("email_draft", "Generated email",
                         f"Subject: Re: Your Request\n\nDear Recipient,\n\nThank you for reaching out. "
                         f"Based on your request about '{user_text[:50]}...', I'd like to provide the following...\n\n"
                         f"Best regards")
    elif "summary" in user_text.lower() or "summarize" in user_text.lower():
        task.add_message("agent", "Here's a summary:")
        task.add_artifact("summary", "Text summary",
                         f"Summary of the provided content: The text discusses key points related to '{user_text[:50]}...'")
    else:
        task.add_message("agent", f"I can help with writing tasks! You asked about: '{user_text[:80]}'")
        task.add_artifact("draft", "Written content",
                         f"Draft content addressing: {user_text[:100]}")
    return task


def _research_handler(task: Task) -> Task:
    user_text = task.messages[-1].text_content() if task.messages else ""
    task.add_message("agent",
                     f"I've researched '{user_text[:60]}'. Here are my findings:")
    task.add_artifact("research_report", "Research findings",
                     f"Research Report: {user_text[:60]}\n\n"
                     f"1. Overview: This topic covers important aspects of {user_text[:30]}...\n"
                     f"2. Key Findings: Multiple sources confirm the significance of this area.\n"
                     f"3. Recommendations: Further investigation is suggested for specific use cases.\n"
                     f"4. Sources: Academic papers, industry reports, and documentation.")
    return task


def create_demo_agents() -> dict[str, A2AAgent]:
    """Create a set of demo A2A agents."""
    math_agent = A2AAgent(
        card=AgentCard(
            name="Math Agent",
            description="Performs mathematical calculations and analysis",
            url="http://localhost:8001",
            skills=[
                AgentSkill("calc", "Calculator", "Arithmetic operations", ["math", "calculate", "add", "multiply", "sum"], ["Add 5 and 3", "Multiply 12 by 7"]),
                AgentSkill("analysis", "Number Analysis", "Statistical analysis", ["statistics", "average", "sqrt"], ["What is sqrt of 144?"]),
            ],
        ),
        handler=_math_handler,
    )

    writer_agent = A2AAgent(
        card=AgentCard(
            name="Writer Agent",
            description="Helps with writing tasks: emails, summaries, reports",
            url="http://localhost:8002",
            skills=[
                AgentSkill("email", "Email Writer", "Draft professional emails", ["email", "write", "draft", "compose"], ["Write an email to the team"]),
                AgentSkill("summary", "Summarizer", "Summarize text content", ["summary", "summarize", "condense"], ["Summarize this article"]),
            ],
            supports_streaming=True,
        ),
        handler=_writer_handler,
    )

    research_agent = A2AAgent(
        card=AgentCard(
            name="Research Agent",
            description="Conducts research and gathers information on topics",
            url="http://localhost:8003",
            skills=[
                AgentSkill("search", "Topic Research", "Research any topic", ["research", "search", "find", "investigate", "learn"], ["Research AI agents"]),
                AgentSkill("compare", "Comparison", "Compare concepts or products", ["compare", "versus", "difference"], ["Compare Python vs Java"]),
            ],
            supports_push_notifications=True,
        ),
        handler=_research_handler,
    )

    return {
        "Math Agent": math_agent,
        "Writer Agent": writer_agent,
        "Research Agent": research_agent,
    }


def create_demo_registry() -> AgentRegistry:
    """Create a registry pre-loaded with demo agents."""
    registry = AgentRegistry()
    for agent in create_demo_agents().values():
        registry.register(agent)
    return registry
