"""Page 24 — A2A Tasks: Task lifecycle, messages, artifacts."""

import streamlit as st
import json
import time
from core.a2a_simulator import (
    Task, TaskState, Message, TextPart, FilePart, DataPart,
    Artifact, create_demo_agents,
)

st.set_page_config(page_title="A2A Tasks", page_icon="📋", layout="wide")

st.title("📋 A2A Tasks & Communication")
st.markdown("Explore the task lifecycle, message passing, and artifact generation in the A2A protocol.")

# ── Task Lifecycle ──────────────────────────────────────────────────────
st.header("1 · Task Lifecycle")

st.markdown("""
A **Task** is the fundamental unit of work. It moves through well-defined states:
""")

st.markdown("""
```
        ┌───────────┐
        │ submitted │ ─── Task created by client
        └─────┬─────┘
              ▼
        ┌───────────┐
   ┌───►│  working  │ ─── Agent actively processing
   │    └──┬──┬──┬──┘
   │       │  │  │
   │       │  │  └──────────────►┌───────────┐
   │       │  │                  │  failed    │ ─── Error occurred
   │       │  │                  └───────────┘
   │       │  │
   │       │  └─────────────────►┌───────────┐
   │       │                     │ canceled   │ ─── Canceled by client
   │       │                     └───────────┘
   │       ▼
   │  ┌────────────────┐
   │  │ input-required │ ─── Agent needs more info
   │  └───────┬────────┘
   │          │ (client sends more input)
   └──────────┘
              │
              ▼
        ┌───────────┐
        │ completed │ ─── Task finished successfully
        └───────────┘
```
""")

# ── Interactive Task Demo ───────────────────────────────────────────────
st.header("2 · Interactive Task Demo")
st.markdown("Send a task to a demo agent and watch the lifecycle unfold.")

agents = create_demo_agents()
agent_name = st.selectbox(
    "Select an agent:",
    list(agents.keys()),
    key="task_agent",
)
agent = agents[agent_name]

st.markdown(f"**{agent.card.name}** — {agent.card.description}")
st.markdown(f"Skills: {', '.join(s.name for s in agent.card.skills)}")

user_message = st.text_input(
    "Your message to the agent:",
    placeholder="e.g., Add 5 and 3" if "Math" in agent_name else "e.g., Write an email about the meeting",
    key="task_msg",
)

if st.button("▶️ Send Task", key="send_task") and user_message:
    with st.status("Task execution...", expanded=True) as status:
        st.write("**State: SUBMITTED**")
        st.caption("Task created and queued")
        time.sleep(0.3)

        st.write("**State: WORKING**")
        st.caption("Agent is processing...")
        time.sleep(0.5)

        task = agent.send_task(user_message)

        if task.state == TaskState.COMPLETED:
            st.write("**State: COMPLETED** ✅")
            status.update(label="Task completed!", state="complete")
        elif task.state == TaskState.FAILED:
            st.write("**State: FAILED** ❌")
            status.update(label="Task failed!", state="error")

    st.subheader("Task Result")

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Task ID", task.id)
    tc2.metric("State", task.state.value)
    tc3.metric("Messages", len(task.messages))

    st.markdown("**Messages:**")
    for msg in task.messages:
        with st.chat_message("user" if msg.role == "user" else "assistant"):
            st.markdown(msg.text_content())

    if task.artifacts:
        st.markdown("**Artifacts:**")
        for artifact in task.artifacts:
            with st.expander(f"📦 {artifact.name} — {artifact.description}"):
                for part in artifact.parts:
                    if isinstance(part, TextPart):
                        st.code(part.text, language="text")

    st.markdown("**State History:**")
    for entry in task.history:
        st.markdown(f"- `{entry['from']}` → `{entry['to']}` — {entry.get('detail', '')}")

    st.markdown("**Full Task JSON:**")
    with st.expander("View JSON"):
        st.code(json.dumps(task.to_dict(), indent=2, default=str), language="json")

# ── Message Parts Explorer ──────────────────────────────────────────────
st.header("3 · Message Parts Explorer")
st.markdown("Messages in A2A can contain different types of parts:")

part_tab1, part_tab2, part_tab3 = st.tabs(["TextPart", "FilePart", "DataPart"])

with part_tab1:
    st.markdown("**TextPart** — Plain text or markdown content")
    demo_text = TextPart(text="Hello! I can help you with research tasks.")
    st.code(json.dumps(demo_text.to_dict(), indent=2), language="json")

    st.markdown("**Try it:**")
    custom_text = st.text_area("Enter text:", value="Your custom message here", key="custom_text_part")
    if custom_text:
        part = TextPart(text=custom_text)
        st.json(part.to_dict())

with part_tab2:
    st.markdown("**FilePart** — Binary file content with MIME type")
    demo_file = FilePart(name="report.pdf", mime_type="application/pdf", data="<base64-encoded-data>")
    st.code(json.dumps(demo_file.to_dict(), indent=2), language="json")

    st.info("In production, the `bytes` field contains base64-encoded file data. This allows agents to exchange documents, images, and other files.")

with part_tab3:
    st.markdown("**DataPart** — Structured JSON data")
    demo_data = DataPart(data={"results": [{"title": "AI Agents", "score": 0.95}], "total": 1})
    st.code(json.dumps(demo_data.to_dict(), indent=2), language="json")

    st.markdown("**Try it:**")
    custom_json = st.text_area("Enter JSON data:", value='{"key": "value", "count": 42}', key="custom_data_part")
    try:
        parsed = json.loads(custom_json)
        part = DataPart(data=parsed)
        st.json(part.to_dict())
    except json.JSONDecodeError:
        st.error("Invalid JSON — please fix the syntax.")

# ── Artifacts ───────────────────────────────────────────────────────────
st.header("4 · Understanding Artifacts")

st.markdown("""
**Artifacts** are distinct from messages. While messages are conversational,
artifacts represent **deliverables** — the actual outputs of the agent's work.

| Aspect | Messages | Artifacts |
|--------|----------|-----------|
| **Purpose** | Communication | Deliverables |
| **Direction** | Bidirectional | Agent → Client |
| **Content** | Conversational | Final outputs |
| **Versioning** | Sequential | Indexed |
| **Examples** | "Working on it..." | Generated report, code, data |
""")

st.markdown("**Example artifact flow:**")
st.code("""
User → "Write a report on AI trends"

Message 1 (agent): "I'll research and write that report for you."
Message 2 (agent): "I've completed the report. Here are the highlights..."

Artifact 1: {
  "name": "ai_trends_report",
  "description": "Comprehensive report on AI trends 2024-2025",
  "parts": [
    {"type": "text", "text": "# AI Trends Report\\n\\n## Executive Summary..."}
  ]
}
""", language="text")

# ── Task API Reference ──────────────────────────────────────────────────
st.header("5 · Task API Reference")

st.markdown("""
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks/send` | POST | Send a new task or continue an existing one |
| `/tasks/get` | GET | Get the current state of a task |
| `/tasks/cancel` | POST | Cancel a running task |
| `/tasks/sendSubscribe` | POST | Send task and subscribe to SSE updates |

**Send Task Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "tasks/send",
  "params": {
    "id": "task-123",
    "sessionId": "session-456",
    "message": {
      "role": "user",
      "parts": [{"type": "text", "text": "Research AI agents"}]
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "result": {
    "id": "task-123",
    "sessionId": "session-456",
    "status": {"state": "completed"},
    "artifacts": [...]
  }
}
```
""")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/4_🤝_A2A/02_🪪_Agent_Cards.py", label="← Agent Cards", icon="🪪")
with cols[2]:
    st.page_link("pages/4_🤝_A2A/04_🌐_A2A_Collaboration.py", label="A2A Collaboration →", icon="🌐")
