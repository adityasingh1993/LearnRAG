"""Page 25 — A2A Collaboration: Multi-agent orchestration patterns."""

import streamlit as st
import json
import time
from core.a2a_simulator import create_demo_registry, TaskState


st.title("🌐 A2A Multi-Agent Collaboration")
st.markdown("See how multiple agents work together through the A2A protocol.")

# ── Collaboration Patterns ──────────────────────────────────────────────
st.header("1 · Collaboration Patterns")

pat1, pat2, pat3 = st.tabs(["Router Pattern", "Pipeline Pattern", "Parallel Pattern"])

with pat1:
    st.markdown("""
    ### 🔀 Router Pattern
    A **router agent** receives requests and delegates to the best specialist.

    ```
    User Request
         │
         ▼
    ┌──────────┐
    │  Router  │ ── analyzes request, picks best agent
    └────┬─────┘
         │
    ┌────┼────────────────┐
    ▼    ▼                ▼
    ┌────────┐  ┌────────┐  ┌──────────┐
    │  Math  │  │ Writer │  │ Research │
    │ Agent  │  │ Agent  │  │  Agent   │
    └────────┘  └────────┘  └──────────┘
    ```

    **How it works:**
    1. Router fetches Agent Cards to understand capabilities
    2. Analyzes the user's request keywords and intent
    3. Routes to the agent with the best skill match
    4. Returns the specialist's response to the user
    """)

with pat2:
    st.markdown("""
    ### ⛓️ Pipeline Pattern
    Tasks flow through a **sequence** of agents, each adding value.

    ```
    User Request
         │
         ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Research │───►│  Writer  │───►│ Reviewer │
    │  Agent   │    │  Agent   │    │  Agent   │
    └──────────┘    └──────────┘    └──────────┘
    (gather info)   (draft text)    (refine)
         │               │               │
         ▼               ▼               ▼
    artifact:        artifact:      artifact:
    raw findings     draft report   final report
    ```

    **How it works:**
    1. Each agent receives the previous agent's output as input
    2. Artifacts accumulate through the pipeline
    3. The final agent's output is returned to the user
    """)

with pat3:
    st.markdown("""
    ### ⚡ Parallel Pattern
    Multiple agents work on **subtasks simultaneously**.

    ```
    User Request: "Compare Python and Rust"
         │
         ▼
    ┌──────────────┐
    │ Orchestrator │
    └──┬─────────┬─┘
       │         │
       ▼         ▼
    ┌────────┐ ┌────────┐
    │Research│ │Research│
    │Python  │ │ Rust   │
    └───┬────┘ └───┬────┘
        │          │
        ▼          ▼
    ┌──────────────┐
    │   Combine    │ ── merges results into comparison
    └──────────────┘
    ```

    **How it works:**
    1. Orchestrator splits work into independent subtasks
    2. Sends tasks to multiple agents concurrently
    3. Collects all results and merges into final output
    """)

# ── Live Router Demo ────────────────────────────────────────────────────
st.header("2 · Live Router Demo")
st.markdown("""
Type a message and watch the router discover the best agent, send a task,
and return the result — all through the A2A protocol.
""")

registry = create_demo_registry()

all_agents = registry.discover()
st.markdown("**Available agents:**")
for card in all_agents:
    skills_str = ", ".join(s.name for s in card.skills)
    st.markdown(f"- **{card.name}** — {card.description} (Skills: {skills_str})")

user_msg = st.text_input(
    "Your message:",
    placeholder="e.g., 'Add 10 and 25' or 'Write an email' or 'Research quantum computing'",
    key="collab_msg",
)

if st.button("▶️ Send to Router", key="send_router") and user_msg:
    with st.status("Router processing...", expanded=True) as status:

        st.write("**Step 1: Discover agents**")
        agents = registry.discover()
        st.caption(f"Found {len(agents)} registered agents")
        time.sleep(0.3)

        st.write("**Step 2: Analyze & route**")
        best_agent, reason = registry.route_task(user_msg)
        if best_agent:
            st.caption(f"Routing to **{best_agent.card.name}** — {reason}")
        time.sleep(0.3)

        st.write("**Step 3: Send task**")
        if best_agent:
            task = best_agent.send_task(user_msg)
            st.caption(f"Task ID: `{task.id}` | State: `{task.state.value}`")
            time.sleep(0.3)

            if task.state == TaskState.COMPLETED:
                st.write("**Step 4: Task completed** ✅")
                status.update(label="Task routed and completed!", state="complete")
            else:
                st.write(f"**Step 4: Task state — {task.state.value}**")
                status.update(label=f"Task {task.state.value}", state="error")

    if best_agent and task:
        st.subheader("Result")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Routed To", best_agent.card.name)
        mc2.metric("Task State", task.state.value)
        mc3.metric("Artifacts", len(task.artifacts))

        for msg in task.messages:
            with st.chat_message("user" if msg.role == "user" else "assistant"):
                st.markdown(msg.text_content())

        if task.artifacts:
            st.markdown("**Artifacts produced:**")
            for artifact in task.artifacts:
                with st.expander(f"📦 {artifact.name} — {artifact.description}"):
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            st.code(part.text, language="text")

        with st.expander("🔍 Protocol Details"):
            st.json({
                "routing": {"agent": best_agent.card.name, "reason": reason},
                "task": task.to_dict(),
                "agent_card": best_agent.card.to_dict(),
            })

# ── Pipeline Demo ───────────────────────────────────────────────────────
st.header("3 · Pipeline Demo")
st.markdown("Watch a task flow through multiple agents in sequence.")

if st.button("▶️ Run Pipeline: Research → Write → Review", key="run_pipeline"):
    topic = "the future of AI agents"

    with st.status("Pipeline running...", expanded=True) as status:

        st.write("**Stage 1: Research Agent** — Gathering information")
        research_agent = registry.get_agent("Research Agent")
        research_task = research_agent.send_task(f"Research {topic}")
        research_output = ""
        if research_task.artifacts:
            for p in research_task.artifacts[0].parts:
                if hasattr(p, "text"):
                    research_output = p.text
        st.caption(f"Research complete — {len(research_output)} chars")
        time.sleep(0.5)

        st.write("**Stage 2: Writer Agent** — Drafting content")
        writer_agent = registry.get_agent("Writer Agent")
        writer_task = writer_agent.send_task(f"Summarize this research: {research_output[:200]}")
        writer_output = ""
        if writer_task.artifacts:
            for p in writer_task.artifacts[0].parts:
                if hasattr(p, "text"):
                    writer_output = p.text
        st.caption(f"Draft complete — {len(writer_output)} chars")
        time.sleep(0.5)

        st.write("**Stage 3: Review** — Final output")
        status.update(label="Pipeline complete!", state="complete")

    st.subheader("Pipeline Results")

    pipeline_stages = [
        ("🔬 Research", research_task),
        ("✍️ Writing", writer_task),
    ]

    for stage_name, task in pipeline_stages:
        with st.expander(f"{stage_name} — {task.state.value}", expanded=True):
            for msg in task.messages:
                if msg.role == "agent":
                    st.markdown(msg.text_content())
            if task.artifacts:
                st.markdown("**Artifact:**")
                for part in task.artifacts[0].parts:
                    if hasattr(part, "text"):
                        st.code(part.text, language="text")

# ── Collaboration Comparison ────────────────────────────────────────────
st.header("4 · Pattern Comparison")

st.markdown("""
| Pattern | Best For | Agents | Latency | Complexity |
|---------|----------|--------|---------|------------|
| **Router** | Task delegation | 1 + N specialists | Low (single hop) | Low |
| **Pipeline** | Sequential processing | N in sequence | Medium (sum of stages) | Medium |
| **Parallel** | Independent subtasks | N concurrent | Low (max of agents) | High |
| **Debate** | Critical decisions | 2-3 debaters + judge | High (multiple rounds) | High |
| **Hierarchical** | Complex workflows | Manager + workers | Variable | Very High |
""")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/24_📋_A2A_Tasks.py", label="← A2A Tasks", icon="📋")
with cols[2]:
    st.page_link("pages/26_🎮_A2A_Playground.py", label="A2A Playground →", icon="🎮")
