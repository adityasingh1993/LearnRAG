"""Page 26 — A2A Playground: Full interactive multi-agent environment."""

import streamlit as st
import json
import time
from core.a2a_simulator import (
    AgentCard, AgentSkill, A2AAgent, AgentRegistry, Task, TaskState,
    TextPart, DataPart, Artifact,
    create_demo_registry, create_demo_agents,
)

st.set_page_config(page_title="A2A Playground", page_icon="🎮", layout="wide")

st.title("🎮 A2A Playground")
st.markdown("""
A full-featured environment to experiment with A2A agent collaboration.
Register agents, send tasks, build pipelines, and inspect every protocol message.
""")

# ── Session State ───────────────────────────────────────────────────────
if "a2a_registry" not in st.session_state:
    st.session_state.a2a_registry = create_demo_registry()
    st.session_state.a2a_chat = []
    st.session_state.a2a_tasks = []
    st.session_state.a2a_mode = "router"

registry: AgentRegistry = st.session_state.a2a_registry

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🤝 A2A Configuration")

    st.subheader("Registered Agents")
    all_cards = registry.discover()
    for card in all_cards:
        with st.expander(f"**{card.name}**"):
            st.caption(card.description)
            st.caption(f"Skills: {', '.join(s.name for s in card.skills)}")
            st.caption(f"URL: {card.url}")

    st.divider()

    st.subheader("Register Custom Agent")
    with st.form("register_agent"):
        ca_name = st.text_input("Agent Name", value="Custom Agent")
        ca_desc = st.text_input("Description", value="A custom A2A agent")
        ca_skill_name = st.text_input("Skill Name", value="custom_skill")
        ca_skill_tags = st.text_input("Skill Tags (comma-separated)", value="custom")
        ca_response = st.text_area("Default Response", value="I'm a custom agent!", height=80)

        if st.form_submit_button("Register"):
            def _custom_handler(task: Task, resp=ca_response) -> Task:
                user_text = task.messages[-1].text_content() if task.messages else ""
                task.add_message("agent", resp)
                task.add_artifact("custom_output", "Custom agent output",
                                 f"Processed: {user_text}\nResponse: {resp}")
                return task

            custom_agent = A2AAgent(
                card=AgentCard(
                    name=ca_name,
                    description=ca_desc,
                    url=f"http://localhost:{9000 + len(all_cards)}",
                    skills=[AgentSkill(
                        id=ca_skill_name.lower().replace(" ", "_"),
                        name=ca_skill_name,
                        description=ca_desc,
                        tags=[t.strip() for t in ca_skill_tags.split(",") if t.strip()],
                    )],
                ),
                handler=_custom_handler,
            )
            registry.register(custom_agent)
            st.success(f"Registered **{ca_name}**!")
            st.rerun()

    st.divider()
    st.subheader("Mode")
    st.session_state.a2a_mode = st.radio(
        "Collaboration mode:", ["router", "direct", "pipeline"],
        format_func=lambda x: {"router": "🔀 Auto-Router", "direct": "🎯 Direct Agent", "pipeline": "⛓️ Pipeline"}[x],
        key="mode_radio",
    )

    st.divider()
    if st.button("🔄 Reset All", key="a2a_reset"):
        st.session_state.a2a_registry = create_demo_registry()
        st.session_state.a2a_chat = []
        st.session_state.a2a_tasks = []
        st.rerun()

# ── Main: Mode-specific config ──────────────────────────────────────────
mode = st.session_state.a2a_mode

if mode == "direct":
    agent_names = [card.name for card in registry.discover()]
    selected_agent = st.selectbox("Select agent to talk to:", agent_names, key="direct_agent")
elif mode == "pipeline":
    agent_names = [card.name for card in registry.discover()]
    pipeline_agents = st.multiselect(
        "Select agents in pipeline order:",
        agent_names,
        default=agent_names[:2] if len(agent_names) >= 2 else agent_names,
        key="pipeline_agents",
    )

# ── Chat Interface ──────────────────────────────────────────────────────
st.header("Chat")

chat_container = st.container(height=450)
with chat_container:
    for entry in st.session_state.a2a_chat:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("metadata"):
                with st.expander("🔍 Details"):
                    st.json(entry["metadata"])

user_input = st.chat_input("Send a message to the agent(s)...")

if user_input:
    st.session_state.a2a_chat.append({"role": "user", "content": user_input})

    if mode == "router":
        best_agent, reason = registry.route_task(user_input)
        if best_agent:
            task = best_agent.send_task(user_input)
            st.session_state.a2a_tasks.append(task)

            agent_response = ""
            for msg in task.messages:
                if msg.role == "agent":
                    agent_response += msg.text_content() + "\n"

            artifact_text = ""
            if task.artifacts:
                for a in task.artifacts:
                    for p in a.parts:
                        if hasattr(p, "text"):
                            artifact_text += f"\n\n**Artifact — {a.name}:**\n```\n{p.text}\n```"

            full_response = agent_response.strip()
            if artifact_text:
                full_response += artifact_text

            st.session_state.a2a_chat.append({
                "role": "assistant",
                "content": full_response,
                "metadata": {
                    "mode": "router",
                    "routed_to": best_agent.card.name,
                    "routing_reason": reason,
                    "task_id": task.id,
                    "task_state": task.state.value,
                    "artifacts": len(task.artifacts),
                },
            })
        else:
            st.session_state.a2a_chat.append({
                "role": "assistant",
                "content": "No agent available to handle this request.",
                "metadata": {"mode": "router", "error": "no_agent_found"},
            })

    elif mode == "direct":
        agent = registry.get_agent(selected_agent)
        if agent:
            task = agent.send_task(user_input)
            st.session_state.a2a_tasks.append(task)

            agent_response = ""
            for msg in task.messages:
                if msg.role == "agent":
                    agent_response += msg.text_content() + "\n"

            artifact_text = ""
            if task.artifacts:
                for a in task.artifacts:
                    for p in a.parts:
                        if hasattr(p, "text"):
                            artifact_text += f"\n\n**Artifact — {a.name}:**\n```\n{p.text}\n```"

            st.session_state.a2a_chat.append({
                "role": "assistant",
                "content": (agent_response.strip() + artifact_text),
                "metadata": {
                    "mode": "direct",
                    "agent": selected_agent,
                    "task_id": task.id,
                    "task_state": task.state.value,
                },
            })

    elif mode == "pipeline":
        if not pipeline_agents:
            st.session_state.a2a_chat.append({
                "role": "assistant",
                "content": "Please select agents for the pipeline in the sidebar.",
            })
        else:
            pipeline_results = []
            current_input = user_input

            for agent_name in pipeline_agents:
                agent = registry.get_agent(agent_name)
                if agent:
                    task = agent.send_task(current_input)
                    st.session_state.a2a_tasks.append(task)

                    agent_out = ""
                    for msg in task.messages:
                        if msg.role == "agent":
                            agent_out += msg.text_content() + " "
                    if task.artifacts:
                        for a in task.artifacts:
                            for p in a.parts:
                                if hasattr(p, "text"):
                                    agent_out += p.text + " "

                    pipeline_results.append({
                        "agent": agent_name,
                        "task_id": task.id,
                        "state": task.state.value,
                        "output": agent_out.strip(),
                    })
                    current_input = agent_out.strip()

            response_parts = [f"**Pipeline: {' → '.join(pipeline_agents)}**\n"]
            for i, pr in enumerate(pipeline_results):
                response_parts.append(f"**Stage {i+1} ({pr['agent']}):** {pr['output']}")

            st.session_state.a2a_chat.append({
                "role": "assistant",
                "content": "\n\n".join(response_parts),
                "metadata": {
                    "mode": "pipeline",
                    "stages": pipeline_results,
                },
            })

    st.rerun()

# ── Task Inspector ──────────────────────────────────────────────────────
st.header("Task Inspector")

if not st.session_state.a2a_tasks:
    st.info("Send messages above to generate tasks.")
else:
    task_ids = [f"{t.id} ({t.state.value})" for t in reversed(st.session_state.a2a_tasks)]
    selected_idx = st.selectbox("Select task:", range(len(task_ids)),
                                format_func=lambda i: task_ids[i], key="task_inspector")

    task = list(reversed(st.session_state.a2a_tasks))[selected_idx]

    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Task ID", task.id)
    ic2.metric("Session", task.session_id)
    ic3.metric("State", task.state.value)
    ic4.metric("Artifacts", len(task.artifacts))

    itab1, itab2, itab3, itab4 = st.tabs(["Messages", "Artifacts", "State History", "Full JSON"])

    with itab1:
        for msg in task.messages:
            with st.chat_message("user" if msg.role == "user" else "assistant"):
                st.markdown(msg.text_content())

    with itab2:
        if task.artifacts:
            for artifact in task.artifacts:
                with st.expander(f"📦 {artifact.name}"):
                    st.caption(artifact.description)
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            st.code(part.text, language="text")
                        elif hasattr(part, "data"):
                            st.json(part.data)
        else:
            st.info("No artifacts for this task.")

    with itab3:
        if task.history:
            for entry in task.history:
                ts = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
                st.markdown(f"**[{ts}]** `{entry['from']}` → `{entry['to']}` — {entry.get('detail', '')}")
        else:
            st.info("No state transitions recorded.")

    with itab4:
        st.code(json.dumps(task.to_dict(), indent=2, default=str), language="json")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/4_🤝_A2A/04_🌐_A2A_Collaboration.py", label="← Collaboration", icon="🌐")
with cols[2]:
    st.page_link("pages/4_🤝_A2A/06_❓_A2A_Help.py", label="A2A Help →", icon="❓")
