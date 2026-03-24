"""Page 23 — Agent Cards & Discovery: How agents describe and find each other."""

import streamlit as st
import json
from core.a2a_simulator import (
    AgentCard, AgentSkill, create_demo_agents, create_demo_registry,
)

st.set_page_config(page_title="Agent Cards & Discovery", page_icon="🪪", layout="wide")

st.title("🪪 Agent Cards & Discovery")
st.markdown("""
Before agents can collaborate, they need to **find each other** and
**understand what each one can do**. Agent Cards make this possible.
""")

# ── Agent Card Anatomy ──────────────────────────────────────────────────
st.header("1 · Anatomy of an Agent Card")

st.markdown("""
An Agent Card is a JSON document served at `/.well-known/agent.json`:
""")

example_card = {
    "name": "Research Agent",
    "description": "Conducts research and gathers information on topics",
    "url": "http://localhost:8003",
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": True,
    },
    "skills": [
        {
            "id": "search",
            "name": "Topic Research",
            "description": "Research any topic in depth",
            "tags": ["research", "search", "investigate"],
            "examples": ["Research AI agents", "Find info about quantum computing"],
        },
        {
            "id": "compare",
            "name": "Comparison",
            "description": "Compare two or more concepts",
            "tags": ["compare", "versus", "difference"],
            "examples": ["Compare Python vs Java", "MCP vs A2A differences"],
        },
    ],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
}

st.code(json.dumps(example_card, indent=2), language="json")

st.markdown("""
| Section | Purpose |
|---------|---------|
| **name / description** | Human-readable identity |
| **url** | Endpoint where the agent accepts tasks |
| **capabilities** | What protocol features the agent supports |
| **skills** | Detailed list of what the agent can do (with tags for matching) |
| **input/output modes** | What media types the agent handles |
""")

# ── Browse Demo Agents ──────────────────────────────────────────────────
st.header("2 · Browse Demo Agent Cards")

agents = create_demo_agents()

for name, agent in agents.items():
    card = agent.card
    with st.expander(f"**{card.name}** — {card.description}", expanded=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"**URL:** `{card.url}`")
            st.markdown(f"**Version:** {card.version}")
            st.markdown(f"**Input modes:** {', '.join(card.input_modes)}")
            st.markdown(f"**Output modes:** {', '.join(card.output_modes)}")
            st.markdown(f"**Streaming:** {'✅' if card.supports_streaming else '❌'}")
            st.markdown(f"**Push Notifications:** {'✅' if card.supports_push_notifications else '❌'}")

        with c2:
            st.markdown("**Skills:**")
            for skill in card.skills:
                st.markdown(f"- **{skill.name}**")
                st.caption(f"  Tags: {', '.join(skill.tags)}")

        st.markdown("**Full Agent Card JSON:**")
        st.code(card.to_json(), language="json")

# ── Discovery Simulator ─────────────────────────────────────────────────
st.header("3 · Agent Discovery Simulator")

st.markdown("""
In production, a client agent discovers other agents by:
1. **Registry lookup** — Query a central registry
2. **Direct URL** — Fetch `/.well-known/agent.json` from known hosts
3. **Keyword matching** — Search by skills, tags, or description

Try searching the registry below:
""")

registry = create_demo_registry()

search_query = st.text_input(
    "Search for agents by skill, tag, or keyword:",
    placeholder="e.g., math, email, research, compare",
    key="discovery_search",
)

if search_query:
    results = registry.discover(search_query)
    if results:
        st.success(f"Found **{len(results)}** matching agent(s):")
        for card in results:
            st.markdown(f"- **{card.name}** — {card.description}")
            matching_skills = [s for s in card.skills
                             if search_query.lower() in s.name.lower()
                             or any(search_query.lower() in t for t in s.tags)]
            if matching_skills:
                for s in matching_skills:
                    st.caption(f"  ✅ Matched skill: **{s.name}** (tags: {', '.join(s.tags)})")
    else:
        st.warning("No agents matched your search. Try: math, email, research, compare")
else:
    st.info("All registered agents:")
    for card in registry.discover():
        st.markdown(f"- **{card.name}** — {card.description} ({len(card.skills)} skills)")

# ── Build Your Own Agent Card ───────────────────────────────────────────
st.header("4 · Build Your Own Agent Card")

st.markdown("Define a custom agent and see its Agent Card JSON:")

bc1, bc2 = st.columns(2)
custom_name = bc1.text_input("Agent Name", value="My Custom Agent", key="custom_name")
custom_desc = bc2.text_input("Description", value="A helpful custom agent", key="custom_desc")
custom_url = st.text_input("URL", value="http://localhost:9000", key="custom_url")

st.markdown("**Skills** (add up to 3):")
custom_skills = []
for i in range(3):
    sc1, sc2, sc3 = st.columns(3)
    s_name = sc1.text_input(f"Skill {i+1} name", key=f"cs_name_{i}")
    s_desc = sc2.text_input(f"Skill {i+1} description", key=f"cs_desc_{i}")
    s_tags = sc3.text_input(f"Skill {i+1} tags (comma-separated)", key=f"cs_tags_{i}")
    if s_name:
        custom_skills.append(AgentSkill(
            id=s_name.lower().replace(" ", "_"),
            name=s_name,
            description=s_desc,
            tags=[t.strip() for t in s_tags.split(",") if t.strip()],
        ))

c_stream = st.checkbox("Supports Streaming", key="c_stream")
c_push = st.checkbox("Supports Push Notifications", key="c_push")

if custom_name:
    custom_card = AgentCard(
        name=custom_name,
        description=custom_desc,
        url=custom_url,
        skills=custom_skills,
        supports_streaming=c_stream,
        supports_push_notifications=c_push,
    )

    st.markdown("**Generated Agent Card:**")
    st.code(custom_card.to_json(), language="json")

    st.markdown("**Discovery URL:**")
    st.code(f"GET {custom_url}/.well-known/agent.json", language="text")

# ── How Discovery Works ─────────────────────────────────────────────────
st.header("5 · How Agent Discovery Works")

st.markdown("""
```
Client Agent                          Registry / Network
     │                                       │
     │── 1. Search for "math" agents ──────►│
     │                                       │
     │◄── 2. Return matching Agent Cards ───│
     │                                       │
     │── 3. Fetch /.well-known/agent.json ─►│ (Agent's URL)
     │                                       │
     │◄── 4. Full Agent Card ──────────────│
     │                                       │
     │── 5. Send task to chosen agent ─────►│
     │                                       │
```

**Discovery patterns:**
1. **Static configuration** — Pre-configured list of known agents
2. **Registry-based** — Central registry maintains agent list
3. **Peer discovery** — Agents announce themselves on the network
4. **Hybrid** — Combine static config with dynamic discovery
""")

# ── Navigation ──────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/22_🤝_A2A_Basics.py", label="← A2A Basics", icon="🤝")
with cols[2]:
    st.page_link("pages/24_📋_A2A_Tasks.py", label="A2A Tasks →", icon="📋")
