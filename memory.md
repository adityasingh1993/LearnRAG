# RAG Learning Lab — Project Memory

> **Last updated:** 2026-03-24
> **Repo:** `git@github.com:adityasingh1993/LearnRAG.git` (branch: `main`)
> **Workspace:** `e:\CursorProjects\RAG`

---

## What This Project Is

An **interactive educational Streamlit app** that teaches **RAG**, **AI Agents**, **MCP (Model Context Protocol)**, and **A2A (Agent-to-Agent Protocol)** from basics to advanced. Instead of slides, users learn by building — every concept has an interactive demo. The app has four tracks: a **RAG track** (9 pages), an **Agents track** (6 pages), an **MCP track** (6 pages), and an **A2A track** (6 pages) — each with its own playground.

**100% Python. No JavaScript. No frontend framework.**

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        app.py (Home)                         │
│              Streamlit multipage entry point                  │
│           Custom CSS, module cards, flow diagram              │
└──────────────┬───────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │    pages/ (27 pages) │
    │  Each page imports:   │
    │  - components/sidebar │
    │  - components/viz     │
    │  - core/* modules     │
    └──────────┬───────────┘
               │
    ┌──────────▼──────────────────────────────────────┐
    │                    core/ (15 modules)             │
    │                                                   │
    │  config.py ──► llm_providers.py                   │
    │                embeddings.py                      │
    │  document_loader.py ──► chunking.py               │
    │                         vector_store.py           │
    │                         retrieval.py              │
    │  guardrails.py                                    │
    │  context.py                                       │
    │  token_tracker.py                                 │
    │                                                   │
    │  rag_pipeline.py  (RAG orchestrator)              │
    │  tools.py         (tool system for agents)        │
    │  agent_loop.py    (agent executor — 4 patterns)   │
    │                                                   │
    │  mcp_simulator.py (MCP Host/Client/Server/Transport)│
    │  a2a_simulator.py (A2A Agent Cards/Tasks/Registry)│
    └───────────────────────────────────────────────────┘
```

---

## Page Flow (User Journey)

| # | Page | File | Purpose |
|---|------|------|---------|
| 1 | Learn Basics | `1_📖_Learn_Basics.py` | What is RAG, why it matters, quiz |
| 2 | Embeddings | `2_🧩_Embeddings.py` | Text → vectors, 2D/3D viz, similarity |
| 3 | Vector Stores | `3_📦_Vector_Stores.py` | Store & search embeddings, upload docs |
| 4 | Retrieval | `4_🔍_Retrieval.py` | Compare similarity vs MMR, interactive |
| 5 | Generation | `5_🤖_Generation.py` | Prompt templates, streaming LLM output |
| 6 | Full Pipeline | `6_🔬_Full_Pipeline.py` | End-to-end RAG with all options exposed |
| 7 | Evaluation | `7_📊_Evaluation.py` | Retrieval metrics, LLM-as-judge, A/B test |
| 8 | Playground | `8_🎮_Playground.py` | Visual builder, chat UI, token dashboard |
| 9 | RAG Help | `9_❓_Help.py` | Reference docs for every RAG Playground feature |
| 10 | Agent Basics | `10_🤖_Agent_Basics.py` | What agents are, agent loop, comparison with RAG |
| 11 | Tools | `11_🔧_Tools.py` | Function calling, 7 built-in tools, custom tool builder |
| 12 | Agent Patterns | `12_🔄_Agent_Patterns.py` | ReAct, Plan-Execute, Reflection, Tool Choice — live demos |
| 13 | Multi-Agent | `13_🌐_Multi_Agent.py` | Router, orchestrator, debate patterns — live router demo |
| 14 | Agent Playground | `14_🎮_Agent_Playground.py` | Build your own agent, tool selection, RAG integration, chat |
| 15 | Agent Help | `15_❓_Agent_Help.py` | Reference for all agent features, patterns, resources |
| 16 | MCP Basics | `16_🔌_MCP_Basics.py` | What MCP is, architecture overview, quiz |
| 17 | MCP Architecture | `17_🏗️_MCP_Architecture.py` | Hosts, clients, servers, transports, handshake simulator |
| 18 | MCP Primitives | `18_🧱_MCP_Primitives.py` | Connect to demo servers, explore resources/tools/prompts |
| 19 | MCP Server Builder | `19_🛠️_MCP_Server_Builder.py` | Build custom MCP server, add primitives, generate code |
| 20 | MCP Playground | `20_🎮_MCP_Playground.py` | Multi-server host, chat simulation, protocol log |
| 21 | MCP Help | `21_❓_MCP_Help.py` | Reference, comparisons, further reading |
| 22 | A2A Basics | `22_🤝_A2A_Basics.py` | What A2A is, MCP vs A2A, quiz |
| 23 | Agent Cards | `23_🪪_Agent_Cards.py` | Agent Card anatomy, discovery simulator, card builder |
| 24 | A2A Tasks | `24_📋_A2A_Tasks.py` | Task lifecycle, messages, parts, artifacts — interactive |
| 25 | A2A Collaboration | `25_🌐_A2A_Collaboration.py` | Router, pipeline, parallel patterns — live demos |
| 26 | A2A Playground | `26_🎮_A2A_Playground.py` | Multi-agent environment, routing, pipelines, task inspector |
| 27 | A2A Help | `27_❓_A2A_Help.py` | A2A vs MCP vs function calling, resources |

---

## Core Modules Detail

### `core/config.py` (94 lines)
- `ProviderConfig` dataclass
- Factory functions: `get_openai_config`, `get_openrouter_config`, `get_ollama_config`, `get_all_providers`
- Reads from env vars (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`)
- Ollama auto-discovers running models via `/api/tags`

### `core/llm_providers.py` (212 lines)
- `LLMResponse` dataclass (text, usage)
- ABC `LLMProvider` → `OpenAIProvider`, `OpenRouterProvider`, `OllamaProvider`
- Each has `generate()` and `generate_stream()` methods
- Factory: `create_llm(provider, **kwargs)`

### `core/embeddings.py` (207 lines)
- ABC `EmbeddingProvider` → `OpenAIEmbeddings`, `OpenRouterEmbeddings`, `OllamaEmbeddings`, `TFIDFEmbeddings`
- `OpenRouterEmbeddings.MODELS` — list of supported embedding models
- `TFIDFEmbeddings` — local fallback, fits on the data (TF-IDF + SVD), no API key needed
- Factory: `create_embeddings(provider, **kwargs)`

### `core/chunking.py` (365 lines)
- `Chunk` dataclass (text, index, metadata, char_count, word_count)
- **8 strategies:** `chunk_by_characters`, `chunk_by_sentences`, `chunk_by_paragraphs`, `chunk_recursive`, `chunk_by_tokens` (tiktoken), `chunk_by_markdown`, `chunk_sliding_window`, `chunk_semantic` (TF-IDF similarity)
- `STRATEGIES` dict maps names to functions
- `STRATEGY_INFO` dict has descriptions
- Entry point: `chunk_text(text, strategy, **kwargs)`

### `core/vector_store.py` (220 lines)
- `SearchResult` dataclass (text, score, index, metadata)
- ABC `VectorStore` → `NumpyVectorStore` (brute-force + MMR), `ChromaVectorStore` (persistent)
- Factory: `create_vector_store(store_type)`

### `core/retrieval.py` (154 lines)
- `RETRIEVAL_STRATEGIES` dict
- Helper: `_bm25_search`, `_rrf_merge`
- Advanced strategies: `hybrid_search` (BM25 + semantic + RRF), `multi_query_search` (LLM rewrites × 3 + RRF), `hyde_search` (LLM generates hypothetical doc, embeds it)
- Each takes `(query, query_embedding, vector_store, embedding_provider, k, llm_provider)`

### `core/guardrails.py` (357 lines)
- `GuardrailResult` dataclass (name, passed, reason)
- **5 input:** `InputLengthGuardrail`, `PIIDetectionGuardrail` (regex), `PromptInjectionGuardrail` (regex), `TopicGuardrail` (LLM), `ToxicityInputGuardrail` (LLM + keywords)
- **4 output:** `HallucinationGuardrail` (LLM), `RelevanceGuardrail` (LLM), `PIIOutputGuardrail` (regex), `ToxicityOutputGuardrail` (LLM)
- `AVAILABLE_INPUT_GUARDRAILS`, `AVAILABLE_OUTPUT_GUARDRAILS` — registry dicts with (label, desc, needs_llm)
- Factories: `create_input_guardrails`, `create_output_guardrails`
- Runners: `run_input_guardrails`, `run_output_guardrails`
- Modes: `off`, `warn`, `block`

### `core/context.py` (373 lines)
- `Turn` dataclass (role, content)
- ABC `ContextManager` → `NoContext`, `FullHistoryContext`, `SlidingWindowContext`, `SummaryBufferContext` (LLM summarises overflow), `TokenBudgetContext` (tiktoken counting), `RelevantHistoryContext` (embedding-based retrieval of past turns)
- `CONTEXT_STRATEGIES` dict
- Factory: `create_context_manager(strategy, **kwargs)`
- All have: `add_turn()`, `get_context_string()`, `clear()`, `turn_count`, `strategy_name`

### `core/token_tracker.py` (155 lines)
- `count_tokens(text, model)` — tiktoken with fallback to word estimate
- `StepTokenUsage` dataclass (step_name, prompt/completion/embedding tokens)
- `TurnTokenUsage` dataclass (list of steps, total_tokens, prompt/completion/embedding_tokens)
- `SessionTokenTracker` — cumulates across turns, has `estimate_cost(model)` method
- `MODEL_PRICING` dict — USD per 1M tokens for GPT-4o-mini, GPT-4o, Claude, etc.

### `core/document_loader.py` (128 lines)
- `SUPPORTED_EXTENSIONS` = `["txt", "pdf", "docx", "vsdx"]`
- `load_text(data, filename)` — dispatches by extension
- Parsers: `_parse_txt`, `_parse_pdf` (pypdf), `_parse_docx` (python-docx), `_parse_vsdx` (XML extraction from OPC/ZIP)

### `core/rag_pipeline.py` (539 lines) — THE ORCHESTRATOR
- Prompt templates: `_PROMPT_STANDARD`, `_PROMPT_COT`, `_PROMPT_ANALYSIS`, `_PROMPT_COT_ANALYSIS`, `_PROMPT_STEP_BACK`, `_PROMPT_SELF_REFLECT`
- All prompts have `{history}`, `{context}`, `{question}` placeholders
- `PROMPT_TEMPLATES` dict, `REASONING_MODES` dict
- `PipelineStep` dataclass (name, duration_ms, details)
- `GuardrailReport` dataclass (results list, all_passed, blocked_reason)
- `RAGResult` dataclass (answer, steps, retrieved_chunks, total_duration_ms, input/output_guardrails, token_usage)
- `RAGPipeline` class:
  - `__init__`: takes embedding_provider, vector_store, llm_provider, chunk_strategy, chunk_kwargs, retrieval_k, retrieval_strategy, reasoning_mode, use_reranking, rerank_top_n, input/output_guardrails, guardrail_mode, context_manager
  - `ingest(text)` → chunks and embeds text, stores in vector store
  - `query(question)` → full pipeline: input guardrails → embed query → retrieve → rerank → build prompt (with history) → generate → output guardrails → track tokens → return RAGResult
  - `_retrieve()` dispatches to similarity/mmr/hybrid/multi_query/hyde
  - `_rerank_with_llm()` — LLM scores each chunk 1-10
  - Token tracking at every step via `core.token_tracker`

### `core/tools.py` (~240 lines) — AGENT TOOL SYSTEM
- `ToolParameter` dataclass (name, type, description, required)
- `Tool` dataclass (name, description, parameters, function, category) — has `run(**kwargs)` and `schema_for_prompt()`
- `ToolRegistry` class — register/get/list/run_tool/format_for_prompt
- **7 built-in tools:** `calculator` (math eval), `datetime` (current time), `text_stats` (word/sentence counts), `web_search` (simulated), `json_parse`, `unit_convert`, `string_transform`
- `BUILTIN_TOOLS` dict
- Factories: `create_tool_registry(tool_names)`, `create_custom_tool(name, description, parameters, code)`

### `core/agent_loop.py` (~300 lines) — AGENT EXECUTION ENGINE
- `AgentStep` dataclass (step_number, thought, action, action_input, observation, is_final, duration_ms)
- `AgentResult` dataclass (answer, steps, total_duration_ms, pattern, tools_used, token_usage)
- `AGENT_PATTERNS` dict — react, plan_execute, reflection, tool_choice
- `AgentExecutor` class:
  - `__init__`: takes llm_provider, tool_registry, pattern, max_steps
  - `run(question, history)` → dispatches to `_run_react`, `_run_plan_execute`, `_run_reflection`, `_run_tool_choice`
  - **ReAct**: Thought → Action → Observation loop until "finish"
  - **Plan-Execute**: LLM creates numbered plan, then executes each step with tools, then synthesises
  - **Reflection**: Tool use → Initial answer → Critique → Refined answer
  - **Tool Choice**: Pick single best tool, use it once, formulate answer
- Prompt templates: `_REACT_SYSTEM`, `_PLAN_SYSTEM`, `_REFLECTION_SYSTEM`, `_TOOL_CHOICE_SYSTEM`

### `core/mcp_simulator.py` (~340 lines) — MCP EDUCATIONAL SIMULATOR
- **Primitives:** `MCPResource` (read-only data with URI), `MCPTool` (executable with JSON Schema + handler), `MCPPrompt` (template with arguments)
- **Transport:** `TransportType` enum (STDIO, SSE), `MCPMessage` (JSON-RPC 2.0), `SimulatedTransport` (message log)
- **Server:** `MCPServer` — registers resources/tools/prompts, handles JSON-RPC requests (`initialize`, `resources/list`, `resources/read`, `tools/list`, `tools/call`, `prompts/list`, `prompts/get`), maintains request log
- **Client:** `MCPClient` — connects to server, lists/reads resources, lists/calls tools, lists/gets prompts
- **Host:** `MCPHost` — manages multiple clients, aggregates capabilities across servers
- **Demo Servers (3):** `create_weather_server` (weather data + alerts), `create_database_server` (SQL queries on fake users/orders), `create_filesystem_server` (file read/list)
- `DEMO_SERVERS` dict with factory functions, descriptions, icons

### `core/a2a_simulator.py` (~340 lines) — A2A EDUCATIONAL SIMULATOR
- **Agent Card:** `AgentSkill` (id, name, tags, examples), `AgentCard` (name, description, url, skills, capabilities, input/output modes) — serializes to JSON
- **Task Lifecycle:** `TaskState` enum (submitted, working, input-required, completed, failed, canceled), `Task` (id, session_id, state, messages, artifacts, history) — has `transition()`, `add_message()`, `add_artifact()`
- **Messages:** `TextPart`, `FilePart`, `DataPart`, `Message` (role, parts, metadata)
- **Artifacts:** `Artifact` (name, description, parts, index)
- **Agent:** `A2AAgent` — wraps AgentCard + handler function, `send_task()`, `get_task()`, `cancel_task()`
- **Registry:** `AgentRegistry` — register/discover/get_agent/route_task (keyword-based routing)
- **Demo Agents (3):** `_math_handler` (arithmetic), `_writer_handler` (email/summary), `_research_handler` (research reports)
- `create_demo_agents()`, `create_demo_registry()`

---

## Components

### `components/sidebar.py` (178 lines)
- `render_provider_config()` — sidebar UI: provider status, LLM provider select, API key input (persisted in session state), model select, embedding provider select, vector store select
- `_resolve_api_key(provider)` — resolves API key from session state or config
- `get_llm_provider()`, `get_embedding_provider()`, `get_vector_store()` — factories using session state config
- **API keys persist in session state** across page navigation (stored in `st.session_state["{provider}_key"]`)

### `components/viz.py` (271 lines)
- `plot_embeddings_2d/3d` — PCA projection scatter plots (Plotly)
- `plot_similarity_heatmap` — cosine similarity matrix
- `plot_retrieval_scores` — horizontal bar chart
- `render_pipeline_flow` — HTML/CSS flow diagram
- `render_step_metrics` — timing table with optional token counts
- `render_token_usage` — per-turn token breakdown
- `render_session_token_summary` — stacked bar chart + cost estimate

---

## Key Design Decisions

1. **Session state for API keys** — keys entered in sidebar persist across all pages via `st.session_state["{provider}_key"]` and `st.session_state["{provider}_api_key"]`. Provider status badges also check session state.

2. **Prompt template sync** — Generation page (page 5) tracks selected template in `st.session_state["_prev_gen_template"]` to update the editable text area when the dropdown changes (Streamlit widget state quirk).

3. **Dynamic chunk kwargs** — The Playground and Full Pipeline pages use a `chunk_kw_map` dict to map the generic `chunk_size`/`chunk_overlap` sliders to strategy-specific parameter names (e.g., `window_size` for sliding window, `max_sentences` for sentence).

4. **Chat UI pattern** — Playground uses `st.container(height=480)` for scrollable chat history, `st.chat_message` for bubbles, `st.chat_input` for input. Controls and dashboards are outside the container to avoid Streamlit layout conflicts.

5. **Evaluation uses LLM-as-judge** — structured prompts returning `SCORE: N\nJUSTIFICATION: ...` format, parsed with string splitting. No external eval framework dependency.

6. **No deployment code** — deployment features (FastAPI, Docker, K8s, agent, SDK, webhook) were removed as they were generating non-functional code. The `core/deploy_gen.py` file was deleted and `fastapi`/`uvicorn` removed from requirements.

7. **TF-IDF as default embedding** — allows the app to work without any API keys for educational purposes.

---

## Dependencies (`requirements.txt`)

```
streamlit>=1.30.0
openai>=1.0.0
chromadb>=0.4.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
python-dotenv>=1.0.0
pyyaml>=6.0
requests>=2.31.0
tiktoken>=0.5.0
streamlit-flow-component>=0.6.0
pypdf>=4.0.0
python-docx>=1.0.0
```

**Note:** `streamlit-flow-component` is listed but the Playground uses custom HTML for the flow diagram instead. Could be removed in a future cleanup.

---

## File Stats

| Area | Files | Lines |
|------|-------|-------|
| `core/` | 15 .py | ~4,500 |
| `pages/` | 27 .py | ~10,000 |
| `components/` | 2 .py | ~450 |
| `app.py` | 1 | ~380 |
| **Total** | **45 .py** | **~15,300** |

---

## Data Flow: Query Lifecycle

```
User types question in Playground chat
        │
        ▼
[Input Guardrails] ── PII / injection / toxicity / topic / length check
        │ (if mode=block and failed → return error)
        ▼
[Embed Query] ── embedding_provider.embed_query(question)
        │
        ▼
[Retrieve] ── dispatched by retrieval_strategy:
        │    similarity → vector_store.search(query_emb, k)
        │    mmr        → vector_store.search_mmr(query_emb, k, lambda)
        │    hybrid     → hybrid_search(BM25 + semantic + RRF)
        │    multi_query→ LLM generates 3 rewrites, each searched, RRF merge
        │    hyde       → LLM generates hypothetical doc, embed it, search
        ▼
[Rerank] ── (optional) LLM scores each chunk 1-10, keep top N
        │
        ▼
[Build Prompt] ── template[reasoning_mode].format(
        │         context=chunks, question=question,
        │         history=context_manager.get_context_string())
        ▼
[Generate] ── llm_provider.generate(prompt)
        │
        ▼
[Output Guardrails] ── hallucination / relevance / PII / toxicity check
        │
        ▼
[Track Tokens] ── count prompt/completion/embedding tokens per step
        │
        ▼
[Update Context] ── context_manager.add_turn(user, question)
                    context_manager.add_turn(assistant, answer)
        │
        ▼
Return RAGResult(answer, steps, chunks, timing, guardrails, token_usage)
```

---

## Environment

- **OS:** Windows 10
- **Shell:** PowerShell (no `&&` support — use `;` to chain commands)
- **Python venv:** `.venv/` in project root
- **Run command:** `python -m streamlit run app.py`
- **Git remote:** `origin` → `git@github.com:adityasingh1993/LearnRAG.git`

---

## Agent Data Flow: Query Lifecycle

```
User types question in Agent Playground chat
        │
        ▼
[AgentExecutor.run(question)]
        │
        ▼
[Pattern dispatch] ── react / plan_execute / reflection / tool_choice
        │
        ▼
[LLM thinks] ── generates Thought + Action + Action Input
        │
        ▼
[Tool execution] ── ToolRegistry.run_tool(action, **kwargs)
        │                returns Observation string
        ▼
[Loop] ── append Observation to conversation, ask LLM again
        │  (ReAct: up to max_steps until Action=finish)
        │  (Plan-Execute: one step per plan item)
        │  (Reflection: initial → critique → refined)
        │  (Tool Choice: single tool call, then answer)
        ▼
[Final Answer] ── returned in AgentResult with all steps
```

---

## MCP Data Flow

```
User interacts in MCP Playground chat
        │
        ▼
[MCPHost] ── manages multiple MCPClients
        │
        ▼
[Keyword matching] ── analyze user query against tool/resource names
        │
        ▼
[MCPClient.call_tool()] ── JSON-RPC: tools/call → MCPServer
   or                                              │
[MCPClient.read_resource()] ── JSON-RPC: resources/read → MCPServer
        │                                          │
        ▼                                          ▼
[MCPServer._dispatch()] ── routes to handler
        │
        ▼
[Tool.execute()] or [Resource.read()] ── returns result
        │
        ▼
Protocol log records all JSON-RPC messages exchanged
```

---

## A2A Data Flow

```
User sends message in A2A Playground
        │
        ▼
[Mode selection] ── router / direct / pipeline
        │
        ├─ router ──► AgentRegistry.route_task(message)
        │              → keyword matching against agent skills
        │              → best_agent.send_task(message)
        │
        ├─ direct ──► selected_agent.send_task(message)
        │
        └─ pipeline ── for each agent in sequence:
                        agent.send_task(current_input)
                        current_input = agent output
        │
        ▼
[A2AAgent.send_task()] ── creates Task, transitions to WORKING
        │                  calls handler function
        │                  handler adds messages + artifacts
        │                  transitions to COMPLETED
        ▼
Return task with messages, artifacts, state history
```

---

## Known Issues / Future Ideas

- `streamlit-flow-component` in requirements but unused (Playground uses custom HTML)
- Ollama embedding auto-discovery could be more robust
- No persistent evaluation results (test suite results are ephemeral)
- No user auth or multi-tenant support (single-user local app)
- Agent tools are simulated (web_search returns canned responses) — could add real API calls
- Could add agent evaluation page (similar to RAG evaluation)
- MCP servers are simulated in-process — could add real stdio/SSE transport
- A2A agents use keyword routing — could add LLM-based intent routing
- MCP and A2A demo servers/agents return canned data — could integrate real APIs
