# 🧠 RAG & Agents Learning Lab

An interactive educational app for learning **Retrieval-Augmented Generation (RAG)** and **AI Agents** from basics to advanced — built entirely in Python with Streamlit.

No boring slides. Learn by building.

## Features

### RAG Track

| Module | What You'll Learn |
|--------|------------------|
| 📖 **Learn Basics** | What RAG is, why it matters, interactive quiz |
| 🧩 **Embeddings** | Text → vectors, similarity, 2D/3D visualization |
| 📦 **Vector Stores** | Store & search embeddings, see it work live |
| 🔍 **Retrieval** | Compare similarity vs MMR vs hybrid search |
| 🤖 **Generation** | Prompt templates (Standard, CoT, Analysis), streaming |
| 🔬 **Full Pipeline** | End-to-end walkthrough with all strategies & guardrails |
| 📊 **Evaluation** | Enterprise-grade retrieval & generation metrics, LLM-as-judge, A/B comparison |
| 🎮 **RAG Playground** | Visual pipeline builder — configure, chat, experiment |
| ❓ **RAG Help** | How every RAG Playground feature works, with learning resources |

### Agents Track

| Module | What You'll Learn |
|--------|------------------|
| 🤖 **Agent Basics** | What agents are, how they differ from RAG, the agent loop |
| 🔧 **Tools** | Function calling, 7 built-in tools, custom tool builder |
| 🔄 **Agent Patterns** | ReAct, Plan-Execute, Reflection, Tool Choice — run them live |
| 🌐 **Multi-Agent** | Router, orchestrator, and debate patterns for complex tasks |
| 🎮 **Agent Playground** | Build your own agent with tools, RAG integration, and chat |
| ❓ **Agent Help** | Reference for all agent features, patterns, and resources |

### Chunking Strategies (8)

| Strategy | Description |
|----------|-------------|
| **Character** | Fixed-size character splits with overlap |
| **Sentence** | Groups of N sentences with overlap |
| **Paragraph** | Split on double-newline paragraph boundaries |
| **Recursive** | Smart recursive split on natural boundaries |
| **Token** | Split by GPT token count (tiktoken) |
| **Markdown** | Header-aware split for structured documents |
| **Sliding Window** | Overlapping windows with configurable stride |
| **Semantic** | Groups sentences by topical similarity (TF-IDF) |

### Retrieval Strategies (5)

| Strategy | Description |
|----------|-------------|
| **Cosine Similarity** | Standard nearest-neighbour search |
| **MMR** | Maximum Marginal Relevance — relevance + diversity |
| **Hybrid** | BM25 keyword + semantic search fused with RRF |
| **Multi-Query** | LLM rewrites query 3 ways, results merged via RRF |
| **HyDE** | Hypothetical Document Embeddings — search with a generated answer |

### Reasoning Modes (6)

| Mode | Description |
|------|-------------|
| **Standard** | Direct answer from retrieved context |
| **Chain-of-Thought** | Step-by-step reasoning before the final answer |
| **Analysis** | Evaluate each chunk's relevance, then synthesise |
| **CoT + Analysis** | Both combined for maximum depth |
| **Step-Back** | Answer a broader question first for context |
| **Self-Reflect** | Generate, critique, and refine the answer |

### Guardrails (Optional, Enterprise-grade)

| Guardrail | Type | Description |
|-----------|------|-------------|
| **Input Length** | Rule | Reject too-short / too-long queries |
| **PII Detection** | Regex | Block queries with emails, SSNs, phones, credit cards |
| **Prompt Injection** | Regex | Catch jailbreak & override attempts |
| **Topic Filter** | LLM | Reject off-topic queries |
| **Toxicity (Input)** | LLM + Keywords | Block harmful queries |
| **Hallucination Check** | LLM | Verify answer is grounded in context |
| **Relevance Check** | LLM | Ensure answer addresses the question |
| **PII in Output** | Regex | Flag personal data in generated answer |
| **Toxicity (Output)** | LLM | Block harmful generated content |

Guardrail modes: **off** (fastest), **warn** (run + display), **block** (reject on failure).

### Conversation Context Management (6 Strategies)

| Strategy | Description |
|----------|-------------|
| **None** | Stateless — every query is independent |
| **Full History** | Pass the entire conversation each time |
| **Sliding Window** | Keep the last N turns |
| **Summary Buffer** | LLM summarises older turns, keeps recent ones verbatim |
| **Token Budget** | Keep as many recent turns as fit within a token limit |
| **Relevant History** | Embed past turns, retrieve only the most relevant ones |

### Token Usage & Cost Tracking

Every pipeline step counts tokens (using tiktoken when available). The UI shows:
- **Per-turn breakdown**: prompt, completion, and embedding tokens for each step
- **Session dashboard**: cumulative tokens across all chat turns with a stacked bar chart
- **Cost estimate**: based on model pricing (GPT-4o-mini, GPT-4o, Claude, etc.)

### LLM Reranking

Enable LLM-based reranking to re-score retrieved chunks with the language model before generation.

### Document Formats

Upload and parse: **PDF**, **DOCX** (Word), **VSDX** (Visio), **TXT**

### Agent Patterns (4)

| Pattern | Description |
|---------|-------------|
| **ReAct** | Reason + Act loop — think, pick a tool, observe, repeat |
| **Plan-and-Execute** | Create a step-by-step plan, then execute each step |
| **Reflection** | Generate answer, critique it, then refine |
| **Tool Choice** | Pick the single best tool and use it once |

### Built-in Agent Tools (7)

| Tool | Category | Description |
|------|----------|-------------|
| **calculator** | Math | Evaluate math expressions (sqrt, sin, log, etc.) |
| **datetime** | Utility | Get current date and time |
| **text_stats** | Text | Analyse text (word count, sentence count, etc.) |
| **web_search** | Search | Search the web for information |
| **json_parse** | Utility | Parse and pretty-print JSON |
| **unit_convert** | Math | Convert between units (km/miles, kg/lbs, etc.) |
| **string_transform** | Text | Transform strings (uppercase, reverse, etc.) |

### Multi-Agent Patterns

| Pattern | Description |
|---------|-------------|
| **Router** | Classifies queries and routes to specialist agents |
| **Orchestrator** | Decomposes complex tasks and delegates to workers |
| **Debate** | Multiple agents argue perspectives, then synthesise |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set up API keys
cp .env.example .env
# Edit .env with your keys

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## LLM Providers

Configure in the sidebar. At least one provider is needed for interactive features:

| Provider | Setup | Cost |
|----------|-------|------|
| **OpenAI** | Set `OPENAI_API_KEY` in `.env` | Paid |
| **OpenRouter** | Set `OPENROUTER_API_KEY` in `.env` | Free tier available |
| **Ollama** | Install from [ollama.com](https://ollama.com), run `ollama pull llama3.2` | Free (local) |

**No API key?** Educational content and local TF-IDF embeddings work without any keys.

## Project Structure

```
RAG/
├── app.py                      # Home page
├── requirements.txt            # Dependencies
├── .env.example                # Environment variables template
├── .streamlit/config.toml      # Streamlit theme
├── core/
│   ├── config.py               # Provider configuration
│   ├── llm_providers.py        # OpenAI / OpenRouter / Ollama
│   ├── embeddings.py           # Embedding providers (incl. local TF-IDF)
│   ├── chunking.py             # 8 text chunking strategies
│   ├── vector_store.py         # NumPy & ChromaDB vector stores
│   ├── retrieval.py            # Advanced retrieval (hybrid, multi-query, HyDE)
│   ├── guardrails.py           # Input + output guardrails (9 types)
│   ├── context.py              # 6 conversation context management strategies
│   ├── token_tracker.py        # Token counting, session tracking, cost estimation
│   ├── document_loader.py      # PDF / DOCX / VSDX / TXT parser
│   ├── rag_pipeline.py         # Pipeline: reasoning, reranking, guardrails, context, tokens
│   ├── tools.py                # Tool system: registry, 7 built-in tools, custom tool builder
│   └── agent_loop.py           # Agent executor: ReAct, Plan-Execute, Reflection, Tool Choice
├── components/
│   ├── sidebar.py              # Shared provider configuration UI
│   └── viz.py                  # Plotly visualizations
├── pages/
│   ├── 1_📖_Learn_Basics.py
│   ├── 2_🧩_Embeddings.py
│   ├── 3_📦_Vector_Stores.py
│   ├── 4_🔍_Retrieval.py
│   ├── 5_🤖_Generation.py
│   ├── 6_🔬_Full_Pipeline.py
│   ├── 7_📊_Evaluation.py
│   ├── 8_🎮_Playground.py
│   ├── 9_❓_Help.py
│   ├── 10_🤖_Agent_Basics.py
│   ├── 11_🔧_Tools.py
│   ├── 12_🔄_Agent_Patterns.py
│   ├── 13_🌐_Multi_Agent.py
│   ├── 14_🎮_Agent_Playground.py
│   └── 15_❓_Agent_Help.py
└── data/samples/               # Sample documents
```

## Tech Stack

- **UI**: Streamlit (multipage app)
- **LLMs**: OpenAI, OpenRouter, Ollama (configurable)
- **Embeddings**: OpenAI, OpenRouter, Ollama, TF-IDF (local fallback)
- **Vector Store**: NumPy (educational) / ChromaDB (production)
- **Retrieval**: Similarity, MMR, Hybrid, Multi-Query, HyDE
- **Guardrails**: 5 input + 4 output (PII, injection, hallucination, toxicity, relevance)
- **Context**: 6 strategies (none, full, sliding window, summary buffer, token budget, relevant)
- **Agents**: 4 patterns (ReAct, Plan-Execute, Reflection, Tool Choice)
- **Agent Tools**: 7 built-in + custom tool builder
- **Multi-Agent**: Router, Orchestrator, Debate patterns
- **Documents**: PDF, DOCX, VSDX, TXT
- **Visualization**: Plotly
- **Language**: 100% Python
