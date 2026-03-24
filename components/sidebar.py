"""
Shared sidebar component for provider configuration across all pages.
"""

import streamlit as st
from core.config import get_all_providers


def render_provider_config():
    """Render the LLM & Embedding provider configuration in the sidebar."""
    with st.sidebar:
        st.markdown("### Settings")

        providers = get_all_providers()

        # Use session-state keys to reflect true availability
        for pkey in ("openai", "openrouter"):
            if st.session_state.get(f"{pkey}_key"):
                providers[pkey].is_available = True

        status_items = []
        for key, cfg in providers.items():
            icon = " ✅" if cfg.is_available else " ❌"
            status_items.append(f"- **{cfg.name}**{icon}")
        st.markdown("**Provider Status**\n" + "\n".join(status_items))

        st.divider()

        llm_provider_name = st.selectbox(
            "LLM Provider",
            options=list(providers.keys()),
            format_func=lambda x: providers[x].name,
            key="llm_provider_select",
        )

        llm_config = providers[llm_provider_name]
        if llm_provider_name in ("openai", "openrouter"):
            saved_key = st.session_state.get(f"{llm_provider_name}_key", "") or llm_config.api_key
            if f"{llm_provider_name}_api_key" not in st.session_state:
                st.session_state[f"{llm_provider_name}_api_key"] = saved_key

            api_key = st.text_input(
                f"{llm_config.name} API Key",
                type="password",
                key=f"{llm_provider_name}_api_key",
            )
            st.session_state[f"{llm_provider_name}_key"] = api_key
        else:
            api_key = st.session_state.get("openai_key", "") or st.session_state.get("openrouter_key", "")
            saved_url = st.session_state.get("ollama_url", "") or llm_config.base_url
            if "ollama_base_url" not in st.session_state:
                st.session_state["ollama_base_url"] = saved_url

            base_url = st.text_input(
                "Ollama URL",
                key="ollama_base_url",
            )
            st.session_state["ollama_url"] = base_url

        llm_model = st.selectbox(
            "LLM Model",
            options=llm_config.models,
            key="llm_model_select",
        )

        st.divider()

        embed_options = ["tfidf"]
        embed_labels = {"tfidf": "TF-IDF (Local, Free)"}
        if providers["openai"].is_available or st.session_state.get("openai_key"):
            embed_options.insert(0, "openai")
            embed_labels["openai"] = "OpenAI Embeddings"
        if providers["openrouter"].is_available or st.session_state.get("openrouter_key"):
            embed_options.insert(0, "openrouter")
            embed_labels["openrouter"] = "OpenRouter Embeddings"
        if providers["ollama"].is_available:
            embed_options.insert(0, "ollama")
            embed_labels["ollama"] = "Ollama Embeddings"

        embed_provider_name = st.selectbox(
            "Embedding Provider",
            options=embed_options,
            format_func=lambda x: embed_labels.get(x, x),
            key="embed_provider_select",
        )

        if embed_provider_name == "openai":
            embed_model = st.selectbox(
                "Embedding Model",
                options=providers["openai"].embedding_models,
                key="embed_model_select",
            )
        elif embed_provider_name == "openrouter":
            from core.embeddings import OpenRouterEmbeddings
            embed_model = st.selectbox(
                "Embedding Model",
                options=OpenRouterEmbeddings.MODELS,
                key="embed_model_select_openrouter",
            )
        elif embed_provider_name == "ollama":
            embed_model = st.selectbox(
                "Embedding Model",
                options=providers["ollama"].embedding_models,
                key="embed_model_select_ollama",
            )
        else:
            embed_model = None

        st.divider()

        vs_type = st.selectbox(
            "Vector Store",
            options=["numpy", "chroma"],
            format_func=lambda x: {"numpy": "NumPy (In-Memory)", "chroma": "ChromaDB"}.get(x, x),
            key="vs_type_select",
        )

        st.session_state["provider_config"] = {
            "llm_provider": llm_provider_name,
            "llm_model": llm_model,
            "llm_api_key": api_key,
            "llm_base_url": llm_config.base_url,
            "embed_provider": embed_provider_name,
            "embed_model": embed_model,
            "vector_store_type": vs_type,
        }

        return st.session_state["provider_config"]


def _resolve_api_key(provider: str) -> str:
    """Get the API key for a provider from session state or config."""
    cfg = st.session_state.get("provider_config", {})
    return (
        st.session_state.get(f"{provider}_key", "")
        or cfg.get("llm_api_key", "")
    )


def get_llm_provider():
    """Create an LLM provider from current session config."""
    from core.llm_providers import create_llm
    cfg = st.session_state.get("provider_config", {})
    provider = cfg.get("llm_provider", "openai")
    kwargs = {"model": cfg.get("llm_model", "gpt-4o-mini")}
    if provider == "openai":
        kwargs["api_key"] = _resolve_api_key("openai")
        kwargs["base_url"] = cfg.get("llm_base_url", "https://api.openai.com/v1")
    elif provider == "openrouter":
        kwargs["api_key"] = _resolve_api_key("openrouter")
    elif provider == "ollama":
        kwargs["base_url"] = st.session_state.get("ollama_url", "http://localhost:11434")
    return create_llm(provider, **kwargs)


def get_embedding_provider():
    """Create an embedding provider from current session config."""
    from core.embeddings import create_embeddings
    cfg = st.session_state.get("provider_config", {})
    provider = cfg.get("embed_provider", "tfidf")
    kwargs = {}
    if provider == "openai":
        kwargs["api_key"] = _resolve_api_key("openai")
        kwargs["model"] = cfg.get("embed_model", "text-embedding-3-small")
    elif provider == "openrouter":
        kwargs["api_key"] = _resolve_api_key("openrouter")
        kwargs["model"] = cfg.get("embed_model", "openai/text-embedding-3-small")
    elif provider == "ollama":
        kwargs["model"] = cfg.get("embed_model", "nomic-embed-text")
        kwargs["base_url"] = st.session_state.get("ollama_url", "http://localhost:11434")
    return create_embeddings(provider, **kwargs)


def get_vector_store():
    """Create a vector store from current session config."""
    from core.vector_store import create_vector_store
    cfg = st.session_state.get("provider_config", {})
    return create_vector_store(cfg.get("vector_store_type", "numpy"))
