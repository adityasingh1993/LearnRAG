"""
Centralized configuration management.
Loads settings from environment variables and provides defaults.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ProviderConfig:
    """Configuration for a single LLM/embedding provider."""
    name: str
    api_key: str = ""
    base_url: str = ""
    models: list = field(default_factory=list)
    embedding_models: list = field(default_factory=list)
    is_available: bool = False


def get_openai_config() -> ProviderConfig:
    api_key = os.getenv("OPENAI_API_KEY", "")
    return ProviderConfig(
        name="OpenAI",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        models=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        embedding_models=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        is_available=bool(api_key and api_key != "sk-your-openai-key-here"),
    )


def get_openrouter_config() -> ProviderConfig:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    return ProviderConfig(
        name="OpenRouter",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        models=[
            "meta-llama/llama-3.1-8b-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
        ],
        embedding_models=[
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
            "openai/text-embedding-ada-002",
            "google/text-embedding-004",
        ],
        is_available=bool(api_key and api_key != "sk-or-your-openrouter-key-here"),
    )


def get_ollama_config() -> ProviderConfig:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    available = False
    models = []
    embedding_models = []
    try:
        import requests
        resp = requests.get(f"{base_url}/api/tags", timeout=2)
        if resp.status_code == 200:
            available = True
            for m in resp.json().get("models", []):
                name = m["name"]
                models.append(name)
                if any(k in name.lower() for k in ["embed", "nomic", "mxbai", "all-minilm"]):
                    embedding_models.append(name)
    except Exception:
        pass
    if not models:
        models = ["llama3.2", "mistral", "phi3", "gemma2"]
    if not embedding_models:
        embedding_models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
    return ProviderConfig(
        name="Ollama",
        base_url=base_url,
        models=models,
        embedding_models=embedding_models,
        is_available=available,
    )


def get_all_providers() -> dict[str, ProviderConfig]:
    return {
        "openai": get_openai_config(),
        "openrouter": get_openrouter_config(),
        "ollama": get_ollama_config(),
    }
