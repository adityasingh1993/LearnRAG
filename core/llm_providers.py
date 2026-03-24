"""
LLM provider abstraction layer.
Supports OpenAI, OpenRouter, and Ollama with a unified interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import requests


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict | None = None


class LLMProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: str | None = None, **kwargs):
        """Yields text chunks."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def _build_messages(self, prompt: str, system_prompt: str | None) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> LLMResponse:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )

    def generate_stream(self, prompt: str, system_prompt: str | None = None, **kwargs):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        stream = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.1-8b-instruct:free"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    def name(self) -> str:
        return f"OpenRouter ({self.model.split('/')[-1]})"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "RAG Learning App",
        }

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1024),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=self.model,
            usage=data.get("usage"),
        )

    def generate_stream(self, prompt: str, system_prompt: str | None = None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1024),
                "stream": True,
            },
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def name(self) -> str:
        return f"Ollama ({self.model})"

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = requests.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(text=data["response"], model=self.model)

    def generate_stream(self, prompt: str, system_prompt: str | None = None, **kwargs):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if not data.get("done", False):
                    yield data.get("response", "")


def create_llm(provider: str, **kwargs) -> LLMProvider:
    """Factory function to create an LLM provider."""
    providers = {
        "openai": OpenAIProvider,
        "openrouter": OpenRouterProvider,
        "ollama": OllamaProvider,
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")
    return providers[provider](**kwargs)
