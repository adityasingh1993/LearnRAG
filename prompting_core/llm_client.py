"""
PromptCraft — LLM Client
Unified interface using OpenRouter API (OpenAI-compatible) with token counting and latency tracking.
"""

import time
import os
import random
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

from prompting_core import config

load_dotenv()


def _get_client() -> OpenAI:
    """Get or create an OpenRouter client."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Set it in your .env file or environment variables."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "PromptCraft",
        },
    )


def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    """
    Generate a response from the LLM via OpenRouter.

    Returns:
        dict with keys: response, tokens_used, latency_ms, model
    """
    client = _get_client()
    model_name = model or config.MODEL_NAME
    temp = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
    p = top_p if top_p is not None else getattr(config, "DEFAULT_TOP_P", 1.0)
    k = top_k if top_k is not None else getattr(config, "DEFAULT_TOP_K", None)
    max_tok = max_tokens or config.MAX_OUTPUT_TOKENS

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Random seed to prevent OpenRouter from caching identical prompts
    seed = random.randint(1, 1_000_000)

    start = time.time()
    
    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temp,
        "top_p": p,
        "max_tokens": max_tok,
        "seed": seed,
    }
    if k is not None:
        kwargs["extra_body"] = {"top_k": k}
        
    response = client.chat.completions.create(**kwargs)
    latency_ms = round((time.time() - start) * 1000)

    text = (response.choices[0].message.content or "") if response.choices else ""
    tokens_used = 0
    if response.usage:
        tokens_used = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)

    return {
        "response": text,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
        "model": model_name,
        "temperature": temp,
        "top_p": p,
        "top_k": k,
    }


def generate_multiple(
    prompt: str,
    system_prompt: Optional[str] = None,
    n: int = 3,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
) -> list[dict]:
    """
    Generate multiple responses for self-consistency.
    Uses slightly varied temperatures to get diverse reasoning paths.
    """
    results = []
    base_temp = temperature if temperature is not None else 0.7
    for i in range(n):
        # Vary temperature slightly for diversity
        t = min(base_temp + (i * 0.15), 1.5)
        result = generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=t,
            top_p=top_p,
            top_k=top_k,
            model=model,
        )
        results.append(result)
    return results


def is_api_key_configured() -> bool:
    """Check if the API key is configured."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    return bool(api_key) and api_key != "your_openrouter_api_key_here"
