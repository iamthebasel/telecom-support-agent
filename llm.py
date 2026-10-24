"""Chat model factory. The provider is picked with the LLM_PROVIDER environment variable."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import Settings


def describe_model(settings: Settings) -> str:
    """e.g. 'ollama:qwen2.5:7b', used in logs and eval reports."""
    models = {"ollama": settings.ollama_model, "anthropic": settings.anthropic_model, "openai": settings.openai_model}
    return f"{settings.llm_provider}:{models.get(settings.llm_provider.lower(), '?')}"


def get_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.anthropic_model, max_tokens=1024)  # reads ANTHROPIC_API_KEY
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, temperature=0)  # reads OPENAI_API_KEY
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (use ollama, anthropic or openai)")
