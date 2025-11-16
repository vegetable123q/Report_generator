"""
Model configuration helpers for LangChain / LangGraph workflows.

The module centralises how OpenAI credentials are loaded and how default model
names are selected for different workflow purposes. This avoids scattering API
key lookups across the codebase and makes it easier to swap providers later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..secrets import OpenAISecrets, Secrets, load_secrets

__all__ = ["ModelPurpose", "ModelRouter"]

ModelPurpose = Literal["general", "deep_research", "creative"]


class LLMProvider(Protocol):
    """Interface implemented by provider-specific factories."""

    name: str

    def create_chat_model(
        self,
        *,
        purpose: ModelPurpose,
        temperature: float,
        timeout: int | None,
        model_override: str | None,
    ) -> BaseChatModel: ...


@dataclass(slots=True)
class OpenAIProvider:
    """LLM provider backed by the OpenAI Chat Completions API."""

    secrets: Secrets
    name: str = "openai"

    def __post_init__(self) -> None:
        if not self.secrets.openai:
            raise RuntimeError("OpenAI credentials are not configured. Populate `.sercrets/secrets.toml` based on the example file.")

    def create_chat_model(
        self,
        *,
        purpose: ModelPurpose,
        temperature: float,
        timeout: int | None,
        model_override: str | None,
    ) -> BaseChatModel:
        model_name = model_override or self._select_model(purpose)
        creds: OpenAISecrets = self.secrets.openai  # type: ignore[assignment]
        return ChatOpenAI(
            api_key=creds.api_key,
            model=model_name,
            temperature=temperature,
            timeout=timeout,
        )

    def _select_model(self, purpose: ModelPurpose) -> str:
        creds: OpenAISecrets = self.secrets.openai  # type: ignore[assignment]
        if purpose == "deep_research" and creds.deep_research_model:
            return creds.deep_research_model
        if purpose == "creative" and creds.chat_model:
            return creds.chat_model
        if creds.chat_model:
            return creds.chat_model
        if creds.model:
            return creds.model
        if creds.deep_research_model:
            return creds.deep_research_model
        return "o4-mini-deep-research"


class ModelRouter:
    """Provider-agnostic chat model router."""

    def __init__(
        self,
        *,
        secrets: Optional[Secrets] = None,
        default_provider: str = "openai",
    ) -> None:
        self._secrets = secrets or load_secrets()
        self._providers: Dict[str, LLMProvider] = {}
        self._default_provider = default_provider
        self.register_provider(OpenAIProvider(self._secrets))
        if default_provider not in self._providers:
            raise ValueError(f"Unknown default LLM provider '{default_provider}'.")

    def register_provider(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def available_providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def create_chat_model(
        self,
        *,
        purpose: ModelPurpose = "general",
        temperature: float = 0.4,
        timeout: int | None = None,
        model_override: str | None = None,
        provider: str | None = None,
    ) -> BaseChatModel:
        provider_name = (provider or self._default_provider).lower()
        if provider_name not in self._providers:
            available = ", ".join(self.available_providers()) or "none"
            raise ValueError(f"Unknown LLM provider '{provider_name}'. Available providers: {available}.")
        factory = self._providers[provider_name]
        return factory.create_chat_model(
            purpose=purpose,
            temperature=temperature,
            timeout=timeout,
            model_override=model_override,
        )
