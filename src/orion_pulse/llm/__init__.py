"""LLM provider abstraction for Orion Pulse."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMProviderName(str, Enum):
    """Supported LLM providers."""

    NVIDIA = "nvidia"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


@dataclass(frozen=True)
class LLMMessage:
    """Message for LLM conversation."""

    role: str  # system, user, assistant
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    usage: dict[str, int] = None  # prompt_tokens, completion_tokens, total_tokens
    metadata: dict[str, Any] = None


class LLMProviderError(Exception):
    """Exception raised when LLM provider encounters an error."""

    def __init__(self, message: str, provider: str, model: str | None = None):
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion from messages."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available/configured."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        ...

    @abstractmethod
    def get_default_model(self) -> str:
        """Get default model name."""
        ...


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._enabled = True

    def get_provider_name(self) -> str:
        return LLMProviderName.MOCK.value

    def is_available(self) -> bool:
        return self._enabled

    def get_default_model(self) -> str:
        return "mock-model"

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        # Return a mock response based on the last user message
        last_user_msg = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )

        # Check for predefined responses
        for key, response in self._responses.items():
            if key.lower() in last_user_msg.lower():
                return LLMResponse(
                    content=response,
                    model=self.get_default_model(),
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                )

        # Default mock response
        return LLMResponse(
            content=f"[MOCK] Analysis for: {last_user_msg[:100]}...",
            model=self.get_default_model(),
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )


class NVIDIAProvider(LLMProvider):
    """NVIDIA API provider (Nemotron, Llama, etc.)."""

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nvidia/nemotron-3-ultra",
        timeout: int = 60,
    ):
        from orion_pulse.config.settings import settings

        self.api_key = api_key or settings.nvidia_api_key
        self.model = model
        self.timeout = timeout
        self._enabled = bool(self.api_key)

    def get_provider_name(self) -> str:
        return LLMProviderName.NVIDIA.value

    def is_available(self) -> bool:
        return self._enabled

    def get_default_model(self) -> str:
        return self.model

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        if not self._enabled:
            raise LLMProviderError(
                "NVIDIA API key not configured",
                provider=self.get_provider_name(),
                model=self.model,
            )

        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            )

        except requests.exceptions.Timeout:
            raise LLMProviderError(
                "NVIDIA API request timed out",
                provider=self.get_provider_name(),
                model=self.model,
            )
        except requests.exceptions.RequestException as e:
            raise LLMProviderError(
                f"NVIDIA API error: {e}",
                provider=self.get_provider_name(),
                model=self.model,
            )


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers: dict[str, LLMProvider] = {}

    @classmethod
    def get_provider(cls, name: str = "nvidia") -> LLMProvider:
        """Get or create a provider instance."""
        if name not in cls._providers:
            if name == "nvidia":
                cls._providers[name] = NVIDIAProvider()
            elif name == "mock":
                cls._providers[name] = MockLLMProvider()
            else:
                raise ValueError(f"Unknown LLM provider: {name}")

        return cls._providers[name]

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider names."""
        providers = []
        for name in ["nvidia", "mock"]:
            provider = cls.get_provider(name)
            if provider.is_available():
                providers.append(name)
        return providers

    @classmethod
    def clear_cache(cls) -> None:
        """Clear provider cache (useful for testing)."""
        cls._providers.clear()
