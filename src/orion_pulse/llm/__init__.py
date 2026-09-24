"""LLM provider abstraction for Orion Pulse."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class LLMProviderName(str, Enum):
    """Supported LLM providers."""

    NVIDIA = "nvidia"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class NVIDIAModel(str, Enum):
    """Available NVIDIA models."""

    NEMOTRON_3_ULTRA = "nvidia/nemotron-3-ultra"
    NEMOTRON_3_ULTRA_253B = "nvidia/nemotron-3-ultra-253b"
    LLAMA_3_1_NEMOTRON_70B = "nvidia/llama-3.1-nemotron-70b-instruct"
    LLAMA_3_1_405B = "meta/llama-3.1-405b-instruct"
    LLAMA_3_1_70B = "meta/llama-3.1-70b-instruct"
    LLAMA_3_1_8B = "meta/llama-3.1-8b-instruct"
    MISTRAL_LARGE = "mistralai/mistral-large"
    MIXTRAL_8X7B = "mistralai/mixtral-8x7b-instruct"
    GEMMA_2_9B = "google/gemma-2-9b-it"
    GEMMA_2_27B = "google/gemma-2-27b-it"


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
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


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
        **kwargs: Any,
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


class NVIDIAProvider(LLMProvider):
    """NVIDIA API provider (Nemotron, Llama, etc.)."""

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    _UNSET = object()

    def __init__(
        self,
        api_key: str | None = _UNSET,
        model: str = "nvidia/nemotron-3-ultra",
        timeout: int = 60,
    ):
        from orion_pulse.config.settings import settings

        # Use provided api_key, or fall back to settings if not explicitly provided
        if api_key is self._UNSET:
            self.api_key = settings.nvidia_api_key
        else:
            self.api_key = api_key
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
        **kwargs: Any,
    ) -> LLMResponse:
        if not self._enabled:
            raise LLMProviderError(
                "NVIDIA API key not configured",
                provider=self.get_provider_name(),
                model=self.model,
            )

        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers: dict[str, str] = {
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

        except requests.exceptions.Timeout as e:
            raise LLMProviderError(
                "NVIDIA API request timed out",
                provider=self.get_provider_name(),
                model=self.model,
            ) from e
        except requests.exceptions.RequestException as e:
            raise LLMProviderError(
                f"NVIDIA API error: {e}",
                provider=self.get_provider_name(),
                model=self.model,
            ) from e


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self._enabled = True

    def get_provider_name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return self._enabled

    def get_default_model(self) -> str:
        return "mock-model"

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> LLMResponse:
        # Check if any message content matches a predefined response
        for msg in messages:
            for key, response in self.responses.items():
                if key.lower() in msg.content.lower():
                    return LLMResponse(
                        content=response,
                        model=self.get_default_model(),
                        usage={
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    )

        # Default mock response
        return LLMResponse(
            content="MOCK RESPONSE: Analysis complete",
            model=self.get_default_model(),
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers: dict[str, LLMProvider] = {}

    @classmethod
    def get_provider(
        cls, name: str = "nvidia", model: str | None = None
    ) -> LLMProvider:
        """Get or create a provider instance."""
        cache_key = f"{name}:{model or 'default'}"
        if cache_key not in cls._providers:
            if name == "nvidia":
                cls._providers[cache_key] = (
                    NVIDIAProvider(model=model) if model else NVIDIAProvider()
                )
            elif name == "mock":
                cls._providers[cache_key] = MockLLMProvider()
            else:
                raise ValueError(f"Unknown LLM provider: {name}")

        return cls._providers[cache_key]

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
