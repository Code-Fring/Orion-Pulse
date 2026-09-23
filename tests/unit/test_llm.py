"""Unit tests for LLM module."""

from unittest.mock import Mock, patch

import pytest

from orion_pulse.llm import (
    LLMMessage,
    LLMProviderError,
    LLMProviderFactory,
    LLMResponse,
    MockLLMProvider,
    NVIDIAProvider,
)


class TestLLMMessage:
    def test_create_message(self):
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestLLMResponse:
    def test_create_response(self):
        resp = LLMResponse(
            content="Response text",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert resp.content == "Response text"
        assert resp.model == "test-model"
        assert resp.usage["total_tokens"] == 15


class TestMockLLMProvider:
    def test_get_provider_name(self):
        provider = MockLLMProvider()
        assert provider.get_provider_name() == "mock"

    def test_is_available(self):
        provider = MockLLMProvider()
        assert provider.is_available() is True

    def test_get_default_model(self):
        provider = MockLLMProvider()
        assert provider.get_default_model() == "mock-model"

    def test_complete_basic(self):
        provider = MockLLMProvider()
        messages = [LLMMessage(role="user", content="Test query")]

        response = provider.complete(messages)

        assert isinstance(response, LLMResponse)
        assert "MOCK" in response.content
        assert response.model == "mock-model"

    def test_complete_with_custom_responses(self):
        responses = {
            "earnings": "Earnings analysis complete",
            "forecast": "Forecast generated",
        }
        provider = MockLLMProvider(responses=responses)

        messages = [LLMMessage(role="user", content="Analyze earnings report")]
        response = provider.complete(messages)

        assert "Earnings analysis complete" in response.content


class TestNVIDIAProvider:
    def test_get_provider_name(self):
        provider = NVIDIAProvider(api_key="test-key")
        assert provider.get_provider_name() == "nvidia"

    def test_is_available_with_key(self):
        provider = NVIDIAProvider(api_key="test-key")
        assert provider.is_available() is True

    def test_is_available_without_key(self):
        provider = NVIDIAProvider(api_key=None)
        assert provider.is_available() is False

    def test_get_default_model(self):
        provider = NVIDIAProvider(api_key="test-key", model="nvidia/test-model")
        assert provider.get_default_model() == "nvidia/test-model"

    def test_complete_without_key_raises(self):
        provider = NVIDIAProvider(api_key=None)

        with pytest.raises(LLMProviderError, match="not configured"):
            provider.complete([LLMMessage(role="user", content="Test")])

    @patch("requests.post")
    def test_complete_success(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "model": "nvidia/nemotron-3-ultra",
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_post.return_value = mock_response

        provider = NVIDIAProvider(api_key="test-key")
        messages = [LLMMessage(role="user", content="Test")]

        response = provider.complete(messages)

        assert response.content == "Test response"
        assert response.model == "nvidia/nemotron-3-ultra"
        assert response.usage["total_tokens"] == 30

    @patch("requests.post")
    def test_complete_timeout(self, mock_post):
        import requests

        mock_post.side_effect = requests.exceptions.Timeout()

        provider = NVIDIAProvider(api_key="test-key")

        with pytest.raises(LLMProviderError, match="timed out"):
            provider.complete([LLMMessage(role="user", content="Test")])

    @patch("requests.post")
    def test_complete_request_error(self, mock_post):
        import requests

        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        provider = NVIDIAProvider(api_key="test-key")

        with pytest.raises(LLMProviderError, match="NVIDIA API error"):
            provider.complete([LLMMessage(role="user", content="Test")])


class TestLLMProviderFactory:
    def test_get_nvidia_provider(self):
        LLMProviderFactory.clear_cache()
        with patch("orion_pulse.config.settings") as mock_settings:
            mock_settings.nvidia_api_key = "test-key"
            provider = LLMProviderFactory.get_provider("nvidia")
            assert isinstance(provider, NVIDIAProvider)

    def test_get_mock_provider(self):
        LLMProviderFactory.clear_cache()
        provider = LLMProviderFactory.get_provider("mock")
        assert isinstance(provider, MockLLMProvider)

    def test_get_unknown_provider_raises(self):
        LLMProviderFactory.clear_cache()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMProviderFactory.get_provider("unknown")

    def test_get_available_providers(self):
        LLMProviderFactory.clear_cache()
        # Create NVIDIA provider with API key directly
        provider = NVIDIAProvider(api_key="test-key")
        # Manually add to factory cache
        LLMProviderFactory._providers["nvidia"] = provider

        available = LLMProviderFactory.get_available_providers()
        assert "nvidia" in available
        assert "mock" in available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
