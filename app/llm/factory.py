"""
LLM Factory for creating LLM instances based on configuration.

Supports:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
"""

from functools import lru_cache
from typing import Optional

from app.core.config import LLMProvider, settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.llm.base import BaseLLM
from app.llm.anthropic_llm import AnthropicLLM
from app.llm.openai_llm import OpenAILLM

logger = get_logger(__name__)


def create_llm(
    provider: Optional[LLMProvider] = None,
    **kwargs,
) -> BaseLLM:
    """
    Create an LLM instance based on the specified provider.
    
    Args:
        provider: The LLM provider to use. Defaults to settings.llm_provider
        **kwargs: Additional arguments to pass to the LLM constructor
        
    Returns:
        An instance of the appropriate LLM implementation
        
    Raises:
        ConfigurationError: If the provider is not configured properly
    """
    provider = provider or settings.llm_provider
    
    logger.info("Creating LLM instance", provider=provider.value)
    
    if provider == LLMProvider.OPENAI:
        api_key = kwargs.pop("api_key", None) or settings.openai_api_key
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not configured",
                details={"provider": "openai", "config_key": "OPENAI_API_KEY"},
            )
        
        return OpenAILLM(
            api_key=api_key,
            model=kwargs.pop("model", None) or settings.openai_model,
            temperature=kwargs.pop("temperature", None) or settings.openai_temperature,
            max_tokens=kwargs.pop("max_tokens", None) or settings.openai_max_tokens,
            **kwargs,
        )
    
    elif provider == LLMProvider.ANTHROPIC:
        api_key = kwargs.pop("api_key", None) or settings.anthropic_api_key
        if not api_key:
            raise ConfigurationError(
                "Anthropic API key not configured",
                details={"provider": "anthropic", "config_key": "ANTHROPIC_API_KEY"},
            )

        return AnthropicLLM(
            api_key=api_key,
            model=kwargs.pop("model", None) or settings.anthropic_model,
            temperature=kwargs.pop("temperature", None) or settings.anthropic_temperature,
            max_tokens=kwargs.pop("max_tokens", None) or settings.anthropic_max_tokens,
            **kwargs,
        )

    else:
        raise ConfigurationError(
            f"Unsupported LLM provider: {provider}",
            details={"provider": provider, "supported": [p.value for p in LLMProvider]},
        )


@lru_cache
def get_llm() -> BaseLLM:
    """
    Get a cached LLM instance based on current settings.
    
    Returns:
        A cached LLM instance
    """
    return create_llm()
