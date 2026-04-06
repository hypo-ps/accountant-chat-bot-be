"""
Azure OpenAI LLM implementation.
"""

import json
from typing import Any, AsyncIterator, Optional

from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.exceptions import LLMResponseError
from app.core.logging import get_logger
from app.llm.base import (
    BaseLLM,
    LLMResponse,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
)

logger = get_logger(__name__)


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI LLM implementation."""
    
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-12-01-preview",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(deployment, temperature, max_tokens, **kwargs)
        self.deployment = deployment
        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        logger.info(
            "Azure OpenAI LLM initialized",
            deployment=deployment,
            endpoint=azure_endpoint,
            api_version=api_version,
        )
    
    def _convert_messages(
        self, messages: list[Message], system_prompt: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Convert messages to OpenAI format."""
        openai_messages = []
        
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            if msg.role == MessageRole.TOOL:
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                openai_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        return openai_messages
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using Azure OpenAI."""
        try:
            openai_messages = self._convert_messages(messages, system_prompt)
            
            request_kwargs: dict[str, Any] = {
                "model": self.deployment,
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            if tools:
                request_kwargs["tools"] = [t.to_openai_format() for t in tools]
                request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")
            
            response = await self.client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in choice.message.tool_calls
                ]
            
            return LLMResponse(
                content=choice.message.content or "",
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                model=response.model,
                raw_response=response,
            )
            
        except Exception as e:
            logger.error("Azure OpenAI generation error", error=str(e))
            raise LLMResponseError(f"Azure OpenAI error: {e}", original_error=e)

    async def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Azure OpenAI."""
        try:
            openai_messages = self._convert_messages(messages, system_prompt)

            stream = await self.client.chat.completions.create(
                model=self.deployment,
                messages=openai_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("Azure OpenAI streaming error", error=str(e))
            raise LLMResponseError(f"Azure OpenAI streaming error: {e}", original_error=e)

    async def health_check(self) -> bool:
        """Check if Azure OpenAI is accessible."""
        try:
            # Make a simple API call to verify connectivity
            await self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            logger.warning("Azure OpenAI health check failed", error=str(e))
            return False