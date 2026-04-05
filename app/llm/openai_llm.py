import json
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from app.core.logging import get_logger
from app.llm.base import BaseLLM, LLMResponse, Message, MessageRole, ToolCall, ToolDefinition

logger = get_logger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
        logger.info("OpenAI LLM initialized", model=model)
    
    def _convert_messages(
        self, messages: list[Message], system_prompt: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        openai_messages: list[dict[str, Any]] = []
        
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            openai_msg: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content or "",
            }
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            openai_messages.append(openai_msg)
        
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
        """Generate a response using OpenAI."""
        try:
            openai_messages = self._convert_messages(messages, system_prompt)
            
            request_kwargs: dict[str, Any] = {
                "model": self.model,
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
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}", original_error=e)
            elif "connection" in error_msg or "timeout" in error_msg:
                raise LLMConnectionError(f"OpenAI connection error: {e}", original_error=e)
            else:
                raise LLMResponseError(f"OpenAI error: {e}", original_error=e)
    
    async def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using OpenAI."""
        try:
            openai_messages = self._convert_messages(messages, system_prompt)
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error("OpenAI streaming error", error=str(e))
            raise LLMResponseError(f"OpenAI streaming error: {e}", original_error=e)
    
    async def health_check(self) -> bool:
        """Check if OpenAI is accessible."""
        try:
            await self.client.models.list()
            return True
        except Exception as e:
            logger.warning("OpenAI health check failed", error=str(e))
            return False
