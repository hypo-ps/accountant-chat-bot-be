from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.core.exceptions import ToolExecutionError
from app.core.logging import get_logger
from app.llm.base import ToolDefinition

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
        }


class BaseTool(ABC):
    """Abstract base class for tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used by LLM to call the tool)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description (tells LLM what the tool does)."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            ToolResult with the execution output
        """
        pass
    
    def to_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for LLM."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class FunctionTool(BaseTool):
    """Tool wrapper for simple functions."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
        is_async: bool = False,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._func = func
        self._is_async = is_async
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function."""
        try:
            if self._is_async:
                result = await self._func(**kwargs)
            else:
                result = self._func(**kwargs)
            
            return ToolResult(
                success=True,
                output=str(result) if not isinstance(result, str) else result,
                data=result if isinstance(result, dict) else None,
            )
        except Exception as e:
            logger.error("Tool execution failed", tool=self.name, error=str(e))
            return ToolResult(
                success=False,
                output=f"Error: {str(e)}",
                error=str(e),
            )


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info("Tool registered", tool_name=tool.name)
    
    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            logger.info("Tool unregistered", tool_name=name)
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[BaseTool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def get_definitions(self) -> list[ToolDefinition]:
        """Get ToolDefinitions for all registered tools."""
        return [tool.to_definition() for tool in self._tools.values()]
    
    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            raise ToolExecutionError(
                f"Tool not found: {name}",
                details={"available_tools": list(self._tools.keys())},
            )
        
        logger.info("Executing tool", tool_name=name, params=kwargs)
        return await tool.execute(**kwargs)
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()


# Global tool registry
tool_registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    is_async: bool = False,
) -> Callable:
    """
    Decorator to register a function as a tool.
    
    Usage:
        @tool(
            name="calculator",
            description="Performs basic math operations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        )
        def calculator(expression: str) -> str:
            return str(eval(expression))
    """
    def decorator(func: Callable) -> Callable:
        tool_instance = FunctionTool(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
            is_async=is_async,
        )
        tool_registry.register(tool_instance)
        return func
    return decorator
