"""Tests for tool system."""

import pytest
from app.agent.tools import BaseTool, ToolResult, ToolRegistry, FunctionTool


class MockTool(BaseTool):
    """A mock tool for testing."""
    
    @property
    def name(self) -> str:
        return "mock_tool"
    
    @property
    def description(self) -> str:
        return "A mock tool for testing"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }
    
    async def execute(self, input: str) -> ToolResult:
        return ToolResult(success=True, output=f"Received: {input}")


class TestToolResult:
    """Tests for ToolResult."""
    
    def test_successful_result(self):
        """Test successful tool result."""
        result = ToolResult(success=True, output="Done")
        
        assert result.success is True
        assert result.output == "Done"
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed tool result."""
        result = ToolResult(success=False, output="Failed", error="Error message")
        
        assert result.success is False
        assert result.error == "Error message"
    
    def test_to_dict(self):
        """Test converting to dict."""
        result = ToolResult(success=True, output="Done", data={"key": "value"})
        data = result.to_dict()
        
        assert data["success"] is True
        assert data["output"] == "Done"
        assert data["data"] == {"key": "value"}


class TestToolRegistry:
    """Tests for ToolRegistry."""
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        
        assert registry.get("mock_tool") == tool
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        registry.unregister("mock_tool")
        
        assert registry.get("mock_tool") is None
    
    def test_list_tools(self):
        """Test listing tools."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0] == tool
    
    def test_get_definitions(self):
        """Test getting tool definitions."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        
        definitions = registry.get_definitions()
        assert len(definitions) == 1
        assert definitions[0].name == "mock_tool"
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        
        result = await registry.execute("mock_tool", input="test")
        
        assert result.success is True
        assert result.output == "Received: test"
    
    def test_clear_registry(self):
        """Test clearing registry."""
        registry = ToolRegistry()
        registry.register(MockTool())
        registry.clear()
        
        assert len(registry.list_tools()) == 0


class TestFunctionTool:
    """Tests for FunctionTool wrapper."""
    
    @pytest.mark.asyncio
    async def test_sync_function(self):
        """Test wrapping a sync function."""
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = FunctionTool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                }
            },
            func=add,
            is_async=False,
        )
        
        result = await tool.execute(a=2, b=3)
        
        assert result.success is True
        assert result.output == "5"
    
    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test wrapping an async function."""
        async def async_greet(name: str) -> str:
            return f"Hello, {name}!"
        
        tool = FunctionTool(
            name="greet",
            description="Greet someone",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            func=async_greet,
            is_async=True,
        )
        
        result = await tool.execute(name="World")
        
        assert result.success is True
        assert result.output == "Hello, World!"
    
    @pytest.mark.asyncio
    async def test_function_error(self):
        """Test function that raises an error."""
        def failing_func():
            raise ValueError("Test error")
        
        tool = FunctionTool(
            name="fail",
            description="Always fails",
            parameters={"type": "object", "properties": {}},
            func=failing_func,
            is_async=False,
        )
        
        result = await tool.execute()
        
        assert result.success is False
        assert "Test error" in result.error
