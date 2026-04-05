#!/usr/bin/env python3
"""
Example demonstrating tool usage with the chatbot.

This example shows how to:
1. Define custom tools
2. Register tools with the agent
3. Let the agent use tools to answer questions
"""

import asyncio
import math
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from app.llm.factory import create_llm
from app.agent.agent import Agent
from app.agent.tools import BaseTool, ToolResult, ToolRegistry


# Define custom tools

class CalculatorTool(BaseTool):
    """A calculator tool for basic math operations."""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Performs mathematical calculations. Supports basic arithmetic (+, -, *, /), powers (**), and common functions like sqrt, sin, cos."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '10 * 5')"
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression safely."""
        try:
            # Safe math evaluation
            allowed_names = {
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'log10': math.log10,
                'pi': math.pi,
                'e': math.e,
                'abs': abs,
                'round': round,
                'pow': pow,
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return ToolResult(success=True, output=f"Result: {result}")
        except Exception as e:
            return ToolResult(success=False, output=f"Error: {str(e)}", error=str(e))


class DateTimeTool(BaseTool):
    """Tool for getting current date and time information."""
    
    @property
    def name(self) -> str:
        return "get_datetime"
    
    @property
    def description(self) -> str:
        return "Gets the current date and time, or calculates date differences."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Output format: 'full', 'date', 'time', or 'iso'",
                    "enum": ["full", "date", "time", "iso"]
                }
            },
            "required": []
        }
    
    async def execute(self, format: str = "full") -> ToolResult:
        """Get current datetime in the specified format."""
        now = datetime.now()
        
        formats = {
            "full": now.strftime("%A, %B %d, %Y at %I:%M %p"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "iso": now.isoformat(),
        }
        
        result = formats.get(format, formats["full"])
        return ToolResult(success=True, output=result)


async def main():
    """Run a chat session with tools."""
    print("=" * 60)
    print("Accountant Chatbot - Tools Example")
    print("=" * 60)
    
    # Initialize LLM
    llm = create_llm()
    
    # Create tool registry and register tools
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(DateTimeTool())
    
    print(f"\nRegistered tools: {[t.name for t in tool_registry.list_tools()]}")
    
    # Create agent with tools
    agent = Agent(
        llm=llm,
        tool_registry=tool_registry,
        max_iterations=5,
    )
    
    # Test the tools
    print("\n" + "-" * 60)
    
    question = "What is 15% of 2500, and what is today's date?"
    print(f"\nUser: {question}")
    
    response = await agent.chat(message=question, use_tools=True)
    
    print(f"\nAssistant: {response.content}")
    print(f"\nTool calls made: {len(response.tool_calls_made)}")
    for tc in response.tool_calls_made:
        print(f"  - {tc['tool']}: {tc['arguments']} -> {tc['result']['output']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
