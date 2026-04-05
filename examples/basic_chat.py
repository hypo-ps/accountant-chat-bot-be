#!/usr/bin/env python3
"""
Basic chat example demonstrating the chatbot usage.

This example shows how to:
1. Initialize the chatbot components
2. Have a simple conversation
3. Use custom system prompts
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.llm.factory import create_llm
from app.agent.agent import Agent
from app.agent.conversation import ConversationManager
from app.core.config import settings


async def main():
    """Run a simple chat session."""
    print("=" * 60)
    print("Accountant Chatbot - Basic Example")
    print("=" * 60)
    
    # Initialize LLM
    print("\n[1] Initializing LLM...")
    llm = create_llm()
    print(f"    LLM Provider: {settings.llm_provider.value}")
    
    # Initialize conversation manager with custom system prompt
    print("\n[2] Setting up conversation manager...")
    conversation_manager = ConversationManager(
        default_system_prompt="""You are a helpful AI assistant specialized in accounting 
and financial matters. You provide accurate, professional advice while being 
conversational and approachable. Always ask clarifying questions when needed."""
    )
    
    # Initialize agent
    print("\n[3] Creating agent...")
    agent = Agent(
        llm=llm,
        conversation_manager=conversation_manager,
        max_iterations=5,
    )
    
    # Have a conversation
    print("\n[4] Starting conversation...")
    print("-" * 60)
    
    # First message
    user_message = "Hi! Can you explain what depreciation means in accounting?"
    print(f"\nUser: {user_message}")
    
    response = await agent.chat(
        message=user_message,
        use_tools=False,  # No tools for this simple example
    )
    
    print(f"\nAssistant: {response.content}")
    print(f"\n[Conversation ID: {response.conversation_id}]")
    print(f"[Tokens used: {response.usage}]")
    
    # Follow-up message (continues the conversation)
    print("-" * 60)
    
    follow_up = "What are the common methods for calculating it?"
    print(f"\nUser: {follow_up}")
    
    response = await agent.chat(
        message=follow_up,
        conversation_id=response.conversation_id,  # Continue same conversation
        use_tools=False,
    )
    
    print(f"\nAssistant: {response.content}")
    
    print("\n" + "=" * 60)
    print("Chat session complete!")


if __name__ == "__main__":
    asyncio.run(main())
