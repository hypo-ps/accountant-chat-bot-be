"""Tests for conversation management."""

import pytest
from app.agent.conversation import Conversation, ConversationManager
from app.llm.base import Message, MessageRole


class TestConversation:
    """Tests for Conversation class."""
    
    def test_create_conversation(self):
        """Test creating a new conversation."""
        conv = Conversation()
        
        assert conv.id is not None
        assert len(conv.messages) == 0
        assert conv.system_prompt is None
    
    def test_add_user_message(self):
        """Test adding a user message."""
        conv = Conversation()
        conv.add_user_message("Hello")
        
        assert len(conv.messages) == 1
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[0].content == "Hello"
    
    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        conv = Conversation()
        conv.add_assistant_message("Hi there!")
        
        assert len(conv.messages) == 1
        assert conv.messages[0].role == MessageRole.ASSISTANT
        assert conv.messages[0].content == "Hi there!"
    
    def test_get_messages_with_limit(self):
        """Test getting limited messages."""
        conv = Conversation()
        for i in range(5):
            conv.add_user_message(f"Message {i}")
        
        messages = conv.get_messages(limit=3)
        assert len(messages) == 3
        assert messages[0].content == "Message 2"
    
    def test_clear_messages(self):
        """Test clearing messages."""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi")
        conv.clear_messages()
        
        assert len(conv.messages) == 0
    
    def test_to_dict(self):
        """Test converting conversation to dict."""
        conv = Conversation(system_prompt="You are helpful")
        conv.add_user_message("Hello")
        
        data = conv.to_dict()
        
        assert "id" in data
        assert "messages" in data
        assert data["system_prompt"] == "You are helpful"
        assert len(data["messages"]) == 1


class TestConversationManager:
    """Tests for ConversationManager class."""
    
    def test_create_conversation(self):
        """Test creating a conversation via manager."""
        manager = ConversationManager()
        conv = manager.create()
        
        assert conv.id is not None
        assert manager.get(conv.id) == conv
    
    def test_create_with_system_prompt(self):
        """Test creating conversation with system prompt."""
        manager = ConversationManager(default_system_prompt="Default prompt")
        conv = manager.create(system_prompt="Custom prompt")
        
        assert conv.system_prompt == "Custom prompt"
    
    def test_get_or_create(self):
        """Test get_or_create behavior."""
        manager = ConversationManager()
        
        # First call creates
        conv1 = manager.get_or_create("test-id")
        assert conv1.id == "test-id"
        
        # Second call returns existing
        conv2 = manager.get_or_create("test-id")
        assert conv1 is conv2
    
    def test_delete_conversation(self):
        """Test deleting a conversation."""
        manager = ConversationManager()
        conv = manager.create()
        
        assert manager.delete(conv.id) is True
        assert manager.get(conv.id) is None
        assert manager.delete(conv.id) is False
    
    def test_list_conversations(self):
        """Test listing conversations."""
        manager = ConversationManager()
        manager.create()
        manager.create()
        
        conversations = manager.list_conversations()
        assert len(conversations) == 2
    
    def test_update_system_prompt(self):
        """Test updating system prompt."""
        manager = ConversationManager()
        conv = manager.create()
        
        assert manager.update_system_prompt(conv.id, "New prompt") is True
        assert conv.system_prompt == "New prompt"
    
    def test_clear_all(self):
        """Test clearing all conversations."""
        manager = ConversationManager()
        manager.create()
        manager.create()
        manager.clear_all()
        
        assert len(manager.list_conversations()) == 0
