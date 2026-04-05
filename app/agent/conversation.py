from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from app.core.logging import get_logger
from app.llm.base import Message, MessageRole

logger = get_logger(__name__)


@dataclass
class Conversation:
    """Represents a conversation with message history."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.add_message(Message(role=MessageRole.USER, content=content))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.add_message(Message(role=MessageRole.ASSISTANT, content=content))
    
    def get_messages(self, limit: Optional[int] = None) -> list[Message]:
        """Get conversation messages, optionally limited."""
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:]
    
    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create conversation from dictionary."""
        messages = [
            Message(
                role=MessageRole(m["role"]),
                content=m["content"],
                name=m.get("name"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in data.get("messages", [])
        ]
        
        return cls(
            id=data.get("id", str(uuid4())),
            messages=messages,
            system_prompt=data.get("system_prompt"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )


class ConversationManager:
    """Manages multiple conversations."""

    def __init__(self, default_system_prompt: Optional[str] = None) -> None:
        """
        Initialize the conversation manager.

        Args:
            default_system_prompt: Default system prompt for new conversations.
                                   If None, uses the accountant prompt from config.
        """
        if default_system_prompt is None:
            from app.core.prompts import ACCOUNTANT_SYSTEM_PROMPT
            default_system_prompt = ACCOUNTANT_SYSTEM_PROMPT
        self._conversations: dict[str, Conversation] = {}
        self.default_system_prompt = default_system_prompt
    
    def create(
        self,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Optional custom ID
            system_prompt: System prompt for this conversation
            metadata: Additional metadata
            
        Returns:
            The created conversation
        """
        conversation = Conversation(
            id=conversation_id or str(uuid4()),
            system_prompt=system_prompt or self.default_system_prompt,
            metadata=metadata or {},
        )
        self._conversations[conversation.id] = conversation
        logger.info("Conversation created", conversation_id=conversation.id)
        return conversation
    
    def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)
    
    def get_or_create(
        self,
        conversation_id: str,
        system_prompt: Optional[str] = None,
    ) -> Conversation:
        """Get an existing conversation or create a new one."""
        conversation = self.get(conversation_id)
        if conversation is None:
            conversation = self.create(
                conversation_id=conversation_id,
                system_prompt=system_prompt,
            )
        return conversation
    
    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation by ID."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info("Conversation deleted", conversation_id=conversation_id)
            return True
        return False
    
    def list_conversations(self) -> list[dict[str, Any]]:
        """List all conversations with basic info."""
        return [
            {
                "id": conv.id,
                "message_count": len(conv.messages),
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            }
            for conv in self._conversations.values()
        ]
    
    def update_system_prompt(
        self,
        conversation_id: str,
        system_prompt: str,
    ) -> bool:
        """Update the system prompt for a conversation."""
        conversation = self.get(conversation_id)
        if conversation:
            conversation.system_prompt = system_prompt
            conversation.updated_at = datetime.utcnow()
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all conversations."""
        self._conversations.clear()
        logger.info("All conversations cleared")
