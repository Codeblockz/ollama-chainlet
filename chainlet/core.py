"""
Core Chainlet Framework

This module provides the core functionality for the chainlet framework.
"""

from enum import Enum
from typing import List, Dict, Any, Optional


class Role(Enum):
    """Enum for message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message:
    """
    Represents a message in a conversation.
    
    Attributes:
        role (Role): The role of the message sender (system, user, or assistant).
        content (str): The content of the message.
    """
    
    def __init__(self, role: Role, content: str):
        """
        Initialize a new message.
        
        Args:
            role (Role): The role of the message sender.
            content (str): The content of the message.
        """
        self.role = role
        self.content = content
    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert the message to a dictionary.
        
        Returns:
            Dict[str, str]: A dictionary representation of the message.
        """
        return {
            "role": self.role.value,
            "content": self.content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Message':
        """
        Create a message from a dictionary.
        
        Args:
            data (Dict[str, str]): A dictionary representation of the message.
            
        Returns:
            Message: A new message instance.
        """
        return cls(
            role=Role(data["role"]),
            content=data["content"]
        )


class Chainlet:
    """
    Base class for chainlet implementations.
    
    A chainlet manages a conversation with a model and provides methods for
    sending messages and receiving responses.
    
    Attributes:
        messages (List[Message]): The conversation history.
        system_prompt (str): The system prompt for the conversation.
    """
    
    def __init__(self, system_prompt: str = ""):
        """
        Initialize a new chainlet.
        
        Args:
            system_prompt (str, optional): The system prompt for the conversation.
        """
        self.messages: List[Message] = []
        if system_prompt:
            self.messages.append(Message(Role.SYSTEM, system_prompt))
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            message (Message): The message to add.
        """
        self.messages.append(message)
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            content (str): The content of the user message.
        """
        self.add_message(Message(Role.USER, content))
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            content (str): The content of the assistant message.
        """
        self.add_message(Message(Role.ASSISTANT, content))
    
    def get_messages(self) -> List[Message]:
        """
        Get the conversation history.
        
        Returns:
            List[Message]: The conversation history.
        """
        return self.messages
    
    def get_messages_as_dicts(self) -> List[Dict[str, str]]:
        """
        Get the conversation history as a list of dictionaries.
        
        Returns:
            List[Dict[str, str]]: The conversation history as dictionaries.
        """
        return [message.to_dict() for message in self.messages]
    
    def clear_messages(self, keep_system_prompt: bool = True) -> None:
        """
        Clear the conversation history.
        
        Args:
            keep_system_prompt (bool, optional): Whether to keep the system prompt.
        """
        if keep_system_prompt and self.messages and self.messages[0].role == Role.SYSTEM:
            system_prompt = self.messages[0]
            self.messages = [system_prompt]
        else:
            self.messages = []
    
    def generate(self, user_message: Optional[str] = None) -> str:
        """
        Generate a response from the model.
        
        Args:
            user_message (Optional[str], optional): A user message to add before generating.
            
        Returns:
            str: The generated response.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        if user_message:
            self.add_user_message(user_message)
        
        raise NotImplementedError("Subclasses must implement the generate method.")
