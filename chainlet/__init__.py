"""
Chainlet Framework for Ollama Integration

This package provides a simple chainlet framework for interacting with Ollama models.
"""

from .core import Chainlet, Message, Role
from .ollama import OllamaChainlet

__all__ = ['Chainlet', 'Message', 'Role', 'OllamaChainlet']
