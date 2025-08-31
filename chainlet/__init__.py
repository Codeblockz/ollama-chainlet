"""
Chainlet Framework for Ollama Integration

This package provides a simple chainlet framework for interacting with Ollama models
with optional Model Context Protocol (MCP) support for external tools and resources.
"""

from .core import Chainlet, Message, Role
from .ollama import OllamaChainlet
from .mcp_chainlet import MCPChainlet
from .mcp import MCPManager, MCPClient, MCPTool, MCPServerConfig

__all__ = [
    'Chainlet', 'Message', 'Role', 'OllamaChainlet', 
    'MCPChainlet', 'MCPManager', 'MCPClient', 'MCPTool', 'MCPServerConfig'
]
