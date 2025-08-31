"""
MCP Infrastructure for Chainlet Framework

This module provides the core MCP (Model Context Protocol) infrastructure for
integrating external tools and resources into chainlet conversations.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    import mcp.types as types
except ImportError as e:
    print(f"Warning: MCP dependencies not found: {e}")
    # Provide fallback classes/functions
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    streamablehttp_client = None
    types = None


logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents an MCP tool with its metadata."""
    name: str
    description: Optional[str]
    input_schema: Optional[Dict[str, Any]]
    output_schema: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    transport: str  # "stdio" or "http"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    
    def validate(self) -> None:
        """Validate server configuration."""
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"Server {self.name}: stdio transport requires command")
        elif self.transport == "http" and not self.url:
            raise ValueError(f"Server {self.name}: http transport requires url")


class MCPClient:
    """Wrapper around MCP SDK client for standardized operations."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._session = None
        self._tools: List[MCPTool] = []
        self._connected = False
    
    @asynccontextmanager
    async def connect(self):
        """Connect to the MCP server and yield the session."""
        try:
            if self.config.transport == "stdio":
                if StdioServerParameters is None:
                    raise RuntimeError("MCP dependencies not available")
                server_params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args or [],
                    env=self.config.env or {}
                )
                if stdio_client is None or ClientSession is None:
                    raise RuntimeError("MCP dependencies not available")
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._session = session
                        self._connected = True
                        yield session
            
            elif self.config.transport == "http":
                if streamablehttp_client is None or ClientSession is None:
                    raise RuntimeError("MCP dependencies not available")
                async with streamablehttp_client(self.config.url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._session = session
                        self._connected = True
                        yield session
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")
        
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            raise
        finally:
            self._session = None
            self._connected = False
    
    async def discover_tools(self) -> List[MCPTool]:
        """Discover available tools from the server."""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")
        
        try:
            tools_response = await self._session.list_tools()
            self._tools = []
            
            for tool in tools_response.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    output_schema=getattr(tool, 'outputSchema', None)
                )
                self._tools.append(mcp_tool)
            
            return self._tools
        
        except Exception as e:
            logger.error(f"Failed to discover tools from {self.config.name}: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool and return the result."""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")
        
        try:
            result = await self._session.call_tool(tool_name, arguments)
            
            # Handle different content types
            if result.content:
                content_block = result.content[0]
                if types and hasattr(types, 'TextContent') and isinstance(content_block, types.TextContent):
                    return {
                        "success": True,
                        "result": content_block.text,
                        "structured": result.structuredContent
                    }
                else:
                    return {
                        "success": True,
                        "result": str(content_block),
                        "structured": result.structuredContent
                    }
            
            return {
                "success": True,
                "result": "Tool executed successfully",
                "structured": result.structuredContent
            }
        
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tools(self) -> List[MCPTool]:
        """Get cached tools list."""
        return self._tools
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected


class MCPManager:
    """Manages multiple MCP server connections and tool operations."""
    
    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}
        self._all_tools: Dict[str, MCPTool] = {}
        self._server_for_tool: Dict[str, str] = {}
    
    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        config.validate()
        self._clients[config.name] = MCPClient(config)
    
    def remove_server(self, server_name: str) -> None:
        """Remove an MCP server."""
        if server_name in self._clients:
            del self._clients[server_name]
            # Remove tools from this server
            tools_to_remove = [
                tool_name for tool_name, srv_name in self._server_for_tool.items()
                if srv_name == server_name
            ]
            for tool_name in tools_to_remove:
                del self._all_tools[tool_name]
                del self._server_for_tool[tool_name]
    
    async def connect_all(self) -> None:
        """Connect to all configured servers and discover tools."""
        self._all_tools.clear()
        self._server_for_tool.clear()
        
        for server_name, client in self._clients.items():
            try:
                async with client.connect() as session:
                    tools = await client.discover_tools()
                    for tool in tools:
                        if tool.name in self._all_tools:
                            logger.warning(f"Tool name conflict: {tool.name} exists in multiple servers")
                            # Prefix with server name to avoid conflicts
                            prefixed_name = f"{server_name}.{tool.name}"
                            self._all_tools[prefixed_name] = tool
                            self._server_for_tool[prefixed_name] = server_name
                        else:
                            self._all_tools[tool.name] = tool
                            self._server_for_tool[tool.name] = server_name
                    
                    logger.info(f"Connected to {server_name}, discovered {len(tools)} tools")
            
            except Exception as e:
                logger.error(f"Failed to connect to server {server_name}: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name with arguments."""
        if tool_name not in self._server_for_tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        server_name = self._server_for_tool[tool_name]
        client = self._clients[server_name]
        
        try:
            async with client.connect() as session:
                return await client.call_tool(tool_name, arguments)
        
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on server {server_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get all available tools from all servers."""
        return list(self._all_tools.values())
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        return self._all_tools.get(tool_name)
    
    def get_tools_summary(self) -> str:
        """Get a formatted summary of all available tools for LLM context."""
        if not self._all_tools:
            return "No MCP tools available."
        
        summary = "Available MCP Tools:\n"
        for tool in self._all_tools.values():
            summary += f"- {tool.name}: {tool.description or 'No description'}\n"
        
        return summary
    
    def load_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load server configurations from a dictionary."""
        servers = config_dict.get("mcpServers", {})
        
        for server_name, server_config in servers.items():
            # Determine transport type based on configuration
            if "command" in server_config:
                transport = "stdio"
                config = MCPServerConfig(
                    name=server_name,
                    transport=transport,
                    command=server_config["command"],
                    args=server_config.get("args", []),
                    env=server_config.get("env", {})
                )
            elif "url" in server_config:
                transport = "http"
                config = MCPServerConfig(
                    name=server_name,
                    transport=transport,
                    url=server_config["url"]
                )
            else:
                logger.warning(f"Invalid server config for {server_name}: missing command or url")
                continue
            
            self.add_server(config)
    
    def load_config_from_file(self, config_path: str) -> None:
        """Load server configurations from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.load_config_from_dict(config_dict)
        except Exception as e:
            logger.error(f"Failed to load MCP config from {config_path}: {e}")
    
    def get_server_names(self) -> List[str]:
        """Get list of configured server names."""
        return list(self._clients.keys())