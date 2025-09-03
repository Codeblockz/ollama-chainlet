"""
MCP Infrastructure for Chainlet Framework

This module provides the core MCP (Model Context Protocol) infrastructure for
integrating external tools and resources into chainlet conversations.
"""

import json
import logging
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager, AsyncExitStack

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError as e:
    MCP_AVAILABLE = False
    MCP_IMPORT_ERROR = str(e)
    
    # Update the error class with specific message
    def _create_mcp_error(operation=""):
        return MCPNotAvailableError(f"MCP dependencies not available: {MCP_IMPORT_ERROR}. "
                           f"Install with: pip install mcp{f' (attempted: {operation})' if operation else ''}")
    
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    streamablehttp_client = None
    types = None


logger = logging.getLogger(__name__)


class MCPNotAvailableError(ImportError):
    """Raised when MCP dependencies are not available."""
    pass


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
        if not MCP_AVAILABLE:
            raise _create_mcp_error("MCPClient initialization")
        
        self.config = config
        self._session = None
        self._tools: List[MCPTool] = []
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._exit_stack = None
        self._last_used = 0
        self._connection_timeout = 300  # 5 minutes
    
    async def _ensure_connection(self):
        """Ensure we have an active connection, creating one if needed."""
        async with self._connection_lock:
            import time
            current_time = time.time()
            
            # Check if connection exists and is not stale
            if (self._session and self._connected and 
                current_time - self._last_used < self._connection_timeout):
                self._last_used = current_time
                return self._session
            
            # Clean up old connection properly
            await self._cleanup_connection()
            
            # Create new connection using AsyncExitStack
            try:
                self._exit_stack = AsyncExitStack()
                
                if self.config.transport == "stdio":
                    if not stdio_client or not StdioServerParameters:
                        raise _create_mcp_error("stdio client setup")
                    if not self.config.command:
                        raise ValueError("stdio transport requires command")
                        
                    server_params = StdioServerParameters(
                        command=self.config.command,
                        args=self.config.args or [],
                        env=self.config.env or {}
                    )
                    
                    # Use AsyncExitStack for proper resource management
                    read, write = await self._exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    
                elif self.config.transport == "http":
                    if not streamablehttp_client:
                        raise _create_mcp_error("http client setup")
                    if not self.config.url:
                        raise ValueError("http transport requires url")
                        
                    # Use AsyncExitStack for proper resource management
                    read, write, _ = await self._exit_stack.enter_async_context(
                        streamablehttp_client(self.config.url)
                    )
                    
                else:
                    raise ValueError(f"Unsupported transport: {self.config.transport}")
                
                # Initialize the session
                if not ClientSession:
                    raise _create_mcp_error("session initialization")
                session = ClientSession(read, write)
                await session.initialize()
                self._session = session
                self._connected = True
                self._last_used = current_time
                
                return self._session
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
                await self._cleanup_connection()
                raise
    
    @asynccontextmanager
    async def connect(self):
        """Connect to the MCP server and yield the session."""
        try:
            session = await self._ensure_connection()
            yield session
        finally:
            # Connection cleanup is handled by the connection manager
            pass
    
    async def discover_tools(self) -> List[MCPTool]:
        """Discover available tools from the server."""
        try:
            session = await self._ensure_connection()
            tools_response = await session.list_tools()
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
    
    async def _cleanup_connection(self):
        """Properly clean up connection resources."""
        try:
            if self._exit_stack:
                try:
                    await self._exit_stack.aclose()
                except Exception as e:
                    logger.debug(f"Error cleaning up exit stack: {e}")
                finally:
                    self._exit_stack = None
        except Exception as e:
            logger.debug(f"Error in connection cleanup: {e}")
        finally:
            self._session = None
            self._connected = False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool and return the result."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                session = await self._ensure_connection()
                
                # Add timeout for tool calls
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments),
                    timeout=60  # 1 minute timeout for tool calls
                )
                
                # Handle different content types
                if result.content:
                    content_block = result.content[0]
                    if types and hasattr(types, 'TextContent') and isinstance(content_block, types.TextContent):
                        return {
                            "success": True,
                            "result": content_block.text,
                            "structured": getattr(result, 'structuredContent', None)
                        }
                    else:
                        return {
                            "success": True,
                            "result": str(content_block),
                            "structured": getattr(result, 'structuredContent', None)
                        }
                
                return {
                    "success": True,
                    "result": "Tool executed successfully",
                    "structured": getattr(result, 'structuredContent', None)
                }
            
            except asyncio.TimeoutError:
                logger.error(f"Tool call {tool_name} timed out")
                return {
                    "success": False,
                    "error": f"Tool call timed out after 60 seconds"
                }
            except ConnectionError as e:
                logger.error(f"Connection error calling tool {tool_name}: {e}")
                await self._cleanup_connection()
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying tool call {tool_name} due to connection error (attempt {attempt + 2}/{max_retries})")
                    continue
                
                return {
                    "success": False,
                    "error": f"Connection error: {str(e)}"
                }
            except Exception as e:
                logger.error(f"Failed to call tool {tool_name}: {e}")
                
                # Clean up connection on error and retry once
                await self._cleanup_connection()
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying tool call {tool_name} (attempt {attempt + 2}/{max_retries})")
                    continue
                
                return {
                    "success": False,
                    "error": str(e)
                }
        
        # Fallback return (should never be reached)
        return {
            "success": False,
            "error": "Tool call failed after all retries"
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
        if not MCP_AVAILABLE:
            raise _create_mcp_error("MCPManager initialization")
        
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
        
        # Use asyncio.gather for parallel connections
        async def connect_server(server_name, client):
            try:
                tools = await client.discover_tools()
                discovered_tools = []
                for tool in tools:
                    if tool.name in self._all_tools:
                        logger.warning(f"Tool name conflict: {tool.name} exists in multiple servers")
                        # Prefix with server name to avoid conflicts
                        prefixed_name = f"{server_name}.{tool.name}"
                        discovered_tools.append((prefixed_name, tool, server_name))
                    else:
                        discovered_tools.append((tool.name, tool, server_name))
                
                logger.info(f"Connected to {server_name}, discovered {len(tools)} tools")
                return discovered_tools
            
            except Exception as e:
                logger.error(f"Failed to connect to server {server_name}: {e}")
                return []
        
        # Connect to all servers in parallel
        tasks = [connect_server(name, client) for name, client in self._clients.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                continue
            if not isinstance(result, list):
                continue
            for tool_name, tool, server_name in result:
                self._all_tools[tool_name] = tool
                self._server_for_tool[tool_name] = server_name
    
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
    
    async def cleanup(self) -> None:
        """Clean up all client connections."""
        cleanup_tasks = []
        for client in self._clients.values():
            cleanup_tasks.append(client._cleanup_connection())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)