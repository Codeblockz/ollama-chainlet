"""
MCP-Enhanced Chainlet Implementation

This module provides an MCP-enabled chainlet that extends OllamaChainlet
with Model Context Protocol capabilities for tool and resource access.
"""

import json
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, Iterator, List, Callable

from .ollama import OllamaChainlet
from .mcp import MCPManager, MCPServerConfig, MCP_AVAILABLE, MCPNotAvailableError


logger = logging.getLogger(__name__)


class MCPChainlet(OllamaChainlet):
    """
    An MCP-enabled chainlet that extends OllamaChainlet with tool capabilities.
    
    This chainlet automatically discovers and integrates MCP tools into conversations,
    allowing the language model to access external tools and resources.
    """
    
    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        mcp_config: Optional[Dict[str, Any]] = None,
        timeout: int = None
    ):
        """
        Initialize an MCP-enabled chainlet.
        
        Args:
            model (str): The name of the Ollama model to use.
            system_prompt (str, optional): The system prompt for the conversation.
            base_url (str, optional): The base URL for the Ollama API.
            temperature (float, optional): The temperature parameter for generation.
            max_tokens (int, optional): The maximum number of tokens to generate.
            mcp_config (Dict[str, Any], optional): MCP server configuration.
        """
        # Initialize MCP manager first (only if MCP is available)
        if MCP_AVAILABLE:
            self.mcp_manager = MCPManager()
            self._mcp_enabled = False
            self._tools_context = ""
            self._mcp_tool_functions = []
            
            # Load MCP configuration if provided
            if mcp_config:
                self._load_mcp_config(mcp_config)
        else:
            self.mcp_manager = None
            self._mcp_enabled = False
            self._tools_context = ""
            self._mcp_tool_functions = []
            if mcp_config:
                logger.warning("MCP configuration provided but MCP dependencies not available. Install with: pip install mcp")
        
        # Initialize the base chainlet with MCP tool functions
        super().__init__(model, system_prompt, base_url, temperature, max_tokens, timeout, tools=self._mcp_tool_functions)
    
    def _run_async_safe(self, coro, timeout=120):
        """Safely run an async coroutine from sync context."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we need to run in a thread
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
                finally:
                    new_loop.close()
                    
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout + 5)  # Add buffer for thread overhead
                
        except RuntimeError:
            # No event loop running, we can create one
            return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
    
    def _load_mcp_config(self, config: Dict[str, Any]) -> None:
        """Load MCP server configurations."""
        try:
            self.mcp_manager.load_config_from_dict(config)
            self._mcp_enabled = True
            logger.info(f"Loaded {len(self.mcp_manager.get_server_names())} MCP servers")
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
    
    async def _initialize_mcp(self) -> None:
        """Initialize MCP connections and discover tools."""
        if not self._mcp_enabled:
            return
        
        try:
            await self.mcp_manager.connect_all()
            tools = self.mcp_manager.get_all_tools()
            
            # Create Python function wrappers for gpt-oss compatibility
            self._mcp_tool_functions = self._create_tool_functions(tools)
            
            # Update the tools in the parent class
            self.tools = self._mcp_tool_functions
            
            self._tools_context = self._generate_tools_context(tools)
            logger.info(f"Initialized MCP with {len(tools)} tools")
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
    
    def _create_tool_functions(self, tools) -> List[Callable]:
        """Create Python function wrappers for MCP tools."""
        tool_functions = []
        
        for tool in tools:
            def create_tool_wrapper(tool_ref):
                def sync_tool_function(**kwargs):
                    """Generated sync wrapper function for MCP tool."""
                    try:
                        async def call_tool():
                            if self.mcp_manager is None:
                                return "Error: MCP manager not available"
                            result = await self.mcp_manager.call_tool(tool_ref.name, kwargs)
                            if result['success']:
                                return result['result']
                            else:
                                return f"Error: {result['error']}"
                        
                        return self._run_async_safe(call_tool(), timeout=120)
                            
                    except Exception as e:
                        return f"Tool execution error: {str(e)}"
                
                # Set function name and docstring for ollama
                sync_tool_function.__name__ = tool_ref.name
                sync_tool_function.__doc__ = tool_ref.description or f"MCP tool: {tool_ref.name}"
                
                return sync_tool_function
            
            tool_functions.append(create_tool_wrapper(tool))
        
        return tool_functions
    
    def _generate_tools_context(self, tools) -> str:
        """Generate tools context for the language model."""
        if not tools:
            return ""
        
        # Detect if we're using gpt-oss model for different formatting
        is_gpt_oss = 'gpt-oss' in self.model.lower()
        
        if is_gpt_oss:
            # For gpt-oss models, use explicit tool definitions in system prompt
            # Based on web search findings: gpt-oss requires tools to be explicitly listed
            context = "\n\n# Available Tools\n"
            context += "You have access to browser automation tools. When you need to use a tool:\n"
            context += "1. State your intention clearly\n"
            context += "2. Use the specific tool format: TOOL_USE: tool_name(arguments)\n\n"
            
            essential_tools = [
                ('browser_navigate', 'Navigate to a URL', 'url'),
                ('browser_take_screenshot', 'Take a screenshot of current page', None),
                ('browser_snapshot', 'Get page structure for analysis', None),
                ('browser_click', 'Click on an element', 'element, ref')
            ]
            
            for tool_name, description, args in essential_tools:
                context += f"- {tool_name}: {description}\n"
                if args:
                    context += f"  Usage: TOOL_USE: {tool_name}({args})\n"
                else:
                    context += f"  Usage: TOOL_USE: {tool_name}()\n"
            
            context += "\nExample: TOOL_USE: browser_navigate(url='https://google.com')\n"
            context += "Note: Always explain what you're doing before using tools.\n"
        else:
            # Standard JSON format for other models
            context = "\n\nAvailable Browser Tools:\n"
            context += "You can use browser automation tools. To use a tool, respond with JSON: {'tool_call': {'name': 'tool_name', 'arguments': {...}}}\n\n"
            
            # Only include essential browser tools to reduce context size
            essential_tools = [
                'browser_navigate', 'browser_click', 'browser_type', 
                'browser_take_screenshot', 'browser_snapshot', 'browser_wait_for'
            ]
            
            for tool in tools:
                if tool.name in essential_tools:
                    context += f"- {tool.name}: {tool.description}\n"
                    # Only show required parameters for essential tools
                    if tool.input_schema and 'required' in tool.input_schema:
                        required = tool.input_schema['required']
                        if required:
                            context += f"  Required: {', '.join(required)}\n"
            
            context += "\nExample: {'tool_call': {'name': 'browser_navigate', 'arguments': {'url': 'https://google.com'}}}\n"
        
        return context
    
    def _extract_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from LLM response."""
        try:
            # Detect if we're using gpt-oss model for different parsing
            is_gpt_oss = 'gpt-oss' in self.model.lower()
            
            if is_gpt_oss:
                # For gpt-oss models, look for TOOL_USE format and natural language
                import re
                
                # First look for explicit TOOL_USE format
                tool_use_pattern = r'TOOL_USE:\s*([a-zA-Z_]+)\(([^)]*)\)'
                match = re.search(tool_use_pattern, response, re.IGNORECASE)
                
                if match:
                    tool_name = match.group(1)
                    args_str = match.group(2)
                    
                    # Parse arguments
                    arguments = {}
                    if args_str.strip():
                        # Simple parsing for key=value pairs
                        arg_pattern = r"([a-zA-Z_]+)\s*=\s*['\"]([^'\"]*)['\"]?"
                        arg_matches = re.findall(arg_pattern, args_str)
                        for key, value in arg_matches:
                            arguments[key] = value
                    
                    return {
                        'name': tool_name,
                        'arguments': arguments
                    }
                
                # Fallback: natural language detection based on research findings
                response_lower = response.lower().strip()
                
                # Screenshot detection
                if any(keyword in response_lower for keyword in ['screenshot', 'capture', 'image of', 'snap', 'picture']):
                    return {
                        'name': 'browser_take_screenshot',
                        'arguments': {}
                    }
                
                # Navigation detection  
                if any(keyword in response_lower for keyword in ['navigate', 'go to', 'visit', 'open', 'load']):
                    # Try to extract URL
                    url_patterns = [
                        r'https?://[^\s]+',
                        r'www\.[^\s]+',
                        r'[^\s]+\.[a-z]{2,}(?:/[^\s]*)?'
                    ]
                    
                    for pattern in url_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            url = match.group()
                            if not url.startswith('http'):
                                url = 'https://' + url
                            return {
                                'name': 'browser_navigate',
                                'arguments': {'url': url}
                            }
                
                # Click detection
                if any(keyword in response_lower for keyword in ['click', 'press', 'select', 'tap']):
                    return {
                        'name': 'browser_snapshot',  # Take snapshot first to see what's available
                        'arguments': {}
                    }
                
                return None
            
            else:
                # Standard JSON parsing for other models
                import re
                import ast
                
                # First try to find a complete JSON object containing tool_call
                json_pattern = r'\{[^{}]*?["\']tool_call["\'][^{}]*?\}'
                match = re.search(json_pattern, response, re.DOTALL)
                
                if match:
                    json_str = match.group()
                    try:
                        # Try standard JSON first
                        tool_call_data = json.loads(json_str)
                        if 'tool_call' in tool_call_data:
                            return tool_call_data['tool_call']
                    except json.JSONDecodeError:
                        try:
                            # Try Python literal eval for single quotes
                            tool_call_data = ast.literal_eval(json_str)
                            if 'tool_call' in tool_call_data:
                                return tool_call_data['tool_call']
                        except:
                            pass
                
                # Fallback: look line by line
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('{') and 'tool_call' in line:
                        try:
                            # Try standard JSON first
                            tool_call_data = json.loads(line)
                            if 'tool_call' in tool_call_data:
                                return tool_call_data['tool_call']
                        except json.JSONDecodeError:
                            try:
                                # Try Python literal eval for single quotes
                                tool_call_data = ast.literal_eval(line)
                                if 'tool_call' in tool_call_data:
                                    return tool_call_data['tool_call']
                            except:
                                continue
                
                return None
                        
        except Exception as e:
            logger.debug(f"Error extracting tool call: {e}")
            return None
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call and return the result."""
        tool_name = tool_call.get('name')
        arguments = tool_call.get('arguments', {})
        
        if not tool_name:
            return "Error: Tool call missing 'name' field"
        
        try:
            async def execute_tool():
                if self.mcp_manager is None:
                    return "Error: MCP manager not available"
                result = await self.mcp_manager.call_tool(tool_name, arguments)
                
                if result['success']:
                    return f"Tool '{tool_name}' result: {result['result']}"
                else:
                    return f"Tool '{tool_name}' error: {result['error']}"
            
            return self._run_async_safe(execute_tool(), timeout=120)
        
        except Exception as e:
            return f"Tool execution error: {str(e)}"
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get system prompt enhanced with MCP tools context."""
        # Get the original system prompt
        base_prompt = ""
        if self.messages and self.messages[0].role.value == "system":
            base_prompt = self.messages[0].content
        
        # Add tools context if available
        if self._tools_context:
            return base_prompt + self._tools_context
        
        return base_prompt
    
    def _create_enhanced_messages(self) -> list:
        """Create a copy of messages with enhanced system prompt without modifying original."""
        if not self._tools_context or not self.messages:
            return self.messages
        
        # Create a copy of messages
        enhanced_messages = self.messages.copy()
        
        # Enhance system prompt in the copy
        if enhanced_messages and enhanced_messages[0].role.value == "system":
            from .core import Message, Role
            enhanced_system_content = enhanced_messages[0].content + self._tools_context
            enhanced_messages[0] = Message(Role.SYSTEM, enhanced_system_content)
        
        return enhanced_messages
    
    def generate(self, user_message: Optional[str] = None) -> str:
        """
        Generate a response with MCP tool support.
        
        Args:
            user_message (Optional[str], optional): A user message to add before generating.
            
        Returns:
            str: The generated response, potentially including tool execution results.
        """
        logger.debug(f"MCPChainlet.generate called with message: {user_message}")
        
        # Initialize MCP if not already done
        if self._mcp_enabled and not self._tools_context:
            logger.debug("Initializing MCP...")
            try:
                self._run_async_safe(self._initialize_mcp(), timeout=60)
                logger.debug(f"MCP initialized. Tools context length: {len(self._tools_context)}")
            except Exception as e:
                logger.error(f"Failed to initialize MCP: {e}")
                # Continue without MCP functionality
                self._mcp_enabled = False
        
        try:
            # For gpt-oss models, just use the parent's generate method 
            # since it now handles tool calls properly
            is_gpt_oss = 'gpt-oss' in self.model.lower()
            if is_gpt_oss:
                logger.debug("Using parent generate for gpt-oss model")
                return super().generate(user_message)
            
            # For other models, keep the original MCP logic with text-based tool calling
            logger.debug("Using legacy MCP tool calling logic")
            
            # Add user message if provided
            if user_message:
                self.add_user_message(user_message)
            
            # Create enhanced messages without modifying original
            original_messages = self.messages
            enhanced_messages = self._create_enhanced_messages()
            
            # Temporarily replace messages for generation
            self.messages = enhanced_messages
            
            try:
                # Generate response using parent class
                response = super().generate()
                
                # Restore original messages
                self.messages = original_messages
                
                # Check if response contains a tool call (legacy text-based approach)
                tool_call = self._extract_tool_call(response)
                
                if tool_call:
                    # Execute the tool call using the safe async runner
                    try:
                        tool_result = self._execute_tool_call(tool_call)
                    except Exception as e:
                        tool_result = f"Tool execution error: {str(e)}"
                    
                    # Add tool result and generate follow-up
                    self.add_user_message(f"Tool result: {tool_result}")
                    enhanced_messages = self._create_enhanced_messages()
                    self.messages = enhanced_messages
                    
                    try:
                        follow_up = super().generate()
                        return follow_up
                    finally:
                        self.messages = original_messages
                
                return response
                
            except Exception as e:
                # Restore messages on error
                self.messages = original_messages
                raise e
        
        except Exception as e:
            logger.error(f"Error in MCPChainlet.generate: {e}", exc_info=True)
            # Fallback to basic generation
            return super().generate(user_message)
    
    def generate_stream(self, user_message: Optional[str] = None) -> Iterator[str]:
        """
        Generate a streaming response with MCP tool support.
        
        Args:
            user_message (Optional[str], optional): A user message to add before generating.
            
        Yields:
            str: Chunks of the generated response.
        """
        # Initialize MCP if not already done
        if self._mcp_enabled and not self._tools_context:
            try:
                self._run_async_safe(self._initialize_mcp(), timeout=60)
            except Exception as e:
                logger.error(f"Failed to initialize MCP in streaming: {e}")
        
        try:
            # For gpt-oss models, just use the parent's generate_stream method 
            # since it now handles tool calls properly
            is_gpt_oss = 'gpt-oss' in self.model.lower()
            if is_gpt_oss:
                logger.debug("Using parent generate_stream for gpt-oss model")
                for chunk in super().generate_stream(user_message):
                    yield chunk
                return
            
            # For other models, keep the original MCP logic (simplified for sync)
            original_messages = self.messages
            enhanced_messages = self._create_enhanced_messages()
            
            # Temporarily replace messages for generation
            self.messages = enhanced_messages
            
            # Collect the full response to check for tool calls
            full_response = ""
            
            try:
                # Generate streaming response using parent class
                for chunk in super().generate_stream(user_message):
                    full_response += chunk
                    yield chunk
            finally:
                # Always restore original messages
                self.messages = original_messages
            
            # Check if the complete response contains a tool call (legacy approach)
            tool_call = self._extract_tool_call(full_response)
            if tool_call:
                # Execute the tool call using the safe async runner
                try:
                    tool_result = self._execute_tool_call(tool_call)
                except Exception as e:
                    tool_result = f"Tool execution error: {str(e)}"
                
                # Add tool result and generate follow-up
                self.add_user_message(f"Tool result: {tool_result}")
                
                # Stream the follow-up response with enhanced messages
                enhanced_messages = self._create_enhanced_messages()
                self.messages = enhanced_messages
                
                try:
                    yield "\n\n"  # Separator
                    for chunk in super().generate_stream():
                        yield chunk
                finally:
                    # Restore original messages
                    self.messages = original_messages
        
        except Exception as e:
            logger.error(f"Error in streaming generate: {e}")
            # Ensure original messages are restored on error
            try:
                # Check if original_messages exists in the local scope before trying to use it
                if 'original_messages' in locals() and hasattr(self, 'messages'):
                    self.messages = locals()['original_messages']
                elif hasattr(self, 'messages') and self.messages:
                    # Fallback: keep current messages as they are
                    pass
            except Exception as restore_error:
                logger.debug(f"Error restoring messages: {restore_error}")
            raise
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available MCP tools."""
        if self.mcp_manager is None:
            return []
        return [tool.to_dict() for tool in self.mcp_manager.get_all_tools()]
    
    def is_mcp_enabled(self) -> bool:
        """Check if MCP is enabled for this chainlet."""
        return self._mcp_enabled
    
    def _safely_modify_messages(self, modification_func):
        """Safely modify messages with automatic restoration."""
        original_messages = self.messages.copy() if self.messages else []
        try:
            modification_func()
            return True
        except Exception as e:
            logger.error(f"Error modifying messages: {e}")
            self.messages = original_messages
            return False
    
    def refresh_tools(self) -> None:
        """Refresh tool discovery from all servers."""
        if self._mcp_enabled:
            try:
                self._run_async_safe(self._initialize_mcp(), timeout=60)
            except Exception as e:
                logger.error(f"Failed to refresh MCP tools: {e}")