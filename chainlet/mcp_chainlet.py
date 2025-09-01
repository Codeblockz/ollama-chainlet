"""
MCP-Enhanced Chainlet Implementation

This module provides an MCP-enabled chainlet that extends OllamaChainlet
with Model Context Protocol capabilities for tool and resource access.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, Iterator, List

from .ollama import OllamaChainlet
from .mcp import MCPManager, MCPServerConfig


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
        mcp_config: Optional[Dict[str, Any]] = None
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
        # Initialize the base chainlet
        super().__init__(model, system_prompt, base_url, temperature, max_tokens)
        
        # Initialize MCP manager
        self.mcp_manager = MCPManager()
        self._mcp_enabled = False
        self._tools_context = ""
        
        # Load MCP configuration if provided
        if mcp_config:
            self._load_mcp_config(mcp_config)
    
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
            self._tools_context = self._generate_tools_context(tools)
            logger.info(f"Initialized MCP with {len(tools)} tools")
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
    
    def _generate_tools_context(self, tools) -> str:
        """Generate tools context for the language model."""
        if not tools:
            return ""
        
        # Simplified, concise tools context
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
            # Look for JSON objects in the response - handle multi-line JSON and thinking tags
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
                        
        except Exception as e:
            logger.debug(f"Error extracting tool call: {e}")
        
        return None
    
    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call and return the result."""
        tool_name = tool_call.get('name')
        arguments = tool_call.get('arguments', {})
        
        if not tool_name:
            return "Error: Tool call missing 'name' field"
        
        try:
            result = await self.mcp_manager.call_tool(tool_name, arguments)
            
            if result['success']:
                return f"Tool '{tool_name}' result: {result['result']}"
            else:
                return f"Tool '{tool_name}' error: {result['error']}"
        
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
    
    async def generate(self, user_message: Optional[str] = None) -> str:
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
            await self._initialize_mcp()
            logger.debug(f"MCP initialized. Tools context length: {len(self._tools_context)}")
        
        # Temporarily update system prompt with tools context
        original_system_prompt = None
        if self._tools_context and self.messages and self.messages[0].role.value == "system":
            original_system_prompt = self.messages[0].content
            enhanced_prompt = self._get_enhanced_system_prompt()
            logger.debug(f"Updating system prompt. Original length: {len(original_system_prompt)}, Enhanced length: {len(enhanced_prompt)}")
            self.messages[0].content = enhanced_prompt
        
        try:
            logger.debug("Calling parent generate method...")
            logger.debug(f"Current message count: {len(self.messages)}")
            
            # Add user message if provided
            if user_message:
                self.add_user_message(user_message)
                logger.debug(f"Added user message, new count: {len(self.messages)}")
            
            # Generate initial response using parent class (don't pass user_message to avoid duplication)
            response = await asyncio.to_thread(super().generate)
            logger.debug(f"Parent generate returned: {repr(response)} (length: {len(response)})")
            
            # Check if response contains a tool call
            tool_call = self._extract_tool_call(response)
            logger.debug(f"Extracted tool call: {tool_call}")
            
            if tool_call:
                logger.debug(f"Executing tool call: {tool_call}")
                # Execute the tool call
                tool_result = await self._execute_tool_call(tool_call)
                logger.debug(f"Tool result: {tool_result}")
                
                # Add tool result to conversation and generate follow-up
                self.add_user_message(f"Tool result: {tool_result}")
                follow_up = await asyncio.to_thread(super().generate)
                logger.debug(f"Follow-up response: {repr(follow_up)}")
                
                return follow_up
            
            logger.debug("No tool call found, returning original response")
            return response
        
        except Exception as e:
            logger.error(f"Error in MCPChainlet.generate: {e}", exc_info=True)
            # Fallback to basic generation without MCP
            logger.debug("Falling back to basic generation")
            if original_system_prompt and self.messages and self.messages[0].role.value == "system":
                self.messages[0].content = original_system_prompt
            
            # Ensure user message is added for fallback
            if user_message:
                # Check if we already added it
                if not self.messages or self.messages[-1].content != user_message:
                    self.add_user_message(user_message)
            
            return await asyncio.to_thread(super().generate)
        
        finally:
            # Restore original system prompt
            if original_system_prompt and self.messages and self.messages[0].role.value == "system":
                logger.debug("Restoring original system prompt")
                self.messages[0].content = original_system_prompt
    
    async def generate_stream(self, user_message: Optional[str] = None) -> Iterator[str]:
        """
        Generate a streaming response with MCP tool support.
        
        Args:
            user_message (Optional[str], optional): A user message to add before generating.
            
        Yields:
            str: Chunks of the generated response.
        """
        # Initialize MCP if not already done
        if self._mcp_enabled and not self._tools_context:
            await self._initialize_mcp()
        
        # Temporarily update system prompt with tools context
        original_system_prompt = None
        if self._tools_context and self.messages and self.messages[0].role.value == "system":
            original_system_prompt = self.messages[0].content
            enhanced_prompt = self._get_enhanced_system_prompt()
            self.messages[0].content = enhanced_prompt
        
        try:
            # Collect the full response to check for tool calls
            full_response = ""
            
            # Generate streaming response using parent class
            for chunk in super().generate_stream(user_message):
                full_response += chunk
                yield chunk
            
            # Check if the complete response contains a tool call
            tool_call = self._extract_tool_call(full_response)
            if tool_call:
                # Execute the tool call
                tool_result = await self._execute_tool_call(tool_call)
                
                # Add tool result and generate follow-up
                self.add_user_message(f"Tool result: {tool_result}")
                
                # Stream the follow-up response
                yield "\n\n"  # Separator
                for chunk in super().generate_stream():
                    yield chunk
        
        finally:
            # Restore original system prompt
            if original_system_prompt and self.messages and self.messages[0].role.value == "system":
                self.messages[0].content = original_system_prompt
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available MCP tools."""
        return [tool.to_dict() for tool in self.mcp_manager.get_all_tools()]
    
    def is_mcp_enabled(self) -> bool:
        """Check if MCP is enabled for this chainlet."""
        return self._mcp_enabled
    
    async def refresh_tools(self) -> None:
        """Refresh tool discovery from all servers."""
        if self._mcp_enabled:
            await self._initialize_mcp()