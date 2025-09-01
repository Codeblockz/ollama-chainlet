"""
Ollama Chainlet Implementation

This module provides a chainlet implementation for interacting with Ollama models.
"""

import ollama
from typing import List, Dict, Any, Optional, Iterator, Callable

from .core import Chainlet, Message, Role


class OllamaChainlet(Chainlet):
    """
    A chainlet implementation for interacting with Ollama models.
    
    Attributes:
        base_url (str): The base URL for the Ollama API.
        model (str): The name of the Ollama model to use.
        temperature (float): The temperature parameter for generation.
        max_tokens (int): The maximum number of tokens to generate.
    """
    
    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = None,
        tools: Optional[List[Callable]] = None
    ):
        """
        Initialize a new OllamaChainlet.
        
        Args:
            model (str): The name of the Ollama model to use.
            system_prompt (str, optional): The system prompt for the conversation.
            base_url (str, optional): The base URL for the Ollama API.
            temperature (float, optional): The temperature parameter for generation.
            max_tokens (int, optional): The maximum number of tokens to generate.
            timeout (int, optional): Timeout for API calls in seconds.
            tools (List[Callable], optional): List of tool functions for function calling.
        """
        super().__init__(system_prompt)
        
        # Set timeout based on model size (large models need more time)
        if timeout is None:
            if any(size in model.lower() for size in ['20b', '32b', '70b', '180b']):
                self.timeout = 600  # 10 minutes for very large models
            elif any(size in model.lower() for size in ['7b', '13b', '14b']):
                self.timeout = 300  # 5 minutes for large models  
            else:
                self.timeout = 120  # 2 minutes for smaller models
        else:
            self.timeout = timeout
        
        self.client = ollama.Client(host=base_url, timeout=self.timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
    
    def generate(self, user_message: Optional[str] = None) -> str:
        """
        Generate a response from the Ollama model.
        
        Args:
            user_message (Optional[str], optional): A user message to add before generating.
            
        Returns:
            str: The generated response.
            
        Raises:
            Exception: If there is an error communicating with the Ollama API.
        """
        if user_message:
            self.add_user_message(user_message)
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Prepare chat parameters
                chat_params = {
                    "model": self.model,
                    "messages": self.get_messages_as_dicts(),
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
                
                # Add tools if available (for models that support function calling)
                if self.tools:
                    chat_params["tools"] = self.tools
                
                response = self.client.chat(**chat_params)
                
                # Handle tool calls if present
                if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
                    # Add the assistant's message (which may contain tool calls)
                    if response.message.content:
                        self.add_assistant_message(response.message.content)
                    
                    # Process tool calls and continue conversation
                    full_response = response.message.content or ""
                    
                    for tool_call in response.message.tool_calls:
                        # Execute the tool call
                        tool_result = self._execute_tool_call(tool_call)
                        
                        # Add tool result message
                        tool_message = {
                            "role": "tool",
                            "content": tool_result,
                            "tool_name": tool_call.function.name
                        }
                        self.messages.append(Message(Role.TOOL, tool_result))
                    
                    # Generate follow-up response with tool results
                    follow_up_response = self.client.chat(**chat_params)
                    follow_up_content = follow_up_response['message']['content']
                    self.add_assistant_message(follow_up_content)
                    
                    return full_response + "\n\n" + follow_up_content
                else:
                    # No tool calls, handle normally
                    assistant_message = response['message']['content']
                    self.add_assistant_message(assistant_message)
                    
                    return assistant_message
                
            except ollama.ResponseError as e:
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"Ollama error on attempt {attempt + 1}, retrying in {retry_delay} seconds: {e}")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    
                error_message = f"Error communicating with Ollama API after {max_retries} attempts: {str(e)}"
                raise Exception(error_message)
            
            except Exception as e:
                if "timeout" in str(e).lower() or "connection" in str(e).lower() or "overload" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"Connection error on attempt {attempt + 1}, retrying in {retry_delay} seconds: {e}")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                        
                raise Exception(f"Unexpected error after {max_retries} attempts: {str(e)}")
    
    def _execute_tool_call(self, tool_call) -> str:
        """
        Execute a tool call by finding the matching tool function and calling it.
        
        Args:
            tool_call: The tool call object from ollama response.
            
        Returns:
            str: The result of the tool call.
        """
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments or {}
        
        # Find the matching tool function
        matching_tool = None
        for tool in self.tools:
            if hasattr(tool, '__name__') and tool.__name__ == tool_name:
                matching_tool = tool
                break
        
        if not matching_tool:
            return f"Error: Tool '{tool_name}' not found in available tools"
        
        try:
            # Call the tool function with arguments
            result = matching_tool(**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    def generate_stream(self, user_message: Optional[str] = None) -> Iterator[str]:
        """
        Generate a streaming response from the Ollama model.
        
        Args:
            user_message (Optional[str], optional): A user message to add before generating.
            
        Yields:
            str: Chunks of the generated response as they become available.
            
        Raises:
            Exception: If there is an error communicating with the Ollama API.
        """
        if user_message:
            self.add_user_message(user_message)
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Prepare chat parameters
                chat_params = {
                    "model": self.model,
                    "messages": self.get_messages_as_dicts(),
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    },
                    "stream": True
                }
                
                # Add tools if available (for models that support function calling)
                if self.tools:
                    chat_params["tools"] = self.tools
                
                full_response = ""
                stream = self.client.chat(**chat_params)
                
                # Collect all chunks and check for tool calls
                last_chunk = None
                for chunk in stream:
                    content = chunk['message']['content']
                    full_response += content
                    yield content
                    last_chunk = chunk
                
                # Check if the final chunk contains tool calls
                tool_calls_present = False
                if last_chunk and hasattr(last_chunk.message, 'tool_calls') and last_chunk.message.tool_calls:
                    tool_calls_present = True
                    
                    # Process tool calls
                    for tool_call in last_chunk.message.tool_calls:
                        tool_result = self._execute_tool_call(tool_call)
                        self.messages.append(Message(Role.TOOL, tool_result))
                    
                    # Generate follow-up response with tool results
                    follow_up_params = chat_params.copy()
                    follow_up_params["stream"] = True
                    follow_up_stream = self.client.chat(**follow_up_params)
                    
                    yield "\n\n"  # Separator between initial response and follow-up
                    follow_up_response = ""
                    for chunk in follow_up_stream:
                        content = chunk['message']['content']
                        follow_up_response += content
                        yield content
                    
                    # Add both messages to conversation history
                    self.add_assistant_message(full_response)
                    self.add_assistant_message(follow_up_response)
                else:
                    # No tool calls, add the complete response to conversation history
                    self.add_assistant_message(full_response)
                
                break  # Success, exit retry loop
                
            except ollama.ResponseError as e:
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"Ollama streaming error on attempt {attempt + 1}, retrying in {retry_delay} seconds: {e}")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                        
                error_message = f"Error streaming from Ollama API after {max_retries} attempts: {str(e)}"
                raise Exception(error_message)
            
            except Exception as e:
                if "timeout" in str(e).lower() or "connection" in str(e).lower() or "overload" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"Streaming connection error on attempt {attempt + 1}, retrying in {retry_delay} seconds: {e}")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                        
                raise Exception(f"Unexpected streaming error after {max_retries} attempts: {str(e)}")
    
    @classmethod
    def list_models(cls, base_url: str = "http://localhost:11434") -> List[str]:
        """
        List available models from the Ollama API.
        
        Args:
            base_url (str, optional): The base URL for the Ollama API.
            
        Returns:
            List[str]: A list of available model names.
            
        Raises:
            Exception: If there is an error communicating with the Ollama API.
        """
        try:
            client = ollama.Client(host=base_url, timeout=30)  # 30 second timeout for listing models
            result = client.list()
            models = [model.model for model in result.models]
            
            return models
            
        except ollama.ResponseError as e:
            error_message = f"Error listing models from Ollama API: {str(e)}"
            raise Exception(error_message)
        except Exception as e:
            error_message = f"Unexpected error listing models: {str(e)}"
            raise Exception(error_message)
