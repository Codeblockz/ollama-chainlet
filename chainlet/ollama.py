"""
Ollama Chainlet Implementation

This module provides a chainlet implementation for interacting with Ollama models.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Tuple, Generator, Iterator

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
        max_tokens: int = 2048
    ):
        """
        Initialize a new OllamaChainlet.
        
        Args:
            model (str): The name of the Ollama model to use.
            system_prompt (str, optional): The system prompt for the conversation.
            base_url (str, optional): The base URL for the Ollama API.
            temperature (float, optional): The temperature parameter for generation.
            max_tokens (int, optional): The maximum number of tokens to generate.
        """
        super().__init__(system_prompt)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
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
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": self.get_messages_as_dicts(),
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": False
        }
        
        # Send the request to the Ollama API
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            assistant_message = result.get("message", {}).get("content", "")
            
            # Add the assistant's response to the conversation history
            self.add_assistant_message(assistant_message)
            
            return assistant_message
        
        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with Ollama API: {str(e)}"
            raise Exception(error_message)
    
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
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": self.get_messages_as_dicts(),
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": True
        }
        
        # Send the request to the Ollama API
        try:
            full_response = ""
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        # Parse the JSON response
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                full_response += content
                                yield content
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            continue
            
            # Add the complete response to the conversation history
            self.add_assistant_message(full_response)
            
        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with Ollama API: {str(e)}"
            raise Exception(error_message)
    
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
            response = requests.get(
                f"{base_url.rstrip('/')}/api/tags",
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            models = [model.get("name") for model in result.get("models", [])]
            
            return models
        
        except requests.exceptions.RequestException as e:
            error_message = f"Error listing models from Ollama API: {str(e)}"
            raise Exception(error_message)
