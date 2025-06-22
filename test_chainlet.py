#!/usr/bin/env python3
"""
Test script for the Ollama Chainlet framework.

This script tests the basic functionality of the chainlet framework
without requiring a running Ollama instance.
"""

import sys
from chainlet import Chainlet, Message, Role


class MockChainlet(Chainlet):
    """A mock chainlet for testing."""
    
    def generate(self, user_message=None):
        """Generate a mock response."""
        if user_message:
            self.add_user_message(user_message)
        
        # Get the last user message
        last_message = None
        for message in reversed(self.messages):
            if message.role == Role.USER:
                last_message = message.content
                break
        
        # Generate a mock response
        response = f"Echo: {last_message}" if last_message else "No user message found."
        
        # Add the response to the conversation history
        self.add_assistant_message(response)
        
        return response


def main():
    """Run the test."""
    print("Testing Chainlet Framework...")
    
    # Create a mock chainlet
    chainlet = MockChainlet(system_prompt="This is a test system prompt.")
    
    # Test adding messages
    print("\nTesting message handling...")
    chainlet.add_user_message("Hello, world!")
    
    # Test generating a response
    print("\nTesting response generation...")
    response = chainlet.generate("How are you?")
    print(f"Response: {response}")
    
    # Test conversation history
    print("\nTesting conversation history...")
    history = chainlet.get_messages_as_dicts()
    for i, message in enumerate(history):
        print(f"Message {i+1}: {message['role']} - {message['content']}")
    
    # Test clearing messages
    print("\nTesting clearing messages...")
    chainlet.clear_messages()
    history = chainlet.get_messages_as_dicts()
    print(f"Messages after clearing: {len(history)}")
    
    print("\nAll tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
