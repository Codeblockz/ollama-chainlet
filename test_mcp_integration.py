#!/usr/bin/env python3
"""
Test MCP Integration

This script tests the MCP integration functionality.
"""

import asyncio
import json
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chainlet.mcp_chainlet import MCPChainlet
from chainlet.mcp import MCPManager, MCPServerConfig


async def test_mcp_manager():
    """Test MCPManager functionality."""
    print("Testing MCPManager...")
    
    manager = MCPManager()
    
    # Test adding a server configuration
    config = MCPServerConfig(
        name="test-server",
        transport="stdio",
        command="echo",
        args=["hello"]
    )
    
    manager.add_server(config)
    print(f"Added server: {config.name}")
    print(f"Server names: {manager.get_server_names()}")
    
    # Test configuration loading from dict
    config_dict = {
        "mcpServers": {
            "echo-server": {
                "command": "echo",
                "args": ["test"],
                "env": {}
            }
        }
    }
    
    manager.load_config_from_dict(config_dict)
    print(f"Loaded config, server names: {manager.get_server_names()}")
    
    return True


async def test_mcp_chainlet():
    """Test MCPChainlet functionality."""
    print("\nTesting MCPChainlet...")
    
    # Test configuration
    mcp_config = {
        "mcpServers": {
            "test-server": {
                "command": "echo",
                "args": ["Available tools: none"],
                "env": {}
            }
        }
    }
    
    # Create MCP chainlet
    chainlet = MCPChainlet(
        model="llama3",
        system_prompt="You are a helpful assistant.",
        mcp_config=mcp_config
    )
    
    print(f"MCP enabled: {chainlet.is_mcp_enabled()}")
    print(f"Available tools: {len(chainlet.get_available_tools())}")
    
    return True


def test_configuration_loading():
    """Test configuration file loading."""
    print("\nTesting configuration loading...")
    
    config_path = "mcp_config/servers.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
        print(f"Servers configured: {list(config.get('mcpServers', {}).keys())}")
        return True
    else:
        print(f"Config file not found: {config_path}")
        return False


async def main():
    """Run all tests."""
    print("Starting MCP Integration Tests...\n")
    
    try:
        # Test 1: MCPManager
        await test_mcp_manager()
        
        # Test 2: MCPChainlet
        await test_mcp_chainlet()
        
        # Test 3: Configuration loading
        test_configuration_loading()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)