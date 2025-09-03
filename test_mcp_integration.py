#!/usr/bin/env python3
"""
Test script to verify MCP integration fixes.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_import_handling():
    """Test that import handling works correctly."""
    logger.info("Testing MCP import handling...")
    
    try:
        from chainlet.mcp import MCP_AVAILABLE, MCPNotAvailableError
        logger.info(f"MCP available: {MCP_AVAILABLE}")
        
        if MCP_AVAILABLE:
            from chainlet.mcp import MCPManager, MCPServerConfig
            logger.info("MCP classes imported successfully")
            
            # Test basic instantiation
            try:
                config = MCPServerConfig(
                    name="test",
                    transport="stdio",
                    command="echo",
                    args=["test"]
                )
                config.validate()
                logger.info("MCPServerConfig validation works")
                
                # Test manager creation (this should work even without servers)
                manager = MCPManager()
                logger.info("MCPManager created successfully")
                
            except Exception as e:
                logger.error(f"Error testing MCP classes: {e}")
                return False
        else:
            logger.info("MCP not available, testing error handling...")
            
            try:
                from chainlet.mcp import MCPManager
                manager = MCPManager()  # This should raise MCPNotAvailableError
                logger.error("Expected MCPNotAvailableError was not raised")
                return False
            except MCPNotAvailableError as e:
                logger.info(f"Correct error raised: {e}")
            except Exception as e:
                logger.error(f"Wrong error type: {e}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error in import test: {e}")
        return False

def test_mcp_chainlet():
    """Test MCPChainlet initialization."""
    logger.info("Testing MCPChainlet initialization...")
    
    try:
        from chainlet.mcp_chainlet import MCPChainlet
        from chainlet.mcp import MCP_AVAILABLE
        
        # Test basic initialization without MCP config
        chainlet = MCPChainlet(model="test-model")
        logger.info("MCPChainlet created without MCP config")
        
        # Test with MCP config (should warn if MCP not available)
        mcp_config = {
            "mcpServers": {
                "test": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        
        chainlet_with_config = MCPChainlet(model="test-model", mcp_config=mcp_config)
        logger.info("MCPChainlet created with MCP config")
        
        # Test methods that should work regardless of MCP availability
        tools = chainlet_with_config.get_available_tools()
        logger.info(f"Available tools: {len(tools)}")
        
        enabled = chainlet_with_config.is_mcp_enabled()
        logger.info(f"MCP enabled: {enabled}")
        
        # This should work even if MCP is not available
        chainlet_with_config.refresh_tools()
        logger.info("refresh_tools() completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in MCPChainlet test: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting MCP integration tests...")
    
    tests = [
        test_import_handling,
        test_mcp_chainlet
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                logger.info(f"✓ {test.__name__} passed")
                passed += 1
            else:
                logger.error(f"✗ {test.__name__} failed")
        except Exception as e:
            logger.error(f"✗ {test.__name__} failed with exception: {e}")
    
    logger.info(f"Tests completed: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        logger.info("All tests passed! MCP integration fixes are working correctly.")
        return 0
    else:
        logger.error("Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())