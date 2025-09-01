"""
Ollama Chainlet Web Application

This module provides a Flask web application for interacting with Ollama models
through a chainlet interface.
"""

import os
import json
import time
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

from chainlet import OllamaChainlet
from chainlet.mcp_chainlet import MCPChainlet

# Load environment variables
load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:32b")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "You are a helpful AI assistant. Respond concisely and accurately to the user's questions."
)
MCP_CONFIG_PATH = os.getenv("MCP_CONFIG_PATH", "mcp_config/servers.json")

# Store active chainlets
chainlets: Dict[str, OllamaChainlet] = {}

# Global thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

def run_async_in_mcp_loop(coro):
    """Run a coroutine using asyncio.run in a dedicated thread."""
    import concurrent.futures
    import functools
    
    def run_coro():
        """Run the coroutine in a fresh event loop."""
        try:
            # Set event loop policy to handle potential thread issues
            if hasattr(asyncio, 'set_event_loop_policy'):
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            return asyncio.run(coro)
        except Exception as e:
            logger.error(f"Async execution error: {e}")
            raise e
    
    # Use ThreadPoolExecutor to run asyncio.run in a separate thread
    # This avoids all event loop conflicts with Flask threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_coro)
        try:
            result = future.result(timeout=90)  # Increased to 90 seconds for browser operations
            return result
        except concurrent.futures.TimeoutError:
            logger.error("MCP operation timed out")
            raise Exception("MCP operation timed out after 90 seconds")
        except Exception as e:
            logger.error(f"MCP operation failed: {e}")
            raise Exception(f"MCP operation failed: {str(e)}")

# Load MCP configuration
def load_mcp_config() -> Optional[Dict[str, Any]]:
    """Load MCP configuration from file if it exists."""
    try:
        if os.path.exists(MCP_CONFIG_PATH):
            with open(MCP_CONFIG_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load MCP config from {MCP_CONFIG_PATH}: {e}")
    return None

MCP_CONFIG = load_mcp_config()


def should_use_mcp(message: str) -> bool:
    """Determine if MCP should be used based on message content."""
    # Only use MCP if explicitly configured and message suggests tool usage
    if not MCP_CONFIG:
        return False
    
    # Simple heuristics for tool usage (can be enhanced later)
    tool_keywords = [
        'search', 'find', 'lookup', 'browse', 'web', 'url',
        'file', 'read', 'write', 'execute', 'run', 'code',
        'api', 'request', 'call', 'fetch', 'get', 'post'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in tool_keywords)


async def async_stream_to_sync(async_gen):
    """Convert async generator to sync list for Flask compatibility."""
    chunks = []
    async for chunk in async_gen:
        chunks.append(chunk)
    return chunks


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    """Get available models from Ollama."""
    try:
        models = OllamaChainlet.list_models(OLLAMA_BASE_URL)
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a chat message and return the response."""
    try:
        data = request.json
        
        # Extract parameters
        conversation_id = data.get("conversation_id", "default")
        model_name = data.get("model", DEFAULT_MODEL)
        message = data.get("message", "")
        system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        temperature = float(data.get("temperature", 0.7))
        
        # Get or create chainlet (use MCP only if message suggests tool usage)
        use_mcp = should_use_mcp(message)
        
        if conversation_id not in chainlets:
            if use_mcp:
                chainlets[conversation_id] = MCPChainlet(
                    model=model_name,
                    system_prompt=system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=temperature,
                    mcp_config=MCP_CONFIG
                )
            else:
                chainlets[conversation_id] = OllamaChainlet(
                    model=model_name,
                    system_prompt=system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=temperature
                )
        
        # Generate response - upgrade to MCP if needed, or use existing chainlet
        current_chainlet = chainlets[conversation_id]
        
        if use_mcp and not isinstance(current_chainlet, MCPChainlet):
            # Upgrade to MCP chainlet for this request
            mcp_chainlet = MCPChainlet(
                model=model_name,
                system_prompt=system_prompt,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
                mcp_config=MCP_CONFIG
            )
            # Copy message history
            mcp_chainlet.messages = current_chainlet.messages.copy()
            current_chainlet = mcp_chainlet
            chainlets[conversation_id] = mcp_chainlet
        
        # Generate response
        if isinstance(current_chainlet, MCPChainlet):
            try:
                response = run_async_in_mcp_loop(current_chainlet.generate(message))
            except Exception as e:
                # Fallback to regular chainlet on MCP failure
                print(f"MCP generation failed, falling back to regular chainlet: {e}")
                fallback_chainlet = OllamaChainlet(
                    model=model_name,
                    system_prompt=system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=temperature
                )
                # Copy message history
                fallback_chainlet.messages = current_chainlet.messages.copy()
                response = fallback_chainlet.generate(message)
                chainlets[conversation_id] = fallback_chainlet  # Replace with fallback
        else:
            response = current_chainlet.generate(message)
        
        # Get conversation history
        history = chainlets[conversation_id].get_messages_as_dicts()
        
        return jsonify({
            "response": response,
            "conversation_id": conversation_id,
            "history": history
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Process a chat message and stream the response."""
    try:
        data = request.json
        
        # Extract parameters
        conversation_id = data.get("conversation_id", "default")
        model_name = data.get("model", DEFAULT_MODEL)
        message = data.get("message", "")
        system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        temperature = float(data.get("temperature", 0.7))
        
        # Get or create chainlet (use MCP only if message suggests tool usage)
        use_mcp = should_use_mcp(message)
        
        if conversation_id not in chainlets:
            if use_mcp:
                chainlets[conversation_id] = MCPChainlet(
                    model=model_name,
                    system_prompt=system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=temperature,
                    mcp_config=MCP_CONFIG
                )
            else:
                chainlets[conversation_id] = OllamaChainlet(
                    model=model_name,
                    system_prompt=system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=temperature
                )
        
        # Upgrade to MCP if needed
        current_chainlet = chainlets[conversation_id]
        if use_mcp and not isinstance(current_chainlet, MCPChainlet):
            # Upgrade to MCP chainlet for this request
            mcp_chainlet = MCPChainlet(
                model=model_name,
                system_prompt=system_prompt,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
                mcp_config=MCP_CONFIG
            )
            # Copy message history
            mcp_chainlet.messages = current_chainlet.messages.copy()
            current_chainlet = mcp_chainlet
            chainlets[conversation_id] = mcp_chainlet
        
        def generate():
            """Generate streaming response."""
            try:
                # Stream the response chunks (async if MCP-enabled)
                if isinstance(current_chainlet, MCPChainlet):
                    try:
                        # Handle async streaming for MCP chainlet using dedicated loop
                        async def async_stream():
                            async for chunk in chainlets[conversation_id].generate_stream(message):
                                yield chunk
                        
                        # Use run_async_in_mcp_loop for MCP streaming
                        def sync_stream_wrapper():
                            """Convert async stream to sync using run_async_in_mcp_loop"""
                            try:
                                # Get the async generator
                                async def get_chunks():
                                    chunks = []
                                    async for chunk in current_chainlet.generate_stream(message):
                                        chunks.append(chunk)
                                    return chunks
                                
                                # Run async stream collection in dedicated thread
                                chunks = run_async_in_mcp_loop(get_chunks())
                                
                                # Yield chunks synchronously
                                for chunk in chunks:
                                    chunk_data = json.dumps({"content": chunk})
                                    yield f"data: {chunk_data}\n\n"
                                    
                            except Exception as e:
                                error_data = json.dumps({"error": str(e)})
                                yield f"data: {error_data}\n\n"
                        
                        # Stream the collected chunks
                        for chunk_response in sync_stream_wrapper():
                            yield chunk_response
                                
                    except Exception as e:
                        # Fallback to regular streaming on MCP failure
                        print(f"MCP streaming failed, falling back: {e}")
                        fallback_chainlet = OllamaChainlet(
                            model=model_name,
                            system_prompt=system_prompt,
                            base_url=OLLAMA_BASE_URL,
                            temperature=temperature
                        )
                        fallback_chainlet.messages = current_chainlet.messages.copy()
                        for chunk in fallback_chainlet.generate_stream(message):
                            chunk_data = json.dumps({"content": chunk})
                            yield f"data: {chunk_data}\n\n"
                        chainlets[conversation_id] = fallback_chainlet
                else:
                    # Handle sync streaming for regular chainlet
                    for chunk in current_chainlet.generate_stream(message):
                        chunk_data = json.dumps({"content": chunk})
                        yield f"data: {chunk_data}\n\n"
                
                # Send done signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                # If an error occurs during streaming, send an error message
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"
        
        # Return a streaming response as Server-Sent Events
        return Response(stream_with_context(generate()), content_type='text/event-stream')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def clear_conversation(conversation_id):
    """Clear a conversation history."""
    try:
        if conversation_id in chainlets:
            chainlets[conversation_id].clear_messages()
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "not_found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mcp/tools", methods=["GET"])
def get_mcp_tools():
    """Get available MCP tools from all configured servers."""
    try:
        if not MCP_CONFIG:
            return jsonify({"tools": [], "mcp_enabled": False})
        
        # Create a temporary MCP chainlet to discover tools
        temp_chainlet = MCPChainlet(
            model=DEFAULT_MODEL,
            mcp_config=MCP_CONFIG
        )
        
        # Run async tool discovery in sync context
        try:
            run_async_in_mcp_loop(temp_chainlet.refresh_tools())
            tools = temp_chainlet.get_available_tools()
        except Exception as e:
            print(f"Failed to refresh MCP tools: {e}")
            tools = []
        
        return jsonify({"tools": tools, "mcp_enabled": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mcp/status", methods=["GET"])
def get_mcp_status():
    """Get MCP configuration status."""
    try:
        status = {
            "mcp_enabled": MCP_CONFIG is not None,
            "config_path": MCP_CONFIG_PATH,
            "config_exists": os.path.exists(MCP_CONFIG_PATH),
            "servers_configured": 0
        }
        
        if MCP_CONFIG:
            servers = MCP_CONFIG.get("mcpServers", {})
            status["servers_configured"] = len(servers)
            status["server_names"] = list(servers.keys())
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Browser session storage
browser_sessions: Dict[str, Dict[str, Any]] = {}


@app.route("/api/browser/navigate", methods=["POST"])
def browser_navigate():
    """Navigate browser to a URL."""
    try:
        data = request.json
        conversation_id = data.get("conversation_id", "default")
        url = data.get("url", "")
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        # Get or create MCP chainlet for browser operations
        if conversation_id not in chainlets:
            chainlets[conversation_id] = MCPChainlet(
                model=DEFAULT_MODEL,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                base_url=OLLAMA_BASE_URL,
                mcp_config=MCP_CONFIG
            )
        
        # Execute browser navigation
        result = run_async_in_mcp_loop(
            chainlets[conversation_id].mcp_manager.call_tool(
                "mcp__playwright__browser_navigate",
                {"url": url}
            )
        )
        
        # Store browser session info
        if conversation_id not in browser_sessions:
            browser_sessions[conversation_id] = {}
        
        browser_sessions[conversation_id].update({
            "current_url": url,
            "last_action": "navigate",
            "timestamp": time.time()
        })
        
        return jsonify({
            "success": result.get("success", False),
            "result": result.get("result", ""),
            "current_url": url
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browser/screenshot", methods=["POST"])
def browser_screenshot():
    """Take a screenshot of the current browser page."""
    try:
        data = request.json
        conversation_id = data.get("conversation_id", "default")
        
        # Get existing chainlet
        if conversation_id not in chainlets or not isinstance(chainlets[conversation_id], MCPChainlet):
            return jsonify({"error": "Browser session not found"}), 404
        
        # Take screenshot
        result = run_async_in_mcp_loop(
            chainlets[conversation_id].mcp_manager.call_tool(
                "mcp__playwright__browser_take_screenshot",
                {"type": "png"}
            )
        )
        
        return jsonify({
            "success": result.get("success", False),
            "result": result.get("result", ""),
            "screenshot_data": result.get("result", "") if result.get("success") else None
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browser/snapshot", methods=["POST"])
def browser_snapshot():
    """Get accessibility snapshot of current browser page."""
    try:
        data = request.json
        conversation_id = data.get("conversation_id", "default")
        
        # Get existing chainlet
        if conversation_id not in chainlets or not isinstance(chainlets[conversation_id], MCPChainlet):
            return jsonify({"error": "Browser session not found"}), 404
        
        # Get page snapshot
        result = run_async_in_mcp_loop(
            chainlets[conversation_id].mcp_manager.call_tool(
                "mcp__playwright__browser_snapshot",
                {}
            )
        )
        
        return jsonify({
            "success": result.get("success", False),
            "result": result.get("result", ""),
            "snapshot_data": result.get("result", "") if result.get("success") else None
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browser/action", methods=["POST"])
def browser_action():
    """Execute a browser action (click, type, etc.)."""
    try:
        data = request.json
        conversation_id = data.get("conversation_id", "default")
        action = data.get("action", "")
        params = data.get("params", {})
        
        if not action:
            return jsonify({"error": "Action is required"}), 400
        
        # Get existing chainlet
        if conversation_id not in chainlets or not isinstance(chainlets[conversation_id], MCPChainlet):
            return jsonify({"error": "Browser session not found"}), 404
        
        # Map action to MCP tool name
        tool_mapping = {
            "click": "mcp__playwright__browser_click",
            "type": "mcp__playwright__browser_type",
            "wait": "mcp__playwright__browser_wait_for",
            "hover": "mcp__playwright__browser_hover",
            "select": "mcp__playwright__browser_select_option",
            "back": "mcp__playwright__browser_navigate_back",
            "evaluate": "mcp__playwright__browser_evaluate"
        }
        
        if action not in tool_mapping:
            return jsonify({"error": f"Unknown action: {action}"}), 400
        
        # Execute browser action
        result = run_async_in_mcp_loop(
            chainlets[conversation_id].mcp_manager.call_tool(
                tool_mapping[action],
                params
            )
        )
        
        # Update browser session
        if conversation_id in browser_sessions:
            browser_sessions[conversation_id].update({
                "last_action": action,
                "timestamp": time.time()
            })
        
        return jsonify({
            "success": result.get("success", False),
            "result": result.get("result", ""),
            "action": action
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browser/status", methods=["GET"])
def browser_status():
    """Get current browser session status."""
    try:
        conversation_id = request.args.get("conversation_id", "default")
        
        if conversation_id not in browser_sessions:
            return jsonify({
                "active": False,
                "current_url": None,
                "last_action": None
            })
        
        session = browser_sessions[conversation_id]
        return jsonify({
            "active": True,
            "current_url": session.get("current_url"),
            "last_action": session.get("last_action"),
            "timestamp": session.get("timestamp")
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)
