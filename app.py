"""
Ollama Chainlet Web Application

This module provides a Flask web application for interacting with Ollama models
through a chainlet interface.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

from chainlet import OllamaChainlet
from chainlet.mcp_chainlet import MCPChainlet

# Load environment variables
load_dotenv()

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


async def async_stream_to_sync(async_gen):
    """Convert async generator to sync generator for Flask compatibility."""
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
        
        # Get or create chainlet (MCP-enabled if config available)
        if conversation_id not in chainlets:
            if MCP_CONFIG:
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
        
        # Generate response (handle async MCP calls in sync context)
        if isinstance(chainlets[conversation_id], MCPChainlet):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(chainlets[conversation_id].generate(message))
            finally:
                loop.close()
        else:
            response = chainlets[conversation_id].generate(message)
        
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
        
        # Get or create chainlet (MCP-enabled if config available)
        if conversation_id not in chainlets:
            if MCP_CONFIG:
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
        
        def generate():
            """Generate streaming response."""
            try:
                # Start with an empty JSON object
                yield '{"conversation_id": "%s", "response": "", "chunks": [' % conversation_id
                
                # Stream the response chunks (async if MCP-enabled)
                first_chunk = True
                if isinstance(chainlets[conversation_id], MCPChainlet):
                    # Handle async streaming for MCP chainlet
                    async def async_stream():
                        async for chunk in chainlets[conversation_id].generate_stream(message):
                            yield chunk
                    
                    # Run async generator in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        for chunk in loop.run_until_complete(async_stream_to_sync(chainlets[conversation_id].generate_stream(message))):
                            if not first_chunk:
                                yield ','
                            else:
                                first_chunk = False
                            
                            chunk_json = json.dumps({"content": chunk})
                            yield chunk_json
                    finally:
                        loop.close()
                else:
                    # Handle sync streaming for regular chainlet
                    for chunk in chainlets[conversation_id].generate_stream(message):
                        if not first_chunk:
                            yield ','
                        else:
                            first_chunk = False
                        
                        chunk_json = json.dumps({"content": chunk})
                        yield chunk_json
                
                # End the JSON array and object
                yield '], "done": true}'
                
            except Exception as e:
                # If an error occurs during streaming, send an error message
                error_json = json.dumps({"error": str(e)})
                yield '], "error": %s}' % error_json
        
        # Return a streaming response
        return Response(stream_with_context(generate()), content_type='application/json')
    
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(temp_chainlet.refresh_tools())
            tools = temp_chainlet.get_available_tools()
        finally:
            loop.close()
        
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




if __name__ == "__main__":
    app.run(debug=True)
