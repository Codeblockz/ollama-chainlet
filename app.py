"""
Ollama Chainlet Web Application

This module provides a Flask web application for interacting with Ollama models
through a chainlet interface.
"""

import os
import json
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

from chainlet import OllamaChainlet

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Global variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "You are a helpful AI assistant. Respond concisely and accurately to the user's questions."
)

# Store active chainlets
chainlets: Dict[str, OllamaChainlet] = {}


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
        
        # Get or create chainlet
        if conversation_id not in chainlets:
            chainlets[conversation_id] = OllamaChainlet(
                model=model_name,
                system_prompt=system_prompt,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature
            )
        
        # Generate response
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
        
        # Get or create chainlet
        if conversation_id not in chainlets:
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
                
                # Stream the response chunks
                first_chunk = True
                for chunk in chainlets[conversation_id].generate_stream(message):
                    if not first_chunk:
                        yield ','
                    else:
                        first_chunk = False
                    
                    # Yield each chunk as a JSON string
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


if __name__ == "__main__":
    app.run(debug=True)
