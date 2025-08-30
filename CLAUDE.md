# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
- **Initial setup**: `./setup.sh` - Creates virtual environment and installs dependencies
- **Activate environment**: `source venv/bin/activate`

### Running the Application
- **Start Flask app**: `python app.py` (runs on http://localhost:5000)
- **Test framework**: `python test_chainlet.py` - Tests core functionality with mock data

### Dependencies
- Core dependencies in `requirements.txt`: Flask==2.3.3, requests==2.31.0, python-dotenv==1.0.0

## Architecture Overview

### Core Components
- **Flask Web Application** (`app.py`): Main web server with REST API endpoints for chat interactions
- **Chainlet Framework** (`chainlet/`): Modular conversation management system
  - `core.py`: Base classes for Message, Role enum, and abstract Chainlet
  - `ollama.py`: OllamaChainlet implementation for Ollama API integration

### Key Design Patterns
- **Conversation Management**: Each conversation has a unique ID and maintains message history
- **Streaming Support**: Both regular and streaming chat responses via `/api/chat` and `/api/chat/stream`
- **Model Abstraction**: Chainlet base class allows for different LLM integrations

### API Endpoints
- `GET /api/models` - List available Ollama models
- `POST /api/chat` - Send message and get complete response
- `POST /api/chat/stream` - Send message and stream response chunks
- `DELETE /api/conversations/<id>` - Clear conversation history

### Environment Configuration
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `DEFAULT_MODEL`: Default model name (default: llama3)
- `DEFAULT_SYSTEM_PROMPT`: Default system prompt for conversations

### Data Flow
1. Frontend sends chat request to Flask API
2. OllamaChainlet manages conversation state and calls Ollama API
3. Responses are returned either complete or streamed
4. Conversation history is maintained in memory per conversation ID