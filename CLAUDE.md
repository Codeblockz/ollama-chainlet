# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
- **Initial setup**: `./setup.sh` - Creates virtual environment and installs dependencies
- **Activate environment**: `source venv/bin/activate`

### Running the Application
- **Start FastAPI app**: `python app.py` (runs on http://localhost:5000)
- **Alternative**: `uvicorn app:app --host 0.0.0.0 --port 5000 --reload` (for development)

### Dependencies
- Core dependencies in `requirements.txt`: fastapi>=0.115.0, uvicorn[standard]>=0.30.0, jinja2>=3.1.2, python-multipart>=0.0.6, requests>=2.31.0, python-dotenv>=1.0.0, mcp>=1.0.0, pydantic>=2.0.0, ollama, anyio>=4.5.0

## Architecture Overview

### Core Components
- **FastAPI Web Application** (`app.py`): Main web server with async REST API endpoints for chat interactions
- **Chainlet Framework** (`chainlet/`): Modular conversation management system
  - `core.py`: Base classes for Message, Role enum, and abstract Chainlet
  - `ollama.py`: OllamaChainlet implementation for Ollama API integration using official ollama Python client
  - `mcp.py`: MCP (Model Context Protocol) infrastructure for external tool integration
  - `mcp_chainlet.py`: MCP-enhanced chainlet that extends OllamaChainlet with tool capabilities

### Key Design Patterns
- **Conversation Management**: Each conversation has a unique ID and maintains message history
- **Async Architecture**: Native async/await support for all I/O operations
- **Streaming Support**: Both regular and streaming chat responses via `/api/chat` and `/api/chat/stream`
- **Model Abstraction**: Chainlet base class allows for different LLM integrations
- **MCP Integration**: Model Context Protocol support for external tool access and enhanced capabilities
- **Type Safety**: Pydantic models for request/response validation

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
1. Frontend sends chat request to FastAPI endpoint
2. Request validated through Pydantic models
3. OllamaChainlet manages conversation state and calls Ollama API
4. Responses are returned either complete or streamed via async generators
5. Conversation history is maintained in memory per conversation ID

### FastAPI Features
- **Automatic API Documentation**: Available at `/docs` (OpenAPI/Swagger UI)
- **Alternative Docs**: Available at `/redoc` (ReDoc UI) 
- **Request Validation**: Automatic validation via Pydantic models
- **Async Support**: Native async/await throughout the application
- **Type Hints**: Full type annotation support