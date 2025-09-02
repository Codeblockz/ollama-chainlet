"""
Ollama Chainlet Web Application

This module provides a FastAPI web application for interacting with Ollama models
through a chainlet interface.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from chainlet import OllamaChainlet
from chainlet.mcp_chainlet import MCPChainlet

# Load environment variables
load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Ollama Chainlet", description="Chat interface for Ollama models with MCP integration")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

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


# Pydantic models for request/response
class ChatRequest(BaseModel):
    conversation_id: str = "default"
    model: str = DEFAULT_MODEL
    message: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7


class BrowserNavigateRequest(BaseModel):
    conversation_id: str = "default"
    url: str


class BrowserActionRequest(BaseModel):
    conversation_id: str = "default"
    action: str
    params: Dict[str, Any] = {}


class BrowserStatusRequest(BaseModel):
    conversation_id: str = "default"


# Browser session storage
browser_sessions: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/models")
async def get_models():
    """Get available models from Ollama."""
    try:
        models = OllamaChainlet.list_models(OLLAMA_BASE_URL)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process a chat message and return the response."""
    try:
        # Get or create chainlet (use MCP only if message suggests tool usage)
        use_mcp = should_use_mcp(request.message)
        
        if request.conversation_id not in chainlets:
            if use_mcp:
                chainlets[request.conversation_id] = MCPChainlet(
                    model=request.model,
                    system_prompt=request.system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=request.temperature,
                    mcp_config=MCP_CONFIG
                )
            else:
                chainlets[request.conversation_id] = OllamaChainlet(
                    model=request.model,
                    system_prompt=request.system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=request.temperature
                )
        
        # Generate response - upgrade to MCP if needed, or use existing chainlet
        current_chainlet = chainlets[request.conversation_id]
        
        if use_mcp and not isinstance(current_chainlet, MCPChainlet):
            # Upgrade to MCP chainlet for this request
            mcp_chainlet = MCPChainlet(
                model=request.model,
                system_prompt=request.system_prompt,
                base_url=OLLAMA_BASE_URL,
                temperature=request.temperature,
                mcp_config=MCP_CONFIG
            )
            # Copy message history
            mcp_chainlet.messages = current_chainlet.messages.copy()
            current_chainlet = mcp_chainlet
            chainlets[request.conversation_id] = mcp_chainlet
        
        # Generate response
        if isinstance(current_chainlet, MCPChainlet):
            try:
                response = await current_chainlet.generate(request.message)
            except Exception as e:
                # Fallback to regular chainlet on MCP failure
                print(f"MCP generation failed, falling back to regular chainlet: {e}")
                fallback_chainlet = OllamaChainlet(
                    model=request.model,
                    system_prompt=request.system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=request.temperature
                )
                # Copy message history
                fallback_chainlet.messages = current_chainlet.messages.copy()
                response = fallback_chainlet.generate(request.message)
                chainlets[request.conversation_id] = fallback_chainlet  # Replace with fallback
        else:
            response = current_chainlet.generate(request.message)
        
        # Get conversation history
        history = chainlets[request.conversation_id].get_messages_as_dicts()
        
        return {
            "response": response,
            "conversation_id": request.conversation_id,
            "history": history
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Process a chat message and stream the response."""
    try:
        # Get or create chainlet (use MCP only if message suggests tool usage)
        use_mcp = should_use_mcp(request.message)
        
        if request.conversation_id not in chainlets:
            if use_mcp:
                chainlets[request.conversation_id] = MCPChainlet(
                    model=request.model,
                    system_prompt=request.system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=request.temperature,
                    mcp_config=MCP_CONFIG
                )
            else:
                chainlets[request.conversation_id] = OllamaChainlet(
                    model=request.model,
                    system_prompt=request.system_prompt,
                    base_url=OLLAMA_BASE_URL,
                    temperature=request.temperature
                )
        
        # Upgrade to MCP if needed
        current_chainlet = chainlets[request.conversation_id]
        if use_mcp and not isinstance(current_chainlet, MCPChainlet):
            # Upgrade to MCP chainlet for this request
            mcp_chainlet = MCPChainlet(
                model=request.model,
                system_prompt=request.system_prompt,
                base_url=OLLAMA_BASE_URL,
                temperature=request.temperature,
                mcp_config=MCP_CONFIG
            )
            # Copy message history
            mcp_chainlet.messages = current_chainlet.messages.copy()
            current_chainlet = mcp_chainlet
            chainlets[request.conversation_id] = mcp_chainlet
        
        async def generate():
            """Generate streaming response."""
            try:
                # Stream the response chunks (async if MCP-enabled)
                if isinstance(current_chainlet, MCPChainlet):
                    try:
                        # Handle async streaming for MCP chainlet
                        async for chunk in current_chainlet.generate_stream(request.message):
                            chunk_data = json.dumps({"content": chunk})
                            yield f"data: {chunk_data}\n\n"
                                
                    except Exception as e:
                        # Fallback to regular streaming on MCP failure
                        print(f"MCP streaming failed, falling back: {e}")
                        fallback_chainlet = OllamaChainlet(
                            model=request.model,
                            system_prompt=request.system_prompt,
                            base_url=OLLAMA_BASE_URL,
                            temperature=request.temperature
                        )
                        fallback_chainlet.messages = current_chainlet.messages.copy()
                        for chunk in fallback_chainlet.generate_stream(request.message):
                            chunk_data = json.dumps({"content": chunk})
                            yield f"data: {chunk_data}\n\n"
                        chainlets[request.conversation_id] = fallback_chainlet
                else:
                    # Handle sync streaming for regular chainlet
                    for chunk in current_chainlet.generate_stream(request.message):
                        chunk_data = json.dumps({"content": chunk})
                        yield f"data: {chunk_data}\n\n"
                
                # Send done signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                # If an error occurs during streaming, send an error message
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"
        
        # Return a streaming response as Server-Sent Events
        return StreamingResponse(generate(), media_type='text/event-stream')
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear a conversation history."""
    try:
        if conversation_id in chainlets:
            chainlets[conversation_id].clear_messages()
            return {"status": "success"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mcp/tools")
async def get_mcp_tools():
    """Get available MCP tools from all configured servers."""
    try:
        if not MCP_CONFIG:
            return {"tools": [], "mcp_enabled": False, "message": "No MCP configuration found"}
        
        # Create a temporary MCP chainlet to discover tools
        temp_chainlet = MCPChainlet(
            model=DEFAULT_MODEL,
            mcp_config=MCP_CONFIG
        )
        
        # Run async tool discovery
        tools = []
        error_messages = []
        
        try:
            await temp_chainlet.refresh_tools()
            tools = temp_chainlet.get_available_tools()
        except Exception as e:
            error_messages.append(f"Tool discovery failed: {str(e)}")
            logger.error(f"Failed to refresh MCP tools: {e}")
        
        return {
            "tools": tools, 
            "mcp_enabled": True,
            "tool_count": len(tools),
            "errors": error_messages if error_messages else None,
            "message": f"Found {len(tools)} tools" if tools else "No tools discovered"
        }
    
    except Exception as e:
        logger.error(f"MCP tools endpoint error: {e}")
        raise HTTPException(status_code=500, detail={
            "error": str(e), 
            "mcp_enabled": bool(MCP_CONFIG),
            "tools": [],
            "message": "Error retrieving MCP tools"
        })


@app.get("/api/mcp/status")
async def get_mcp_status():
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
        
        return status
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/browser/navigate")
async def browser_navigate(request: BrowserNavigateRequest):
    """Navigate browser to a URL."""
    try:
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Get or create MCP chainlet for browser operations
        if request.conversation_id not in chainlets:
            chainlets[request.conversation_id] = MCPChainlet(
                model=DEFAULT_MODEL,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                base_url=OLLAMA_BASE_URL,
                mcp_config=MCP_CONFIG
            )
        
        # Execute browser navigation
        result = await chainlets[request.conversation_id].mcp_manager.call_tool(
            "mcp__playwright__browser_navigate",
            {"url": request.url}
        )
        
        # Store browser session info
        if request.conversation_id not in browser_sessions:
            browser_sessions[request.conversation_id] = {}
        
        browser_sessions[request.conversation_id].update({
            "current_url": request.url,
            "last_action": "navigate",
            "timestamp": time.time()
        })
        
        return {
            "success": result.get("success", False),
            "result": result.get("result", ""),
            "current_url": request.url
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/browser/screenshot")
async def browser_screenshot(request: BrowserStatusRequest):
    """Take a screenshot of the current browser page."""
    try:
        # Get existing chainlet
        if request.conversation_id not in chainlets or not isinstance(chainlets[request.conversation_id], MCPChainlet):
            raise HTTPException(status_code=404, detail="Browser session not found")
        
        # Take screenshot
        result = await chainlets[request.conversation_id].mcp_manager.call_tool(
            "mcp__playwright__browser_take_screenshot",
            {"type": "png"}
        )
        
        return {
            "success": result.get("success", False),
            "result": result.get("result", ""),
            "screenshot_data": result.get("result", "") if result.get("success") else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/browser/snapshot")
async def browser_snapshot(request: BrowserStatusRequest):
    """Get accessibility snapshot of current browser page."""
    try:
        # Get existing chainlet
        if request.conversation_id not in chainlets or not isinstance(chainlets[request.conversation_id], MCPChainlet):
            raise HTTPException(status_code=404, detail="Browser session not found")
        
        # Get page snapshot
        result = await chainlets[request.conversation_id].mcp_manager.call_tool(
            "mcp__playwright__browser_snapshot",
            {}
        )
        
        return {
            "success": result.get("success", False),
            "result": result.get("result", ""),
            "snapshot_data": result.get("result", "") if result.get("success") else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/browser/action")
async def browser_action(request: BrowserActionRequest):
    """Execute a browser action (click, type, etc.)."""
    try:
        if not request.action:
            raise HTTPException(status_code=400, detail="Action is required")
        
        # Get existing chainlet
        if request.conversation_id not in chainlets or not isinstance(chainlets[request.conversation_id], MCPChainlet):
            raise HTTPException(status_code=404, detail="Browser session not found")
        
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
        
        if request.action not in tool_mapping:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
        
        # Execute browser action
        result = await chainlets[request.conversation_id].mcp_manager.call_tool(
            tool_mapping[request.action],
            request.params
        )
        
        # Update browser session
        if request.conversation_id in browser_sessions:
            browser_sessions[request.conversation_id].update({
                "last_action": request.action,
                "timestamp": time.time()
            })
        
        return {
            "success": result.get("success", False),
            "result": result.get("result", ""),
            "action": request.action
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/browser/status")
async def browser_status(conversation_id: str = "default"):
    """Get current browser session status."""
    try:
        if conversation_id not in browser_sessions:
            return {
                "active": False,
                "current_url": None,
                "last_action": None
            }
        
        session = browser_sessions[conversation_id]
        return {
            "active": True,
            "current_url": session.get("current_url"),
            "last_action": session.get("last_action"),
            "timestamp": session.get("timestamp")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)