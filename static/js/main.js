/**
 * Ollama Chainlet Frontend Logic
 */

// DOM Elements
const modelSelect = document.getElementById('model-select');
const systemPrompt = document.getElementById('system-prompt');
const temperatureSlider = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');
const streamingToggle = document.getElementById('streaming-toggle');
const newChatBtn = document.getElementById('new-chat-btn');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const messagesContainer = document.getElementById('messages');
const statusDiv = document.getElementById('status');
const statusText = document.getElementById('status-text');

// State
let conversationId = generateId();
let isProcessing = false;
let currentMessageDiv = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    setupEventListeners();
    adjustTextareaHeight();
});

/**
 * Load available models from the API
 */
async function loadModels() {
    try {
        showStatus('Loading models...', 'info');
        
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        populateModelSelect(data.models);
        hideStatus();
    } catch (error) {
        showStatus(`Error loading models: ${error.message}`, 'danger');
        console.error('Error loading models:', error);
    }
}

/**
 * Populate the model select dropdown
 */
function populateModelSelect(models) {
    modelSelect.innerHTML = '';
    
    if (models.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No models available';
        option.disabled = true;
        option.selected = true;
        modelSelect.appendChild(option);
        return;
    }
    
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    });
    
    // Select the first model by default
    modelSelect.value = models[0];
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Temperature slider
    temperatureSlider.addEventListener('input', () => {
        temperatureValue.textContent = temperatureSlider.value;
    });
    
    // New chat button
    newChatBtn.addEventListener('click', startNewChat);
    
    // Chat form submission
    chatForm.addEventListener('submit', handleSubmit);
    
    // Auto-resize textarea
    userInput.addEventListener('input', adjustTextareaHeight);
    
    // Streaming toggle
    if (streamingToggle) {
        streamingToggle.addEventListener('change', () => {
            localStorage.setItem('streaming-enabled', streamingToggle.checked);
        });
        
        // Load saved preference
        const savedPreference = localStorage.getItem('streaming-enabled');
        if (savedPreference !== null) {
            streamingToggle.checked = savedPreference === 'true';
        }
    }
}

/**
 * Start a new chat
 */
function startNewChat() {
    conversationId = generateId();
    
    // Clear messages except the welcome message
    messagesContainer.innerHTML = `
        <div class="message system">
            <div class="message-content">
                <p>Welcome to Ollama Chainlet! Select a model and start chatting.</p>
            </div>
        </div>
    `;
    
    // Clear the input
    userInput.value = '';
    adjustTextareaHeight();
    
    // Clear the conversation on the server
    fetch(`/api/conversations/${conversationId}`, {
        method: 'DELETE'
    }).catch(error => {
        console.error('Error clearing conversation:', error);
    });
    
    showStatus('Started new chat', 'success', 2000);
}

/**
 * Handle form submission
 */
async function handleSubmit(event) {
    event.preventDefault();
    
    const message = userInput.value.trim();
    if (!message || isProcessing) return;
    
    const selectedModel = modelSelect.value;
    if (!selectedModel) {
        showStatus('Please select a model', 'warning', 3000);
        return;
    }
    
    // Add user message to the UI
    addMessage(message, 'user');
    
    // Clear the input
    userInput.value = '';
    adjustTextareaHeight();
    
    // Show processing indicator
    isProcessing = true;
    addLoadingMessage();
    
    // Check if streaming is enabled
    const isStreamingEnabled = streamingToggle && streamingToggle.checked;
    
    try {
        if (isStreamingEnabled) {
            // Use streaming endpoint
            await handleStreamingResponse(message, selectedModel);
        } else {
            // Use regular endpoint
            await handleRegularResponse(message, selectedModel);
        }
    } catch (error) {
        // Remove loading indicator
        removeLoadingMessage();
        
        // Show error message
        showStatus(`Error: ${error.message}`, 'danger', 5000);
        console.error('Error sending message:', error);
        
        // Add error message to the UI
        addMessage('Error: Unable to generate a response. Please try again.', 'system');
    } finally {
        isProcessing = false;
    }
}

/**
 * Handle regular (non-streaming) response
 */
async function handleRegularResponse(message, selectedModel) {
    // Send the message to the API
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            conversation_id: conversationId,
            model: selectedModel,
            message: message,
            system_prompt: systemPrompt.value,
            temperature: parseFloat(temperatureSlider.value)
        })
    });
    
    const data = await response.json();
    
    // Remove loading indicator
    removeLoadingMessage();
    
    if (data.error) {
        throw new Error(data.error);
    }
    
    // Add assistant message to the UI
    addMessage(data.response, 'assistant');
}

/**
 * Handle streaming response
 */
async function handleStreamingResponse(message, selectedModel) {
    // Remove loading indicator and prepare for streaming
    removeLoadingMessage();
    
    // Create an empty message div for streaming content
    currentMessageDiv = document.createElement('div');
    currentMessageDiv.className = 'message assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<p></p>'; // Start with an empty paragraph
    
    currentMessageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(currentMessageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Prepare the request
    const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            conversation_id: conversationId,
            model: selectedModel,
            message: message,
            system_prompt: systemPrompt.value,
            temperature: parseFloat(temperatureSlider.value)
        })
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Network response was not ok');
    }
    
    // Read the stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    
    while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
            break;
        }
        
        // Decode the chunk and add it to the buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete JSON objects from the buffer
        try {
            // Try to parse the entire buffer as JSON
            const result = JSON.parse(buffer);
            
            // If we got here, the buffer contains a complete JSON object
            if (result.error) {
                throw new Error(result.error);
            }
            
            // Update the UI with all chunks
            if (result.chunks && result.chunks.length > 0) {
                for (const chunk of result.chunks) {
                    updateStreamingMessage(chunk.content);
                    fullResponse += chunk.content;
                }
            }
            
            // If done, break the loop
            if (result.done) {
                break;
            }
            
            // Clear the buffer
            buffer = '';
        } catch (e) {
            // If we get here, the buffer doesn't contain a complete JSON object yet
            // Just continue reading more data
        }
    }
    
    // Finalize the message with markdown rendering
    finalizeStreamingMessage(fullResponse);
}

/**
 * Update the streaming message with new content
 */
function updateStreamingMessage(content) {
    if (!currentMessageDiv) return;
    
    const contentDiv = currentMessageDiv.querySelector('.message-content');
    const paragraph = contentDiv.querySelector('p');
    
    // Append the new content
    paragraph.textContent += content;
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Finalize the streaming message with markdown rendering
 */
function finalizeStreamingMessage(fullContent) {
    if (!currentMessageDiv) return;
    
    const contentDiv = currentMessageDiv.querySelector('.message-content');
    
    // Replace the content with markdown-rendered version
    contentDiv.innerHTML = marked.parse(fullContent);
    
    // Reset the current message div
    currentMessageDiv = null;
}

/**
 * Add a message to the UI
 */
function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Use marked.js to render markdown
    contentDiv.innerHTML = marked.parse(content);
    
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Add a loading message to the UI
 */
function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant loading-message';
    loadingDiv.innerHTML = `
        <div class="message-content">
            <div class="loading"></div>
            <span class="ms-2">Generating response...</span>
        </div>
    `;
    messagesContainer.appendChild(loadingDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Remove the loading message from the UI
 */
function removeLoadingMessage() {
    const loadingMessage = document.querySelector('.loading-message');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

/**
 * Show a status message
 */
function showStatus(message, type = 'info', duration = 0) {
    statusDiv.className = `alert alert-${type} mt-auto mb-0`;
    statusText.textContent = message;
    statusDiv.classList.remove('d-none');
    
    if (duration > 0) {
        setTimeout(() => {
            hideStatus();
        }, duration);
    }
}

/**
 * Hide the status message
 */
function hideStatus() {
    statusDiv.classList.add('d-none');
}

/**
 * Adjust the height of the textarea based on its content
 */
function adjustTextareaHeight() {
    userInput.style.height = 'auto';
    userInput.style.height = `${userInput.scrollHeight}px`;
}

/**
 * Generate a random ID
 */
function generateId() {
    return Math.random().toString(36).substring(2, 15);
}
