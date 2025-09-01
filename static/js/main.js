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
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const sendButton = document.getElementById('send-button');

// Browser elements
const browserToggle = document.getElementById('browser-toggle');
const browserContainer = document.getElementById('browser-container');
const chatContainer = document.getElementById('chat-container');
const browserUrl = document.getElementById('browser-url');
const browserGoBtn = document.getElementById('browser-go');
const browserBackBtn = document.getElementById('browser-back');
const browserForwardBtn = document.getElementById('browser-forward');
const browserRefreshBtn = document.getElementById('browser-refresh');
const browserScreenshotBtn = document.getElementById('browser-screenshot');
const browserInspectBtn = document.getElementById('browser-inspect');
const browserLoading = document.getElementById('browser-loading');
const browserScreenshotView = document.getElementById('browser-screenshot-view');
const browserScreenshotImg = document.getElementById('browser-screenshot-img');

// State
let conversationId = generateId();
let isProcessing = false;
let currentMessageDiv = null;
let browserActive = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    setupEventListeners();
    adjustTextareaHeight();
    setupMobileMenu();
    setupBrowser();
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
    userInput.addEventListener('input', () => {
        adjustTextareaHeight();
        updateSendButtonState();
    });
    
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
    
    // Enter key handling for textarea
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isProcessing && userInput.value.trim()) {
                handleSubmit(e);
            }
        }
    });
}

/**
 * Start a new chat
 */
function startNewChat() {
    conversationId = generateId();
    
    // Clear messages and show welcome message
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-content">
                <h1>How can I help you today?</h1>
                <p>Choose a model from the sidebar and start chatting.</p>
            </div>
        </div>
    `;
    
    // Remove has-messages class to show welcome message
    messagesContainer.classList.remove('has-messages');
    
    // Clear the input
    userInput.value = '';
    adjustTextareaHeight();
    updateSendButtonState();
    
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
    
    // Hide welcome message and add user message to the UI
    messagesContainer.classList.add('has-messages');
    addMessage(message, 'user');
    
    // Check for URLs and offer browser navigation
    detectAndOfferBrowsing(message);
    
    // Clear the input
    userInput.value = '';
    adjustTextareaHeight();
    updateSendButtonState();
    
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
        updateSendButtonState();
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
    
    // Add avatar
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = 'AI';
    currentMessageDiv.appendChild(avatarDiv);
    
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
    
    // Read the Server-Sent Events stream
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
        
        // Process complete SSE messages from the buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the incomplete line in buffer
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const jsonData = line.substring(6); // Remove 'data: ' prefix
                    if (jsonData.trim() === '') continue; // Skip empty data lines
                    
                    const result = JSON.parse(jsonData);
                    
                    if (result.error) {
                        throw new Error(result.error);
                    }
                    
                    if (result.content) {
                        updateStreamingMessage(result.content);
                        fullResponse += result.content;
                    }
                    
                    if (result.done) {
                        // Finalize the message with markdown rendering
                        finalizeStreamingMessage(fullResponse);
                        return;
                    }
                } catch (e) {
                    console.error('Error parsing SSE data:', e, 'Data:', line);
                }
            }
        }
    }
    
    // Finalize the message with markdown rendering (fallback)
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
    
    // Add avatar for user and assistant messages
    if (role === 'user' || role === 'assistant') {
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.textContent = role === 'user' ? 'U' : 'AI';
        messageDiv.appendChild(avatarDiv);
    }
    
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
    loadingDiv.className = 'loading-message';
    
    // Add avatar
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = 'AI';
    loadingDiv.appendChild(avatarDiv);
    
    // Add loading indicator
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading-indicator';
    loadingIndicator.innerHTML = `
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
    `;
    loadingDiv.appendChild(loadingIndicator);
    
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

/**
 * Update send button state based on input and processing state
 */
function updateSendButtonState() {
    const hasText = userInput.value.trim().length > 0;
    const shouldEnable = hasText && !isProcessing;
    
    sendButton.disabled = !shouldEnable;
    
    if (isProcessing) {
        sendButton.classList.add('loading');
    } else {
        sendButton.classList.remove('loading');
    }
}

/**
 * Setup mobile menu functionality
 */
function setupMobileMenu() {
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
        
        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 && 
                sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) && 
                !sidebarToggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
        
        // Close sidebar on window resize to desktop
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768) {
                sidebar.classList.remove('open');
            }
        });
    }
}

/**
 * Setup browser functionality
 */
function setupBrowser() {
    // Browser toggle event listener
    if (browserToggle) {
        browserToggle.addEventListener('change', toggleBrowser);
        
        // Load saved preference
        const savedBrowserState = localStorage.getItem('browser-enabled');
        if (savedBrowserState === 'true') {
            browserToggle.checked = true;
            toggleBrowser();
        }
    }
    
    // Browser navigation event listeners
    if (browserGoBtn) {
        browserGoBtn.addEventListener('click', navigateBrowser);
    }
    
    if (browserUrl) {
        browserUrl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                navigateBrowser();
            }
        });
    }
    
    // Browser control event listeners
    if (browserBackBtn) {
        browserBackBtn.addEventListener('click', () => browserAction('back'));
    }
    
    if (browserForwardBtn) {
        browserForwardBtn.addEventListener('click', () => browserAction('forward'));
    }
    
    if (browserRefreshBtn) {
        browserRefreshBtn.addEventListener('click', () => {
            if (browserUrl.value) {
                navigateBrowser();
            }
        });
    }
    
    if (browserScreenshotBtn) {
        browserScreenshotBtn.addEventListener('click', takeBrowserScreenshot);
    }
    
    if (browserInspectBtn) {
        browserInspectBtn.addEventListener('click', inspectBrowserPage);
    }
}

/**
 * Toggle browser visibility
 */
function toggleBrowser() {
    const isEnabled = browserToggle && browserToggle.checked;
    browserActive = isEnabled;
    
    if (browserContainer && chatContainer) {
        if (isEnabled) {
            browserContainer.style.display = 'flex';
            chatContainer.classList.add('split-view');
        } else {
            browserContainer.style.display = 'none';
            chatContainer.classList.remove('split-view');
        }
    }
    
    // Save preference
    localStorage.setItem('browser-enabled', isEnabled.toString());
    
    // Adjust layout
    adjustLayout();
}

/**
 * Navigate browser to URL
 */
async function navigateBrowser() {
    const url = browserUrl.value.trim();
    if (!url) return;
    
    // Add protocol if missing
    const finalUrl = url.match(/^https?:\/\//) ? url : `https://${url}`;
    browserUrl.value = finalUrl;
    
    try {
        showBrowserLoading(true);
        hideBrowserScreenshot();
        
        const response = await fetch('/api/browser/navigate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: conversationId,
                url: finalUrl
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Automatically take screenshot after navigation
        setTimeout(() => {
            takeBrowserScreenshot();
        }, 2000);
        
        showStatus(`Navigated to ${finalUrl}`, 'success', 3000);
        
    } catch (error) {
        showBrowserLoading(false);
        showStatus(`Navigation error: ${error.message}`, 'danger', 5000);
        console.error('Navigation error:', error);
    }
}

/**
 * Execute browser action
 */
async function browserAction(action, params = {}) {
    try {
        showBrowserLoading(true);
        
        const response = await fetch('/api/browser/action', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: conversationId,
                action: action,
                params: params
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Take screenshot after action
        setTimeout(() => {
            takeBrowserScreenshot();
        }, 1000);
        
        showStatus(`Browser action completed: ${action}`, 'success', 2000);
        
    } catch (error) {
        showBrowserLoading(false);
        showStatus(`Browser action error: ${error.message}`, 'danger', 5000);
        console.error('Browser action error:', error);
    }
}

/**
 * Take browser screenshot
 */
async function takeBrowserScreenshot() {
    try {
        showBrowserLoading(true);
        
        const response = await fetch('/api/browser/screenshot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: conversationId
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        if (data.screenshot_data) {
            // Display screenshot in browser view
            showBrowserScreenshot(data.screenshot_data);
            
            // Also add to chat
            addMessage(`📸 Browser Screenshot taken`, 'system');
        }
        
        showBrowserLoading(false);
        
    } catch (error) {
        showBrowserLoading(false);
        showStatus(`Screenshot error: ${error.message}`, 'danger', 5000);
        console.error('Screenshot error:', error);
    }
}

/**
 * Inspect browser page
 */
async function inspectBrowserPage() {
    try {
        showBrowserLoading(true);
        
        const response = await fetch('/api/browser/snapshot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: conversationId
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        if (data.snapshot_data) {
            // Add page structure to chat
            addMessage(`🔍 Page Structure:\n\`\`\`\n${data.snapshot_data}\n\`\`\``, 'system');
        }
        
        showBrowserLoading(false);
        
    } catch (error) {
        showBrowserLoading(false);
        showStatus(`Inspection error: ${error.message}`, 'danger', 5000);
        console.error('Inspection error:', error);
    }
}

/**
 * Show browser loading state
 */
function showBrowserLoading(show) {
    if (browserLoading) {
        browserLoading.style.display = show ? 'flex' : 'none';
    }
}

/**
 * Show browser screenshot
 */
function showBrowserScreenshot(screenshotData) {
    if (browserScreenshotView && browserScreenshotImg) {
        browserScreenshotImg.src = `data:image/png;base64,${screenshotData}`;
        browserScreenshotView.style.display = 'block';
        
        // Hide placeholder
        const placeholder = document.querySelector('.browser-placeholder');
        if (placeholder) {
            placeholder.style.display = 'none';
        }
    }
}

/**
 * Hide browser screenshot
 */
function hideBrowserScreenshot() {
    if (browserScreenshotView) {
        browserScreenshotView.style.display = 'none';
        
        // Show placeholder
        const placeholder = document.querySelector('.browser-placeholder');
        if (placeholder) {
            placeholder.style.display = 'flex';
        }
    }
}

/**
 * Adjust layout for browser/chat split
 */
function adjustLayout() {
    // This will be handled by CSS classes
    // The split-view class is added/removed in toggleBrowser()
}

/**
 * Detect URLs in chat messages and offer browser navigation
 */
function detectAndOfferBrowsing(message) {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    const urls = message.match(urlRegex);
    
    if (urls && urls.length > 0 && !browserActive) {
        // Show suggestion to enable browser
        const suggestion = document.createElement('div');
        suggestion.className = 'url-suggestion';
        suggestion.innerHTML = `
            <p>🌐 URL detected: <strong>${urls[0]}</strong></p>
            <button class="btn btn-sm btn-primary" onclick="enableBrowserAndNavigate('${urls[0]}')">
                Open in Browser
            </button>
        `;
        
        messagesContainer.appendChild(suggestion);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    } else if (urls && urls.length > 0 && browserActive) {
        // Auto-navigate if browser is already active
        browserUrl.value = urls[0];
        navigateBrowser();
    }
}

/**
 * Enable browser and navigate to URL
 */
function enableBrowserAndNavigate(url) {
    if (browserToggle) {
        browserToggle.checked = true;
        toggleBrowser();
    }
    
    setTimeout(() => {
        if (browserUrl) {
            browserUrl.value = url;
            navigateBrowser();
        }
    }, 100);
}

// Make enableBrowserAndNavigate globally available
window.enableBrowserAndNavigate = enableBrowserAndNavigate;
