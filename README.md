# Ollama Chainlet

A Python web application that provides a chainlet interface for interacting with local Ollama models.

## Features

- Web-based chat interface for Ollama models
- Model selection from available Ollama models
- Conversation history management
- Simple chainlet framework for extensibility

## Requirements

- Python 3.8+
- Ollama installed and running locally
- Internet connection for web assets (Bootstrap, etc.)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd ollama-chainlet
   ```

2. Run the setup script to create a virtual environment and install dependencies:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

## Usage

1. Ensure Ollama is running locally.

2. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

3. Start the application:
   ```
   python app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

5. Select a model from the dropdown and start chatting!

## Project Structure

```
ollama-chainlet/
├── app.py                 # Main Flask application
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── main.js        # Frontend logic
├── templates/
│   └── index.html         # Main interface
├── chainlet/
│   ├── __init__.py
│   ├── core.py            # Core chainlet functionality
│   └── ollama.py          # Ollama integration
├── requirements.txt       # Dependencies
├── .gitignore             # Git ignore file
└── README.md              # Documentation
```

## License

MIT

## Acknowledgements

- [Ollama](https://ollama.ai/) for providing the local LLM runtime
- [Flask](https://flask.palletsprojects.com/) for the web framework
