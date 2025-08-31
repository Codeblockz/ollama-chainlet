# MCP Configuration

This directory contains configuration files for Model Context Protocol (MCP) servers.

## Configuration Format

The `servers.json` file follows this format:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "command-to-run",
      "args": ["arg1", "arg2"],
      "env": {
        "ENV_VAR": "value"
      }
    },
    "http-server": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## Server Types

### STDIO Servers
Use `command` and `args` for servers that communicate via stdin/stdout:

```json
{
  "my-tool-server": {
    "command": "python",
    "args": ["-m", "my_mcp_server"],
    "env": {
      "API_KEY": "your-api-key"
    }
  }
}
```

### HTTP Servers
Use `url` for servers that communicate via HTTP:

```json
{
  "web-server": {
    "url": "http://localhost:8000/mcp"
  }
}
```

## Environment Variables

Set `MCP_CONFIG_PATH` to use a different configuration file:

```bash
export MCP_CONFIG_PATH="/path/to/your/config.json"
```