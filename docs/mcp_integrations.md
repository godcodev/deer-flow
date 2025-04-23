# MCP Integrations

## APIs

### Get Information of MCP Server

**POST /api/mcp/server/metadata**

For stdio type:
```json
{
  "type": "stdio",
  "command": "npx",
  "args": ["@agentdeskai/browser-tools-mcp@1.2.0"]
  "env": {
    "MCP_SERVER_ID": "mcp-github-trending"
  }
}
```

For SSE type:
```json
{
  "type": "sse",
  "url": "http://localhost:3000/sse",
  "env": {
    "API_KEY": "value"
  }
}
```

### Chat Stream

**POST /api/chat/stream**

```json
{
  ...
  "mcp_settings": {
    "servers": {
      "mcp-github-trending": {
        "command": "uvx",
        "args": ["mcp-github-trending"],
        "env": {
          "MCP_SERVER_ID": "mcp-github-trending"
        },
        "enabled_tools": ["get_github_trending_repositories"],
        "add_to_agents": ["researcher"]
      }
    }
  },
}
```
