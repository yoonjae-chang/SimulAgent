# AI Agent Streaming API

FastAPI application for streaming agent thinking and results asynchronously using Server-Sent Events (SSE).

## Features

- **Server-Sent Events (SSE) Streaming**: Real-time streaming of agent thinking, tool usage, and results
- **Async Execution**: Non-blocking agent execution with FastAPI's async support
- **Multiple Event Types**: Track thinking, tool usage, partial outputs, and final results
- **CORS Support**: Pre-configured for frontend integration
- **Simulation Mode**: Test streaming without making actual API calls
- **Comprehensive Error Handling**: Detailed logging and error responses

## Installation

1. Create and activate virtual environment:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
```

## Running the Server

### Development Mode (with auto-reload):
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- **GET** `/` - Root endpoint with API information
- **GET** `/health` - Health check endpoint

### Agent Execution

#### 1. Streaming Execution (SSE)
**POST** `/api/agent/stream`

Stream agent execution with real-time events.

**Request Body:**
```json
{
  "input": "Research the latest AI developments in 2024",
  "model": "openai/gpt-4.1",
  "mcp_servers": ["joerup/exa-mcp", "windsor/brave-search-mcp"],
  "stream": true
}
```

**Response:** Server-Sent Events stream

**Event Types:**
- `thinking`: Agent reasoning and thinking process
- `tool_use`: Agent is using a tool/MCP server
- `tool_result`: Result from tool execution
- `partial_output`: Partial response being generated
- `final_output`: Complete final response
- `error`: Error occurred during execution
- `complete`: Execution completed

**Example Event:**
```
event: thinking
data: {"event":"thinking","data":{"content":"Analyzing the request..."},"timestamp":"2024-10-24T12:00:00Z"}

event: final_output
data: {"event":"final_output","data":{"content":"Here are the results...","success":true},"timestamp":"2024-10-24T12:00:05Z"}

event: complete
data: {"event":"complete","data":{"message":"Agent execution completed successfully","success":true},"timestamp":"2024-10-24T12:00:05Z"}
```

#### 2. Non-Streaming Execution
**POST** `/api/agent/execute`

Execute agent and return final result only.

**Request Body:**
```json
{
  "input": "Research the latest AI developments in 2024",
  "model": "openai/gpt-4.1",
  "mcp_servers": ["joerup/exa-mcp", "windsor/brave-search-mcp"]
}
```

**Response:**
```json
{
  "success": true,
  "final_output": "Here are the latest AI developments...",
  "error": null,
  "execution_time": 15.3
}
```

#### 3. Simulated Streaming (Testing)
**POST** `/api/agent/stream/simulate`

Test streaming infrastructure with simulated events (no actual API calls).

Same request format as streaming endpoint, but returns simulated events with artificial delays.

### Utility Endpoints

#### Get Available Models
**GET** `/api/agent/models`

Returns list of available AI models.

**Response:**
```json
{
  "models": [
    {
      "id": "openai/gpt-4.1",
      "name": "GPT-4.1",
      "provider": "OpenAI"
    },
    {
      "id": "anthropic/claude-3-5-sonnet-20241022",
      "name": "Claude 3.5 Sonnet",
      "provider": "Anthropic"
    }
  ]
}
```

#### Get Available MCP Servers
**GET** `/api/agent/mcp-servers`

Returns list of available MCP servers.

**Response:**
```json
{
  "mcp_servers": [
    {
      "id": "joerup/exa-mcp",
      "name": "Exa Search",
      "description": "Semantic search engine"
    },
    {
      "id": "windsor/brave-search-mcp",
      "name": "Brave Search",
      "description": "Privacy-focused web search"
    }
  ]
}
```

## Frontend Integration

### JavaScript/TypeScript Example

```javascript
async function streamAgent(input) {
  const response = await fetch('http://localhost:8000/api/agent/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input: input,
      model: 'openai/gpt-4.1',
      mcp_servers: ['joerup/exa-mcp']
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.substring(6));

        switch (data.event) {
          case 'thinking':
            console.log('Thinking:', data.data.content);
            break;
          case 'tool_use':
            console.log('Using tool:', data.data.tool);
            break;
          case 'final_output':
            console.log('Final output:', data.data.content);
            break;
          case 'complete':
            console.log('Completed!');
            break;
        }
      }
    }
  }
}
```

### React Example with EventSource

```jsx
import { useEffect, useState } from 'react';

function AgentChat() {
  const [messages, setMessages] = useState([]);
  const [thinking, setThinking] = useState('');

  const executeAgent = async (input) => {
    // Note: EventSource doesn't support POST, so we need to use fetch
    const response = await fetch('http://localhost:8000/api/agent/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input: input,
        model: 'openai/gpt-4.1',
        mcp_servers: ['joerup/exa-mcp']
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.substring(6));

          if (data.event === 'thinking') {
            setThinking(data.data.content);
          } else if (data.event === 'final_output') {
            setMessages(prev => [...prev, data.data.content]);
            setThinking('');
          }
        }
      }
    }
  };

  return (
    <div>
      {thinking && <div className="thinking">{thinking}</div>}
      <div className="messages">
        {messages.map((msg, i) => <div key={i}>{msg}</div>)}
      </div>
    </div>
  );
}
```

## Project Structure

```
backend/
├── app.py                  # FastAPI application entry point
├── models.py              # Pydantic models for requests/responses
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── routers/
│   ├── __init__.py
│   └── agent_router.py   # Agent API endpoints
└── services/
    ├── __init__.py
    └── agent_service.py  # Agent execution logic
```

## Environment Variables

Required environment variables (add to `.env`):

```bash
# OpenAI API Key (if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key

# Anthropic API Key (if using Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Other API keys as needed for MCP servers
```

## Event Stream Format

All streaming responses follow the Server-Sent Events (SSE) format:

```
event: <event_type>
data: <json_payload>

```

Each event contains:
- `event`: Event type (thinking, tool_use, final_output, etc.)
- `data`: JSON object with event details
- `timestamp`: ISO 8601 timestamp

## Error Handling

The API includes comprehensive error handling:

1. **Validation Errors**: 422 status with details about invalid request data
2. **Execution Errors**: Streamed as error events with error details
3. **Server Errors**: 500 status with error message

Example error event:
```
event: error
data: {"event":"error","data":{"error":"API key not found","success":false},"timestamp":"2024-10-24T12:00:00Z"}
```

## Testing

Test the streaming endpoint with curl:

```bash
# Test simulated streaming
curl -X POST http://localhost:8000/api/agent/stream/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Test streaming",
    "model": "openai/gpt-4.1"
  }'

# Test actual agent execution
curl -X POST http://localhost:8000/api/agent/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Research AI developments",
    "model": "openai/gpt-4.1",
    "mcp_servers": ["joerup/exa-mcp"]
  }'
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Development

### Adding New Event Types

1. Add new event type to `EventType` enum in `models.py`
2. Update `_format_sse_event` in `services/agent_service.py`
3. Add handling in frontend integration

### Adding New MCP Servers

1. Update the MCP server list in `routers/agent_router.py` (`get_available_mcp_servers`)
2. Ensure the server is available in your environment

## Logging

Logs are written to stdout with the following format:
```
2024-10-24 12:00:00 - service_name - INFO - Log message
```

Log levels:
- INFO: General information and successful operations
- WARNING: Non-critical issues
- ERROR: Errors with stack traces

## License

[Your License Here]
