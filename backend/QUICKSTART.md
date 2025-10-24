# Quick Start Guide

Get the FastAPI Agent Streaming API up and running in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Setup

1. **Install dependencies** (already done if you have the .venv folder):
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Start the server**:
   ```bash
   python app.py
   ```

   Or use the run script:
   ```bash
   ./run.sh
   ```

   The server will start on `http://localhost:8000`

## Test the API

### Option 1: Use the test client
```bash
# In a new terminal, with venv activated
python test_client.py
```

### Option 2: Use curl
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test simulated streaming
curl -N -X POST http://localhost:8000/api/agent/stream/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Test streaming",
    "model": "openai/gpt-4.1"
  }'
```

### Option 3: Browse the API docs
Open your browser and go to:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### 1. Stream Agent Execution (SSE)
```bash
POST /api/agent/stream
```

Real-time streaming of agent thinking and results.

**Example request:**
```json
{
  "input": "Research the latest AI developments",
  "model": "openai/gpt-4.1",
  "mcp_servers": ["joerup/exa-mcp"]
}
```

### 2. Non-Streaming Execution
```bash
POST /api/agent/execute
```

Get only the final result.

### 3. Simulated Streaming (Testing)
```bash
POST /api/agent/stream/simulate
```

Test the streaming infrastructure without making actual API calls.

## Frontend Integration Example

```javascript
async function streamAgent(input) {
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
    const lines = chunk.split('\\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.substring(6));
        console.log(data.event, data.data);
      }
    }
  }
}
```

## Event Types

The streaming endpoint emits these events:

- `thinking` - Agent reasoning process
- `tool_use` - Agent using a tool/MCP server
- `tool_result` - Result from tool execution
- `partial_output` - Partial response
- `final_output` - Complete response
- `error` - Error occurred
- `complete` - Execution finished

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use: `lsof -i :8000`
- Try a different port: `uvicorn app:app --port 8001`

### Import errors
- Make sure virtual environment is activated: `source .venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### API key errors
- Check your `.env` file has the correct API keys
- Verify environment variables are loaded: `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out the API documentation at http://localhost:8000/docs
- Integrate with your frontend application

## Support

For issues or questions, please check the README.md or create an issue in the repository.
