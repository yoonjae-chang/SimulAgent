# Agent Streaming Flow Diagram

## Overview

This document illustrates the complete flow of data through the FastAPI Agent Streaming API.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                            │
│                     (React/Next.js)                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP POST Request
                            │ { input, model, mcp_servers }
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        FASTAPI                              │
│                        (app.py)                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ - CORS Middleware                                    │  │
│  │ - Request Validation                                 │  │
│  │ - Route Matching                                     │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    AGENT ROUTER                             │
│                  (agent_router.py)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ POST /api/agent/stream                               │  │
│  │ POST /api/agent/execute                              │  │
│  │ POST /api/agent/stream/simulate                      │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGENT SERVICE                             │
│                (agent_service.py)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ execute_agent_streaming()                            │  │
│  │ execute_agent()                                      │  │
│  │ simulate_streaming_execution()                       │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   DEDALUS AGENT                             │
│                 (dedalus_labs)                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ AsyncDedalus Client                                  │  │
│  │ DedalusRunner                                        │  │
│  │ MCP Servers Integration                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Streaming Event Flow

```
┌─────────────┐
│   Client    │
│  Initiates  │
│   Request   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVENT STREAM STARTS                      │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────┐
│   Event: THINKING    │  ← "Initializing agent execution..."
│   Timestamp: T0      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Event: THINKING    │  ← "Analyzing the request..."
│   Timestamp: T1      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Event: TOOL_USE    │  ← Using joerup/exa-mcp
│   Timestamp: T2      │
│   Data: {tool, input}│
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Event: TOOL_RESULT  │  ← Results from search
│   Timestamp: T3      │
│   Data: {tool, result}│
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Event: TOOL_USE    │  ← Using brave-search-mcp
│   Timestamp: T4      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Event: TOOL_RESULT  │  ← Results from search
│   Timestamp: T5      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Event: PARTIAL_OUTPUT│  ← "Based on my research..."
│   Timestamp: T6      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Event: PARTIAL_OUTPUT│  ← "I found several insights..."
│   Timestamp: T7      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Event: FINAL_OUTPUT │  ← Complete response
│   Timestamp: T8      │
│   Data: {content,    │
│          success}    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Event: COMPLETE    │  ← Execution finished
│   Timestamp: T9      │
│   Data: {message,    │
│          success}    │
└──────┬───────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                  EVENT STREAM ENDS                          │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Client Processes    │
│   Final Result       │
└──────────────────────┘
```

## Error Flow

```
┌─────────────┐
│   Client    │
│  Request    │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│   Event: THINKING    │  ← Normal execution starts
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Event: TOOL_USE    │  ← Tool execution
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│    ❌ ERROR          │  ← API key invalid, timeout, etc.
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Event: ERROR       │  ← Error event with details
│   Data: {error,      │
│          success:    │
│          false}      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Event: COMPLETE    │  ← Completion event with error status
│   Data: {message,    │
│          success:    │
│          false}      │
└──────┬───────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                  EVENT STREAM ENDS                          │
└─────────────────────────────────────────────────────────────┘
```

## Server-Sent Events (SSE) Format

Each event is transmitted in this format:

```
event: thinking
data: {"event":"thinking","data":{"content":"Analyzing..."},"timestamp":"2024-10-24T12:00:00Z"}

event: tool_use
data: {"event":"tool_use","data":{"tool":"joerup/exa-mcp","input":{...}},"timestamp":"2024-10-24T12:00:01Z"}

event: final_output
data: {"event":"final_output","data":{"content":"Here is the result...","success":true},"timestamp":"2024-10-24T12:00:05Z"}

event: complete
data: {"event":"complete","data":{"message":"Completed successfully","success":true},"timestamp":"2024-10-24T12:00:05Z"}
```

## Request/Response Lifecycle

### 1. Client → Server (Request)

```javascript
POST /api/agent/stream
Content-Type: application/json

{
  "input": "Research the latest AI developments",
  "model": "openai/gpt-4.1",
  "mcp_servers": ["joerup/exa-mcp", "windsor/brave-search-mcp"],
  "stream": true
}
```

### 2. Server → Client (Response Headers)

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

### 3. Server → Client (Event Stream)

```
event: thinking
data: {...}

event: tool_use
data: {...}

event: final_output
data: {...}

event: complete
data: {...}
```

### 4. Connection Close

Client closes connection after receiving `complete` event.

## Data Flow Through Components

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Request arrives at FastAPI                                │
│    - CORS check                                              │
│    - Pydantic validation (AgentRequest model)                │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Router receives request                                   │
│    - Route: POST /api/agent/stream                           │
│    - Calls: agent_service.execute_agent_streaming()          │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Agent Service starts execution                            │
│    - Creates AsyncDedalus client                             │
│    - Initializes DedalusRunner                               │
│    - Yields initial "thinking" event                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. DedalusRunner executes                                    │
│    - Processes input                                         │
│    - Calls MCP servers                                       │
│    - Generates thinking steps                                │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Agent Service streams events                              │
│    - Formats events as SSE                                   │
│    - Yields events via async generator                       │
│    - Adds timestamps                                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. FastAPI StreamingResponse                                 │
│    - Consumes async generator                                │
│    - Sends events to client                                  │
│    - Manages connection                                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. Client receives and processes                             │
│    - Parses SSE format                                       │
│    - Updates UI in real-time                                 │
│    - Handles each event type                                 │
└──────────────────────────────────────────────────────────────┘
```

## Event Type Details

### THINKING
```json
{
  "event": "thinking",
  "data": {
    "content": "Agent's internal reasoning process..."
  },
  "timestamp": "2024-10-24T12:00:00Z"
}
```

**Use case**: Display what the agent is thinking in real-time

### TOOL_USE
```json
{
  "event": "tool_use",
  "data": {
    "tool": "joerup/exa-mcp",
    "input": {"query": "AI developments 2024"}
  },
  "timestamp": "2024-10-24T12:00:01Z"
}
```

**Use case**: Show which tools the agent is using

### TOOL_RESULT
```json
{
  "event": "tool_result",
  "data": {
    "tool": "joerup/exa-mcp",
    "result": "Found 25 relevant articles..."
  },
  "timestamp": "2024-10-24T12:00:02Z"
}
```

**Use case**: Display tool execution results

### PARTIAL_OUTPUT
```json
{
  "event": "partial_output",
  "data": {
    "content": "Based on my research, "
  },
  "timestamp": "2024-10-24T12:00:03Z"
}
```

**Use case**: Build the response incrementally (like ChatGPT typing effect)

### FINAL_OUTPUT
```json
{
  "event": "final_output",
  "data": {
    "content": "Complete response text here...",
    "success": true
  },
  "timestamp": "2024-10-24T12:00:05Z"
}
```

**Use case**: Display the complete final response

### ERROR
```json
{
  "event": "error",
  "data": {
    "error": "API key not found",
    "success": false
  },
  "timestamp": "2024-10-24T12:00:05Z"
}
```

**Use case**: Show error messages to the user

### COMPLETE
```json
{
  "event": "complete",
  "data": {
    "message": "Agent execution completed successfully",
    "success": true
  },
  "timestamp": "2024-10-24T12:00:05Z"
}
```

**Use case**: Signal end of stream, close connection, update UI state

## Connection Management

```
Client                          Server
  │                               │
  │──── Open Connection ──────────▶
  │                               │
  │◀──── 200 OK ─────────────────│
  │     (text/event-stream)       │
  │                               │
  │◀──── Event: thinking ────────│
  │                               │
  │◀──── Event: tool_use ─────────│
  │                               │
  │      (Connection stays open)  │
  │                               │
  │◀──── Event: final_output ─────│
  │                               │
  │◀──── Event: complete ─────────│
  │                               │
  │──── Close Connection ─────────▶
  │                               │
```

## Performance Characteristics

- **Latency**: Events streamed as generated (< 100ms delay)
- **Throughput**: Limited by agent processing, not streaming infrastructure
- **Memory**: Constant memory usage (streaming, not buffering)
- **Concurrency**: Limited by number of available workers (default: CPU cores)

## Future Enhancements

1. **WebSocket Support**: Alternative to SSE for bi-directional communication
2. **Resume Support**: Resume interrupted streams
3. **Batch Streaming**: Stream multiple agent executions in parallel
4. **Progress Tracking**: Add progress percentage to events
5. **Cancellation**: Support client-initiated cancellation via WebSocket
