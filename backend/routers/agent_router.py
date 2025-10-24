"""
Agent router with streaming and non-streaming endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import logging

from models import AgentRequest, AgentResponse
from services import AgentService

logger = logging.getLogger(__name__)

router = APIRouter()
agent_service = AgentService()


@router.post("/agent/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """
    Execute agent task without streaming (standard POST request)

    Args:
        request: AgentRequest containing input, model, and mcp_servers

    Returns:
        AgentResponse with the final output and execution details
    """
    try:
        logger.info(f"Received non-streaming agent request: {request.input[:100]}...")

        result = await agent_service.execute_agent(request)

        return AgentResponse(**result)

    except Exception as e:
        logger.error(f"Error in execute_agent endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/stream")
async def stream_agent(request: AgentRequest):
    """
    Execute agent task with Server-Sent Events (SSE) streaming

    This endpoint streams the agent's thinking process, tool usage, and results in real-time.

    Args:
        request: AgentRequest containing input, model, and mcp_servers

    Returns:
        StreamingResponse with SSE events

    Event Types:
        - thinking: Agent is thinking/reasoning
        - tool_use: Agent is using a tool
        - tool_result: Result from a tool
        - partial_output: Partial output being generated
        - final_output: Final complete output
        - error: An error occurred
        - complete: Execution completed

    Example usage with JavaScript:
    ```javascript
    const eventSource = new EventSource('/api/agent/stream', {
        method: 'POST',
        body: JSON.stringify({
            input: 'Your prompt here',
            model: 'openai/gpt-4.1',
            mcp_servers: ['joerup/exa-mcp']
        })
    });

    eventSource.addEventListener('thinking', (event) => {
        const data = JSON.parse(event.data);
        console.log('Thinking:', data.data.content);
    });

    eventSource.addEventListener('final_output', (event) => {
        const data = JSON.parse(event.data);
        console.log('Final Output:', data.data.content);
    });

    eventSource.addEventListener('complete', (event) => {
        eventSource.close();
    });
    ```
    """
    try:
        logger.info(f"Received streaming agent request: {request.input[:100]}...")

        return StreamingResponse(
            agent_service.execute_agent_streaming(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering in nginx
            }
        )

    except Exception as e:
        logger.error(f"Error in stream_agent endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/stream/simulate")
async def simulate_stream_agent(request: AgentRequest):
    """
    Simulate streaming agent execution for testing

    This endpoint simulates the streaming response with artificial delays,
    useful for testing the frontend integration without using actual API calls.

    Args:
        request: AgentRequest containing input, model, and mcp_servers

    Returns:
        StreamingResponse with simulated SSE events
    """
    try:
        logger.info(f"Received simulated streaming request: {request.input[:100]}...")

        return StreamingResponse(
            agent_service.simulate_streaming_execution(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Error in simulate_stream_agent endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/models")
async def get_available_models():
    """
    Get list of available models

    Returns:
        List of available model identifiers
    """
    return {
        "models": [
            {
                "id": "openai/gpt-4.1",
                "name": "GPT-4.1",
                "provider": "OpenAI"
            },
            {
                "id": "openai/gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "provider": "OpenAI"
            },
            {
                "id": "anthropic/claude-3-5-sonnet-20241022",
                "name": "Claude 3.5 Sonnet",
                "provider": "Anthropic"
            },
            {
                "id": "anthropic/claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "provider": "Anthropic"
            }
        ]
    }


@router.get("/agent/mcp-servers")
async def get_available_mcp_servers():
    """
    Get list of available MCP servers

    Returns:
        List of available MCP server identifiers
    """
    return {
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
