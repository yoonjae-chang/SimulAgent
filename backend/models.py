"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class MCPServer(str, Enum):
    """Available MCP servers"""
    EXA_SEARCH = "joerup/exa-mcp"
    BRAVE_SEARCH = "windsor/brave-search-mcp"


class AgentRequest(BaseModel):
    """Request model for agent execution"""
    input: str = Field(
        ...,
        description="The input prompt or task for the agent to execute",
        min_length=1
    )
    model: str = Field(
        default="openai/gpt-4.1",
        description="The model to use for the agent (e.g., openai/gpt-4.1, anthropic/claude-3-5-sonnet-20241022)"
    )
    mcp_servers: List[str] = Field(
        default=["joerup/exa-mcp", "windsor/brave-search-mcp"],
        description="List of MCP servers to use for the agent"
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream the response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "input": "Research the latest AI developments in 2024",
                "model": "openai/gpt-4.1",
                "mcp_servers": ["joerup/exa-mcp", "windsor/brave-search-mcp"],
                "stream": True
            }
        }


class EventType(str, Enum):
    """Types of events in the streaming response"""
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    PARTIAL_OUTPUT = "partial_output"
    FINAL_OUTPUT = "final_output"
    ERROR = "error"
    COMPLETE = "complete"


class StreamEvent(BaseModel):
    """Model for streaming events"""
    event: EventType = Field(..., description="Type of event")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: Optional[str] = Field(None, description="Timestamp of the event")

    class Config:
        json_schema_extra = {
            "example": {
                "event": "thinking",
                "data": {
                    "content": "I need to search for AI developments..."
                },
                "timestamp": "2024-10-24T12:00:00Z"
            }
        }


class AgentResponse(BaseModel):
    """Response model for non-streaming agent execution"""
    success: bool = Field(..., description="Whether the execution was successful")
    final_output: Optional[str] = Field(None, description="The final output from the agent")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "final_output": "Here are the latest AI developments...",
                "error": None,
                "execution_time": 15.3
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: Optional[str] = Field(None, description="API version")
