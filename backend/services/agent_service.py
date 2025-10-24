"""
Agent service for executing AI agent tasks with streaming support
"""
import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime

from dedalus_labs import AsyncDedalus, DedalusRunner
from models import AgentRequest, EventType, StreamEvent

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing agent execution and streaming"""

    def __init__(self):
        """Initialize the agent service"""
        self.client = AsyncDedalus()
        logger.info("AgentService initialized")

    async def execute_agent_streaming(
        self,
        request: AgentRequest
    ) -> AsyncGenerator[str, None]:
        """
        Execute agent with streaming support

        Args:
            request: The agent request containing input, model, and mcp_servers

        Yields:
            Server-Sent Events (SSE) formatted strings
        """
        try:
            logger.info(f"Starting streaming agent execution with model: {request.model}")

            # Send initial event
            yield self._format_sse_event(
                EventType.THINKING,
                {"content": "Initializing agent execution..."}
            )

            runner = DedalusRunner(self.client)

            # Track execution state
            execution_started = False
            partial_output = []

            # Execute the agent run
            logger.info(f"Executing agent with input length: {len(request.input)}")

            # For now, we'll execute and then send results
            # The dedalus_labs library may have streaming capabilities we can tap into
            result = await runner.run(
                input=request.input,
                model=request.model,
                mcp_servers=request.mcp_servers
            )

            # Send thinking events during execution
            if hasattr(result, 'steps') and result.steps:
                for step in result.steps:
                    if hasattr(step, 'thinking'):
                        yield self._format_sse_event(
                            EventType.THINKING,
                            {"content": step.thinking}
                        )

                    if hasattr(step, 'tool_calls'):
                        for tool_call in step.tool_calls:
                            yield self._format_sse_event(
                                EventType.TOOL_USE,
                                {
                                    "tool": tool_call.get("name", "unknown"),
                                    "input": tool_call.get("input", {})
                                }
                            )

            # Send partial outputs if available
            if hasattr(result, 'intermediate_outputs') and result.intermediate_outputs:
                for output in result.intermediate_outputs:
                    yield self._format_sse_event(
                        EventType.PARTIAL_OUTPUT,
                        {"content": output}
                    )

            # Send final output
            final_output = result.final_output if hasattr(result, 'final_output') else str(result)

            yield self._format_sse_event(
                EventType.FINAL_OUTPUT,
                {
                    "content": final_output,
                    "success": True
                }
            )

            # Send completion event
            yield self._format_sse_event(
                EventType.COMPLETE,
                {
                    "message": "Agent execution completed successfully",
                    "success": True
                }
            )

            logger.info("Streaming agent execution completed successfully")

        except Exception as e:
            logger.error(f"Error during streaming agent execution: {str(e)}", exc_info=True)

            # Send error event
            yield self._format_sse_event(
                EventType.ERROR,
                {
                    "error": str(e),
                    "success": False
                }
            )

            # Send completion event with error
            yield self._format_sse_event(
                EventType.COMPLETE,
                {
                    "message": f"Agent execution failed: {str(e)}",
                    "success": False
                }
            )

    async def execute_agent(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Execute agent without streaming (standard POST request)

        Args:
            request: The agent request containing input, model, and mcp_servers

        Returns:
            Dict containing the execution results
        """
        try:
            logger.info(f"Starting non-streaming agent execution with model: {request.model}")

            start_time = datetime.now()

            runner = DedalusRunner(self.client)

            result = await runner.run(
                input=request.input,
                model=request.model,
                mcp_servers=request.mcp_servers
            )

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            final_output = result.final_output if hasattr(result, 'final_output') else str(result)

            logger.info(f"Non-streaming agent execution completed in {execution_time:.2f}s")

            return {
                "success": True,
                "final_output": final_output,
                "error": None,
                "execution_time": execution_time
            }

        except Exception as e:
            logger.error(f"Error during agent execution: {str(e)}", exc_info=True)
            return {
                "success": False,
                "final_output": None,
                "error": str(e),
                "execution_time": None
            }

    def _format_sse_event(self, event_type: EventType, data: Dict[str, Any]) -> str:
        """
        Format data as Server-Sent Event (SSE)

        Args:
            event_type: Type of the event
            data: Event data

        Returns:
            Formatted SSE string
        """
        event = StreamEvent(
            event=event_type,
            data=data,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

        # SSE format: event: <type>\ndata: <json>\n\n
        return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"

    async def simulate_streaming_execution(
        self,
        request: AgentRequest
    ) -> AsyncGenerator[str, None]:
        """
        Simulate streaming execution with artificial delays
        This is useful for testing the streaming infrastructure

        Args:
            request: The agent request

        Yields:
            SSE formatted events
        """
        try:
            # Simulate thinking
            thinking_steps = [
                "Analyzing the input request...",
                "Planning the execution strategy...",
                "Preparing to use MCP servers...",
                "Starting information gathering..."
            ]

            for step in thinking_steps:
                yield self._format_sse_event(
                    EventType.THINKING,
                    {"content": step}
                )
                await asyncio.sleep(0.5)

            # Simulate tool use
            for mcp_server in request.mcp_servers:
                yield self._format_sse_event(
                    EventType.TOOL_USE,
                    {
                        "tool": mcp_server,
                        "input": {"query": request.input[:50] + "..."}
                    }
                )
                await asyncio.sleep(0.5)

                # Simulate tool result
                yield self._format_sse_event(
                    EventType.TOOL_RESULT,
                    {
                        "tool": mcp_server,
                        "result": f"Results from {mcp_server}"
                    }
                )
                await asyncio.sleep(0.5)

            # Simulate partial outputs
            partial_texts = [
                "Based on my research, ",
                "I found several key insights: ",
                "1. Recent developments show...",
                "2. Industry trends indicate..."
            ]

            for text in partial_texts:
                yield self._format_sse_event(
                    EventType.PARTIAL_OUTPUT,
                    {"content": text}
                )
                await asyncio.sleep(0.5)

            # Final output
            yield self._format_sse_event(
                EventType.FINAL_OUTPUT,
                {
                    "content": "This is a simulated response for testing. " +
                              "In production, this would contain the actual agent output.",
                    "success": True
                }
            )

            # Complete
            yield self._format_sse_event(
                EventType.COMPLETE,
                {
                    "message": "Simulation completed successfully",
                    "success": True
                }
            )

        except Exception as e:
            logger.error(f"Error during simulated execution: {str(e)}")
            yield self._format_sse_event(
                EventType.ERROR,
                {"error": str(e), "success": False}
            )
