"""
Test client for streaming agent API
"""
import asyncio
import httpx
import json
import sys


async def test_streaming():
    """Test the streaming endpoint"""
    url = "http://localhost:8000/api/agent/stream/simulate"

    request_data = {
        "input": "Research the latest AI developments in 2024",
        "model": "openai/gpt-4.1",
        "mcp_servers": ["joerup/exa-mcp"]
    }

    print("Testing streaming endpoint...")
    print(f"Request: {json.dumps(request_data, indent=2)}\n")
    print("Streaming events:\n")
    print("-" * 80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            url,
            json=request_data,
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(await response.aread())
                return

            current_event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]

                elif line.startswith("data: "):
                    data = json.loads(line[6:])

                    event_type = data.get("event", "unknown")
                    event_data = data.get("data", {})
                    timestamp = data.get("timestamp", "")

                    print(f"[{event_type.upper()}] {timestamp}")

                    if event_type == "thinking":
                        print(f"  → {event_data.get('content', '')}")

                    elif event_type == "tool_use":
                        tool = event_data.get('tool', 'unknown')
                        print(f"  → Using tool: {tool}")

                    elif event_type == "tool_result":
                        tool = event_data.get('tool', 'unknown')
                        result = event_data.get('result', '')
                        print(f"  → Result from {tool}: {result}")

                    elif event_type == "partial_output":
                        print(f"  → {event_data.get('content', '')}")

                    elif event_type == "final_output":
                        content = event_data.get('content', '')
                        print(f"  → {content}")

                    elif event_type == "error":
                        error = event_data.get('error', '')
                        print(f"  → ERROR: {error}")

                    elif event_type == "complete":
                        message = event_data.get('message', '')
                        success = event_data.get('success', False)
                        status = "SUCCESS" if success else "FAILED"
                        print(f"  → {message} [{status}]")

                    print()

    print("-" * 80)
    print("\nStreaming test completed!")


async def test_non_streaming():
    """Test the non-streaming endpoint"""
    url = "http://localhost:8000/api/agent/execute"

    request_data = {
        "input": "What is the capital of France?",
        "model": "openai/gpt-4.1",
        "mcp_servers": ["joerup/exa-mcp"]
    }

    print("\nTesting non-streaming endpoint...")
    print(f"Request: {json.dumps(request_data, indent=2)}\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=request_data)

        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def main():
    """Main test function"""
    print("=" * 80)
    print("FastAPI Agent Streaming Test Client")
    print("=" * 80)
    print()

    try:
        # Test streaming
        await test_streaming()

        # Uncomment to test non-streaming endpoint
        # await test_non_streaming()

    except httpx.ConnectError:
        print("\nError: Could not connect to the server.")
        print("Make sure the FastAPI server is running on http://localhost:8000")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
