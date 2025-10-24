import asyncio

from dotenv import load_dotenv

from dedalus_labs import AsyncDedalus, DedalusRunner
from dedalus_labs.utils.streaming import stream_async

load_dotenv()

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    result = await runner.run(
        input="""I need to research the latest developments in AI agents for 2024.
        Please help me:
        1. Find recent news articles about AI agent breakthroughs
        2. Search for academic papers on multi-agent systems
        3. Look up startup companies working on AI agents
        4. Find GitHub repositories with popular agent frameworks
        5. Summarize the key trends and provide relevant links

        Focus on developments from the past 6 months.""",
        model="openai/gpt-4.1",
        mcp_servers=[
            "joerup/exa-mcp",         # Semantic search engine
            "windsor/brave-search-mcp"  # Privacy-focused web search
        ]
    )

    print(f"Web Search Results:\n{result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())