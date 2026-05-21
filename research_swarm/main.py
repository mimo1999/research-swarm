"""CLI entry point -- run a research swarm session from the terminal."""
import asyncio
import uuid

from research_swarm.config import settings
from research_swarm.schemas import ResearchDepth, ResearchQuery


async def main() -> None:
    session_id = str(uuid.uuid4())
    query = ResearchQuery(
        topic="Impact of large language models on scientific research",
        depth=ResearchDepth.standard,
        max_sources=settings.max_sources,
        audience="technical",
    )
    print(f"Session {session_id}: researching '{query.topic}' ...")
    # Graph execution will be wired in Phase 4.
    raise NotImplementedError("Graph not yet wired -- see Phase 4.")


if __name__ == "__main__":
    asyncio.run(main())
