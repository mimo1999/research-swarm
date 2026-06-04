"""Quick timed run to verify the shallow/Ollama-cloud path end-to-end."""
import asyncio
import time


async def main() -> None:

    from research_swarm.graph.builder import build_graph, get_thread_config
    from research_swarm.schemas import ResearchQuery
    from research_swarm.schemas.query import ResearchDepth

    graph = build_graph(interrupt_before_writer=False)  # uses MemorySaver(_serde) by default
    config = get_thread_config("speed-test")
    initial = {
        "messages": [],
        "query": ResearchQuery(
            topic="GAMs for temporal data",
            depth=ResearchDepth.shallow,
            max_sources=3,
            audience="technical",
        ),
        "plan": None,
        "findings": [],
        "critiques": [],
        "draft_report": None,
        "final_report": None,
        "human_feedback": None,
        "writer_instructions": None,
        "iteration_count": 0,
        "next_agent": None,
        "session_id": "speed-test",
        "model_provider": "ollama",
        "model_name": "minimax-m2.5:cloud",
    }

    t0 = time.time()
    step = 0
    print("Starting stream …", flush=True)
    async for chunk in graph.astream(initial, config, stream_mode="updates"):
        for node, upd in chunk.items():
            step += 1
            msgs = upd.get("messages") or []
            txt = next(
                (getattr(m, "content", "")[:70] for m in msgs if getattr(m, "content", "")),
                "",
            )
            print(f"  [{time.time() - t0:5.1f}s] step {step:2d}  {node:<15}  {txt}", flush=True)

    elapsed = time.time() - t0
    snap = await graph.aget_state(config)
    report = snap.values.get("final_report")
    print(f"\nTOTAL {elapsed:.1f}s", flush=True)
    print(f"   Report title : {getattr(report, 'title', None)}", flush=True)
    if report:
        secs = getattr(report, "sections", [])
        print(f"   Sections     : {len(secs)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
