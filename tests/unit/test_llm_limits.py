"""Process-wide LLM concurrency cap (llm_slot), its use inside ainvoke_with_retry, and the
call-site behaviour built on it (retry, loud fallbacks, thinking policy, top-up)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.agents._utils import ainvoke_with_retry
from research_swarm.config import settings
from research_swarm.runtime import limits
from research_swarm.runtime.limits import llm_slot, set_llm_context


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _fresh_limits(monkeypatch):
    """Each test gets its own semaphores and a known cap."""
    monkeypatch.setattr(limits, "_llm_semaphores", {})
    monkeypatch.setattr(settings, "max_concurrent_llm_calls_ollama", 2)
    monkeypatch.setattr(settings, "max_concurrent_llm_calls_anthropic", 8)
    monkeypatch.setattr(settings, "max_concurrent_llm_calls_openai", 8)
    set_llm_context("ollama", None)


class TestLlmSlot:
    @pytest.mark.asyncio
    async def test_never_exceeds_the_cap_across_many_tasks(self):
        running = peak = 0

        async def call():
            nonlocal running, peak
            async with llm_slot():
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0.01)
                running -= 1

        await asyncio.gather(*(call() for _ in range(20)))
        assert peak == 2 and running == 0


class TestSlotInsideRetry:
    @staticmethod
    def _runnable(side_effect):
        r = MagicMock()
        r.ainvoke = AsyncMock(side_effect=side_effect)
        return r


    @pytest.mark.asyncio
    async def test_slot_is_free_while_backing_off(self):
        """A retrying call must not hold capacity during its backoff sleep."""
        seen_free: list[bool] = []
        r = self._runnable([_StatusError("busy", 429), "ok"])

        async def fake_sleep(_delay):
            # if the retrying call still held a slot, only 1 of the 2 would be acquirable here
            async with llm_slot(), llm_slot():
                seen_free.append(True)

        with patch("research_swarm.agents._utils.asyncio.sleep", new=fake_sleep):
            assert await ainvoke_with_retry(r, ["m"]) == "ok"
        assert seen_free == [True]


# ---------------------------------------------------------------------------
# Call sites: transient errors are retried, parse errors still reach recovery
# ---------------------------------------------------------------------------

def _structured(side_effect):
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=side_effect)
    return llm


class TestCallSitesRetry:


    @pytest.mark.asyncio
    async def test_still_no_plan_after_the_retry_falls_back_loudly(self, caplog):
        from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor
        from tests.unit.test_graph import _make_state

        no_plan = SupervisorDecision(reasoning="x", next_agent="dispatch", plan=None)
        llm = _structured([no_plan, no_plan])
        with patch("research_swarm.agents.supervisor.trace_event") as trace,              caplog.at_level("ERROR"):
            out = await run_supervisor(_make_state(plan=None), llm)

        assert out.reasoning.startswith("FALLBACK PLAN")
        assert len(out.plan.sub_questions) == 1                    # never a plan-less decision
        assert any("fallback plan" in r.message for r in caplog.records)
        assert "supervisor.fallback" in [c.args[1] for c in trace.call_args_list]


# ---------------------------------------------------------------------------
# Thinking policy
# ---------------------------------------------------------------------------

class _FakeOllama:
    """Stands in for ChatOllama: has the two fields and model_copy semantics we rely on."""

    def __init__(self, reasoning=True, num_predict=None, callbacks=None):
        self.reasoning = reasoning
        self.num_predict = num_predict
        self.callbacks = callbacks

    def model_copy(self, update=None):
        clone = _FakeOllama(self.reasoning, self.num_predict, self.callbacks)
        for k, v in (update or {}).items():
            setattr(clone, k, v)
        return clone


class TestWithoutThinking:
    def test_disables_reasoning_and_caps_output_on_ollama(self):
        from research_swarm.agents.base import without_thinking

        out = without_thinking(_FakeOllama(reasoning=True), max_tokens=4096)
        assert out.reasoning is False and out.num_predict == 4096


# ---------------------------------------------------------------------------
# Top-up rule for thin sub-questions
# ---------------------------------------------------------------------------

def _paper(n):
    return {"url": f"https://p/{n}", "title": f"P{n}", "snippet": "x" * 200, "source_type": "web"}
