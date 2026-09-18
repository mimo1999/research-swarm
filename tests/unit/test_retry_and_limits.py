"""Retry-with-backoff for transient LLM errors, and the per-session concurrency limiter."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.agents._utils import ainvoke_with_retry, is_transient
from research_swarm.runtime.limits import limiter


class _StatusError(Exception):
    """Mimics ollama.ResponseError / anthropic.RateLimitError: exposes .status_code."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class TestIsTransient:
    def test_transient_status_codes(self):
        for status in (408, 429, 500, 502, 503, 504, 529):
            assert is_transient(_StatusError("boom", status)), status
        for status in (400, 401, 403, 404, 422):
            assert not is_transient(_StatusError("boom", status)), status


class TestAinvokeWithRetry:
    @staticmethod
    def _runnable(side_effect):
        r = MagicMock()
        r.ainvoke = AsyncMock(side_effect=side_effect)
        return r


    @pytest.mark.asyncio
    async def test_retries_transient_errors_then_succeeds(self):
        r = self._runnable([_StatusError("busy", 429), _StatusError("busy", 503), "ok"])
        with patch("asyncio.sleep", new=AsyncMock()) as sleep:
            assert await ainvoke_with_retry(r, ["msg"], attempts=4) == "ok"
        assert r.ainvoke.await_count == 3
        assert sleep.await_count == 2


    @pytest.mark.asyncio
    async def test_non_transient_errors_are_not_retried(self):
        r = self._runnable([ValueError("Invalid json output")])
        with patch("asyncio.sleep", new=AsyncMock()) as sleep, pytest.raises(ValueError):
            await ainvoke_with_retry(r, ["m"])
        assert r.ainvoke.await_count == 1
        sleep.assert_not_awaited()


class TestLimiter:
    @pytest.mark.asyncio
    async def test_never_exceeds_the_limit_and_all_branches_finish(self):
        running = peak = 0

        async def branch():
            nonlocal running, peak
            async with limiter("doc", "sess", 2):
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0.01)
                running -= 1

        await asyncio.gather(*(branch() for _ in range(8)))
        assert peak == 2 and running == 0
