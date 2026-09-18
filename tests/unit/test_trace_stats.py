"""trace_stats.analyze: the concurrency / retry / fallback signals used to verify the LLM cap."""
from __future__ import annotations

import json

from research_swarm.runtime.trace_stats import analyze


def _ev(t, ev, agent="a", **kw):
    return {"ts": "x", "t": t, "agent": agent, "ev": ev, **kw}


class TestAnalyzeSignals:
    def _write(self, tmp_path, events):
        path = tmp_path / "s.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        return path

    def test_reports_retries_slot_waits_and_fallbacks(self, tmp_path):
        events = [
            _ev(0.0, "llm_start", "critic"), _ev(1.0, "llm_end", "critic", dur=1.0),
            _ev(1.0, "note", "critic", retry=1, delay=2.0, error="429"),
            _ev(1.5, "note", "critic", retry=2, delay=4.0, error="429"),
            _ev(2.0, "note", "critic", slot_wait_s=3.2),
            _ev(2.5, "note", "critic", slot_wait_s=1.3),
            _ev(3.0, "note", "critic.fallback", error="boom"),
            _ev(3.5, "note", "critic.fallback", error="boom"),
            _ev(4.0, "note", "supervisor.fallback", error="boom"),
        ]
        a = analyze(self._write(tmp_path, events))

        assert a["peak_llm_concurrency"] == 1
        assert a["llm_retries"] == 2
        assert a["slot_wait_s"] == 4.5
        assert a["fallbacks"] == {"critic.fallback": 2, "supervisor.fallback": 1}
