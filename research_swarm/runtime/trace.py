"""Timestamped, per-session trace log of every agent step, LLM call and tool call.

Purpose: make a swarm run reviewable after the fact. Every event is written as
one JSON line to ``<data_dir>/traces/<session_id>.jsonl`` and mirrored to the
``research_swarm.trace`` logger as a compact human-readable line. Each event
carries a wall-clock ISO timestamp (``ts``) and seconds since the session's
first event (``t``), so a run can be replayed as a timeline and bottlenecks
computed from the durations (``dur``).

Event kinds (``ev``):
  node_start / node_end   -- one graph node invocation (supervisor, worker_node, ...)
  llm_start / llm_end     -- one LLM round trip, with tokens, latency, and the
                             response text (full, capped) for review
  llm_error               -- an LLM call that raised
  tool                    -- one tool call (search / fetch)
  step                    -- any other timed sub-step (source ranking, embedding, ...)
  note                    -- free-form annotation (parse-failure recovery, etc.)

Tracing is best-effort: nothing here may ever raise into the graph.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("research_swarm.trace")

# Cap on any single free-text field written to the trace. Large enough to keep a
# whole worker claim / report section readable, small enough that a runaway
# reasoning trace can't bloat the file.
TEXT_CAP = 8000

_lock = threading.Lock()
_starts: dict[str, float] = {}
_enabled = True


def set_enabled(value: bool) -> None:
    global _enabled
    _enabled = value


def trace_dir() -> Path:
    from research_swarm.config import settings

    d = settings.data_dir / "traces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def trace_path(session_id: str) -> Path:
    return trace_dir() / f"{session_id}.jsonl"


def _clip(value: Any, cap: int = TEXT_CAP) -> Any:
    if isinstance(value, str) and len(value) > cap:
        return value[:cap] + f"...[+{len(value) - cap} chars]"
    return value


def trace_event(session_id: str | None, agent: str, ev: str, **fields: Any) -> None:
    """Append one event to the session trace. Never raises."""
    if not _enabled:
        return
    try:
        sid = session_id or "default"
        now = time.time()
        with _lock:
            t0 = _starts.setdefault(sid, now)
            record = {
                "ts": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
                "t": round(now - t0, 3),
                "agent": agent,
                "ev": ev,
                **{k: _clip(v) for k, v in fields.items()},
            }
            with trace_path(sid).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        summary = {k: v for k, v in fields.items() if k not in {"text", "prompt_tail", "output"}}
        logger.info(
            "[%7.2fs] %-28s %-10s %s",
            record["t"], agent, ev,
            " ".join(f"{k}={_clip(v, 120)!r}" for k, v in summary.items()),
        )
    except Exception:  # noqa: BLE001 - tracing must never break a run
        logger.debug("trace_event failed", exc_info=True)


def reset_session(session_id: str) -> None:
    """Forget a session's t=0 (a fresh run should start its own timeline)."""
    with _lock:
        _starts.pop(session_id, None)


@contextmanager
def timed(
    session_id: str | None, agent: str, ev: str = "step", **fields: Any,
) -> Iterator[dict[str, Any]]:
    """Time a block; emits one event with ``dur`` (seconds) when it exits.

    The yielded dict can be filled in by the caller to attach results::

        with timed(sid, "fetch", "step", name="pubmed") as info:
            info["n"] = len(results)
    """
    extra: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        yield extra
    except BaseException as exc:
        extra["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace_event(
            session_id, agent, ev, dur=round(time.perf_counter() - t0, 3), **fields, **extra,
        )


def traced_node(name: str):
    """Decorator for graph node functions: log start/end, duration and a summary
    of what the node returned. Preserves the wrapped coroutine's identity for
    ``unittest.mock.patch`` (functools.wraps)."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(state):
            sid = state.get("session_id", "default") if isinstance(state, dict) else "default"
            t0 = time.perf_counter()
            trace_event(sid, name, "node_start", **_node_input_summary(state))
            try:
                out = await fn(state)
            except BaseException as exc:
                trace_event(
                    sid, name, "node_end", dur=round(time.perf_counter() - t0, 3),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            trace_event(
                sid, name, "node_end", dur=round(time.perf_counter() - t0, 3),
                **_node_output_summary(out),
            )
            return out

        return wrapper

    return deco


def _node_input_summary(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    out: dict[str, Any] = {}
    if state.get("active_sub_question"):
        out["sub_question"] = state["active_sub_question"]
    if state.get("findings"):
        out["n_findings_in"] = len(state["findings"])
    if state.get("research_rounds"):
        out["round"] = state["research_rounds"]
    return out


def _node_output_summary(out: Any) -> dict[str, Any]:
    if not isinstance(out, dict):
        return {}
    summary: dict[str, Any] = {}
    if out.get("findings"):
        summary["n_findings_out"] = len(out["findings"])
    if out.get("critiques"):
        summary["n_critiques_out"] = len(out["critiques"])
    if out.get("next_agent"):
        summary["next_agent"] = out["next_agent"]
    if out.get("final_report") is not None:
        summary["report"] = True
    msgs = out.get("messages") or []
    if msgs:
        summary["msg"] = _clip(getattr(msgs[-1], "content", str(msgs[-1])), 400)
    return summary


# ---------------------------------------------------------------------------
# LLM call tracing
# ---------------------------------------------------------------------------

def _msg_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


class TraceCallback(BaseCallbackHandler):
    """LangChain callback that logs every LLM round trip made by one agent."""

    # Run on the calling thread so start/end timestamps aren't skewed by
    # executor scheduling delay.
    run_inline = True

    def __init__(self, session_id: str, agent: str, tier: str = "") -> None:
        super().__init__()
        self.session_id = session_id
        self.agent = agent
        self.tier = tier
        self._runs: dict[Any, float] = {}

    def on_chat_model_start(  # noqa: ARG002
        self, serialized: dict[str, Any], messages: list[list[Any]], *,
        run_id: Any = None, **kwargs: Any,
    ) -> None:
        self._runs[run_id] = time.perf_counter()
        flat = messages[0] if messages else []
        prompt_chars = sum(len(_msg_text(m)) for m in flat)
        last = _msg_text(flat[-1]) if flat else ""
        model = (
            (kwargs.get("metadata") or {}).get("ls_model_name")
            or (kwargs.get("invocation_params") or {}).get("model")
            or (serialized or {}).get("name", "")
        )
        trace_event(
            self.session_id, self.agent, "llm_start",
            tier=self.tier, model=model, n_messages=len(flat), prompt_chars=prompt_chars,
            prompt_tail=last[-1500:],
        )

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:  # noqa: ARG002
        t0 = self._runs.pop(run_id, None)
        dur = round(time.perf_counter() - t0, 3) if t0 is not None else None
        text, tool_calls, in_tok, out_tok, reasoning_chars, meta = "", [], 0, 0, 0, {}
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            text = _msg_text(msg) if msg is not None else getattr(gen, "text", "")
            if msg is not None:
                tool_calls = [
                    {"name": tc.get("name"), "args": tc.get("args")}
                    for tc in (getattr(msg, "tool_calls", None) or [])
                ]
                usage = getattr(msg, "usage_metadata", None) or {}
                in_tok = usage.get("input_tokens", 0) or 0
                out_tok = usage.get("output_tokens", 0) or 0
                extra = getattr(msg, "additional_kwargs", None) or {}
                reasoning_chars = len(extra.get("reasoning_content", "") or "")
                rm = getattr(msg, "response_metadata", None) or {}
                # Ollama reports nanoseconds; surface the server-side split.
                for key in (
                    "total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
                ):
                    if rm.get(key):
                        meta[key.replace("_duration", "_s")] = round(rm[key] / 1e9, 2)
                if rm.get("done_reason"):
                    meta["done_reason"] = rm["done_reason"]
        except Exception:  # noqa: BLE001
            pass
        trace_event(
            self.session_id, self.agent, "llm_end",
            tier=self.tier, dur=dur, in_tokens=in_tok, out_tokens=out_tok,
            reasoning_chars=reasoning_chars, out_chars=len(text), tool_calls=tool_calls,
            **meta, output=text,
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:  # noqa: ARG002
        t0 = self._runs.pop(run_id, None)
        trace_event(
            self.session_id, self.agent, "llm_error", tier=self.tier,
            dur=round(time.perf_counter() - t0, 3) if t0 is not None else None,
            error=f"{type(error).__name__}: {error}",
        )
