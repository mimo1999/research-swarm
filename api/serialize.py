"""Recursively turn a graph node-update dict (Pydantic models, LangChain
messages, enums, datetimes) into plain JSON-able Python for SSE payloads.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items() if k != "messages"}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    # LangChain BaseMessage and similar objects expose `.content`
    if hasattr(value, "content"):
        return {"type": type(value).__name__, "content": str(value.content)}
    return str(value)


def update_jsonable(update: dict | None) -> dict:
    """Serialize a node update dict, dropping the noisy `messages` key.

    LangGraph's `updates` stream mode reports a node that wrote to no
    observed output channel as `None` (see map_output_updates in
    langgraph.pregel._io) -- not every node write is guaranteed to land on a
    channel the graph declares as output, so this is a legitimate case, not
    an error.
    """
    if not update:
        return {}
    return {k: to_jsonable(v) for k, v in update.items() if k != "messages"}
