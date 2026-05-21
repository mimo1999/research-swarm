"""Shared helpers used across agent modules."""
from __future__ import annotations


def _field(obj, name: str, default=None):
    """Read an attribute from a Pydantic model or dict uniformly.

    Using ``getattr`` on a Pydantic model and ``dict.get`` on a plain dict
    avoids the repetitive ``hasattr(obj, x) else obj.get(x, default)``
    pattern that would otherwise appear dozens of times across the agent files.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _latest_verdicts(critiques: list) -> dict[str, str]:
    """Return the most recent verdict string for each finding_id.

    Because the critiques list is append-only, iterating in order and
    overwriting leaves the last verdict per finding when the loop ends.
    Verdict enum values are normalised to plain strings via ``.value``.
    """
    latest: dict[str, str] = {}
    for c in critiques:
        fid = _field(c, "finding_id", "")
        v = _field(c, "verdict", "")
        latest[fid] = v.value if hasattr(v, "value") else str(v)
    return latest
