"""Shared helpers used across agent modules."""
from __future__ import annotations

import json


def _field(obj, name: str, default=None):
    """Read an attribute from a Pydantic model or dict uniformly.

    Using ``getattr`` on a Pydantic model and ``dict.get`` on a plain dict
    avoids the repetitive ``hasattr(obj, x) else obj.get(x, default)``
    pattern that would otherwise appear dozens of times across the agent files.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def json_output_instruction(example: dict) -> str:
    """Return a system-prompt suffix that forces raw JSON output.

    Works as a belt-and-suspenders measure alongside ``with_structured_output``:
    - ``with_structured_output`` enforces the schema at the API/parsing layer
    - This instruction enforces it at the prompt layer so models that ignore
      the schema parameter (common with Ollama cloud models) still comply.
    """
    return (
        "\n\n"
        "OUTPUT FORMAT — CRITICAL:\n"
        "Your entire response MUST be a single raw JSON object.\n"
        "Do NOT use markdown, code fences (```), bullet points, or any prose.\n"
        "Do NOT include any text before or after the JSON.\n"
        "Required structure (replace values, keep all keys):\n"
        f"{json.dumps(example, indent=2)}"
    )


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
