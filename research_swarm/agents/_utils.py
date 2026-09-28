"""Shared helpers used across agent modules."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from research_swarm.runtime.limits import current_llm_session, llm_slot
from research_swarm.runtime.trace import trace_event

logger = logging.getLogger(__name__)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _field(obj, name: str, default=None):
    """Read an attribute from a Pydantic model or dict uniformly.

    Using ``getattr`` on a Pydantic model and ``dict.get`` on a plain dict
    avoids the repetitive ``hasattr(obj, x) else obj.get(x, default)``
    pattern that would otherwise appear dozens of times across the agent files.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _strip_schema_titles(node):
    """Drop Pydantic's auto-generated ``"title": "<str>"`` metadata from a JSON schema.

    Property names (dict-valued keys under ``properties``) are untouched — only
    string-valued ``title`` entries, which are pure metadata, are removed.
    """
    if isinstance(node, dict):
        return {
            k: _strip_schema_titles(v)
            for k, v in node.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_strip_schema_titles(v) for v in node]
    return node


def schema_output_instruction(schema_class: type) -> str:
    """Return a system-prompt suffix that injects the full Pydantic schema.

    Works as a belt-and-suspenders measure alongside ``with_structured_output``:
    - ``with_structured_output`` enforces the schema at the API/parsing layer
    - This instruction enforces it at the prompt layer so models that ignore
      the schema parameter (common with Ollama cloud models) still comply.

    Unlike the old ``json_output_instruction`` (hand-written example), this
    generates the instruction from ``model_json_schema()`` so it stays in sync
    with field names, types, descriptions, and constraints automatically.

    The schema is serialised compactly (no indentation, no auto-generated
    ``title`` metadata) — this suffix rides on every LLM call, so its size
    directly multiplies prompt-token cost across the whole session.
    """
    schema = _strip_schema_titles(schema_class.model_json_schema())
    return (
        "\n\n"
        "OUTPUT FORMAT — CRITICAL:\n"
        "Your entire response MUST be a single raw JSON object.\n"
        "Do NOT use markdown, code fences (```), bullet points, or any prose.\n"
        "Do NOT include any text before or after the JSON.\n"
        "Your response must conform to this JSON Schema:\n"
        f"{json.dumps(schema, separators=(',', ':'))}"
    )


def json_output_instruction(example: dict) -> str:
    """Return a system-prompt suffix that forces raw JSON output (legacy).

    Prefer ``schema_output_instruction`` for new code — it generates the
    instruction from the Pydantic schema rather than a hand-written example.
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


_COMPLETION_RE = re.compile(r"from completion (.*)\.\s*Got:", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding ```json ... ``` (or bare ```) code fence, if present."""
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _extract_json_object(text: str) -> str:
    """Trim to the outermost {...} span, in case of leading/trailing prose."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def repair_json_brackets(text: str) -> str:
    """Close brackets a JSON text left unbalanced: where a closer does not match the innermost
    open bracket, the missing closers are inserted first; a closer with no opener is dropped; any
    brackets still open at the end are closed. String contents (with escapes) are left alone.

    For unconstrained replies (Ollama Cloud did not enforce the JSON schema for gemma4:31b-cloud)
    that were otherwise complete: one reply closed a nested object but forgot the list around it.
    """
    closer_of = {"{": "}", "[": "]"}
    out: list[str] = []
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in closer_of:
            stack.append(closer_of[ch])
        elif ch in "}]":
            if ch not in stack:
                continue
            while stack[-1] != ch:
                out.append(stack.pop())
            stack.pop()
        out.append(ch)
    out.extend(reversed(stack))
    return "".join(out)


def _repair_json_backslashes(text: str) -> str:
    r"""Escape backslashes that aren't already a valid JSON escape sequence.

    Structured-output models occasionally emit raw LaTeX (``\in``, ``\mathbb``,
    ``\times``, ``\text{...}``) inside a JSON string value instead of escaping
    the backslash, which breaks ``json.loads`` with "Invalid \escape". A
    backslash is left alone only when it's followed by another backslash or a
    double-quote (i.e. already escaped); every other backslash is doubled.

    This deliberately does NOT treat \n/\t/\r/\b/\f/\u as "already valid"
    escapes and leave them alone: several extremely common LaTeX commands --
    \text, \theta, \tau, \times, \top, \underline, \begin, \frac, \right --
    start with exactly the letters n/t/r/b/f/u that would make them look like
    one of those escapes. Treating them as legitimate would silently corrupt
    the content (e.g. \text becomes a literal tab character followed by
    "ext") instead of fixing the parse failure. Content in this domain
    (research claims) essentially never intentionally escapes a literal
    tab/newline/unicode point inside a claim string, so this trade-off favours
    the case that's actually breaking.
    """
    return re.sub(r'\\(?!["\\])', r"\\\\", text)


def _recover_from_bad_escapes(exc: Exception, model_class: type[_ModelT]) -> _ModelT | None:
    """Recover from a JSON parse failure caused by unescaped backslashes.

    Uses ``OutputParserException.llm_output`` -- the raw text LangChain's JSON
    parser attaches to the exception it raises (see
    ``langchain_core.output_parsers.json.py``) -- rather than regex-matching
    the exception's string message, which is what the schema-echo recovery
    below does for a different failure shape. Returns None (never raises) so
    callers can fall through to their own default, same as a cache miss.
    """
    raw_output = getattr(exc, "llm_output", None)
    if not raw_output:
        return None
    candidate = _extract_json_object(_strip_code_fence(raw_output))
    repaired = _repair_json_backslashes(candidate)
    try:
        data = json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return model_class(**data)
    except Exception:
        return None


def recover_from_parse_failure(exc: Exception, model_class: type[_ModelT]) -> _ModelT | None:
    """Best-effort recovery when a ``with_structured_output`` call fails to parse.

    Tries two independent failure shapes, in order:

    1. Unescaped backslashes (raw LaTeX in a claim) breaking ``json.loads`` --
       see ``_recover_from_bad_escapes`` / ``_repair_json_backslashes``.
    2. Observed with gemma4:31b-cloud: instead of a flat instance, the model
       echoes the JSON *schema* back -- e.g. ``{"properties": {"claim": "...",
       "confidence": 0.95}, "required": [...], "type": "object"}``. Pydantic
       correctly rejects this shape, but the actual synthesized content (the
       claim, the score) is sitting right there in the raw completion, one
       level down.

    Falling back to a generic placeholder in either case throws away real
    work the model already did. This recovers it when possible.

    Returns a validated ``model_class`` instance, or None if nothing usable
    could be recovered — callers should fall back to their own default in
    that case, exactly as if this function didn't exist.
    """
    recovered = _recover_from_bad_escapes(exc, model_class)
    if recovered is not None:
        return recovered

    match = _COMPLETION_RE.search(str(exc))
    if not match:
        return None
    try:
        raw = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    candidates = [raw]
    if isinstance(raw.get("properties"), dict):
        candidates.append(raw["properties"])

    field_names = set(model_class.model_fields)
    for candidate in candidates:
        payload = {k: v for k, v in candidate.items() if k in field_names}
        if not payload:
            continue
        try:
            return model_class(**payload)
        except Exception:
            continue
    return None


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


# ---------------------------------------------------------------------------
# Retry with backoff for transient provider errors
# ---------------------------------------------------------------------------

_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_TRANSIENT_TEXT = (
    "too many concurrent requests",
    "concurrent request slot",
    "rate limit",
    "overloaded",
    "temporarily unavailable",
    "timed out",
    "connection reset",
    "connection error",
)


def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying: rate limits, 5xx, timeouts and dropped connections.

    Recognises the status code however the client library exposes it (``status_code`` on
    Ollama/Anthropic/OpenAI errors, ``response.status_code`` on httpx errors) and falls
    back to the message text, since some wrappers only keep the string. Parse/validation
    failures are deliberately NOT transient -- retrying the same prompt rarely fixes those,
    and callers already have ``recover_from_parse_failure`` for them.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_TEXT)


async def ainvoke_with_retry(
    runnable: Any,
    messages: Any,
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    session_id: str | None = None,
    agent: str = "",
) -> Any:
    """``await runnable.ainvoke(messages)`` under the process-wide LLM slot, retrying transient
    errors with jittered backoff.

    Every LLM call in the pipeline goes through here. The slot (``llm_slot``, capped per
    provider) is held only while a request is in flight and is released during backoff sleeps,
    so a retrying call never holds capacity another stage could use. Backoff waits ~base,
    2*base, 4*base ... (capped at *max_delay*, each scaled by a random 0.5-1.0 so parallel
    workers don't retry in lockstep). Non-transient errors, and the last attempt's error,
    propagate unchanged. Retries and long slot waits are logged and traced. Each attempt is a
    real LLM call and counts against the session's call budget.

    *session_id* defaults to the session recorded when the node built its LLM
    (``runtime.limits.set_llm_context``).
    """
    sid = session_id if session_id is not None else current_llm_session.get()
    for attempt in range(1, attempts + 1):
        try:
            async with llm_slot() as waited:
                if waited >= 1.0:
                    trace_event(sid, agent or "llm", "note", slot_wait_s=round(waited, 2))
                return await runnable.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts or not is_transient(exc):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            logger.warning(
                "%s: transient LLM error (%s: %s) -- retry %d/%d in %.1fs",
                agent or "LLM call", type(exc).__name__, str(exc)[:120],
                attempt, attempts - 1, delay,
            )
            trace_event(
                sid, agent or "llm", "note", retry=attempt, delay=round(delay, 2),
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover
