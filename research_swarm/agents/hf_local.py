"""In-process Hugging Face ``transformers`` backend: provider ``"huggingface"``.

Runs the research stages' small model (default ``google/gemma-4-E2B-it``) inside the app process
instead of behind an Ollama daemon, for hosts that have a GPU but no Ollama, i.e. a Hugging Face
Space on ZeroGPU. Four parts:

- ``load()``: loads the model and tokenizer once per process. On ZeroGPU this must run at module
  level of the Space's app (``.to("cuda")`` outside a ``@spaces.GPU`` function is emulated and
  materialised when a GPU is attached).
- ``generate_batch(requests)``: one batched ``generate()`` over several prompts, each optionally
  constrained to its JSON schema (lm-format-enforcer), so a 2B model's output always parses the
  way Ollama's ``format=<schema>`` did. Arguments and results are plain picklable dicts because
  ZeroGPU runs the decorated function in a separate process.
- A per-session micro-batcher: the pipeline fans calls out concurrently (one scoring / extraction
  call per sub-question, verifier batches), and ZeroGPU charges each visitor's daily quota
  (2 min unauthenticated, 5 min free account) by GPU time. Running six concurrent requests as one
  batched ``generate`` costs about the time of the longest one instead of the sum.
- ``ChatHFLocal``: a LangChain chat model over the batcher, so every stage keeps calling
  ``llm.with_structured_output(Schema)`` exactly as with the other providers.

``set_gpu_runner`` lets the Space wrap ``generate_batch`` in ``spaces.GPU``; everywhere else the
plain function runs on whatever device the model was loaded to.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# ── Model state (one model per process) ───────────────────────────────────────

_state: dict[str, Any] = {}
# asyncio.wait_for raises asyncio.TimeoutError, a separate class from the builtin before 3.11 --
# and the Space's image runs 3.10.
_WAIT_TIMEOUT = (TimeoutError, asyncio.TimeoutError)
_load_lock = threading.Lock()


def loaded_model_id() -> str | None:
    return _state.get("model_id")


def load(model_id: str, device: str | None = None, dtype: str = "auto") -> None:
    """Load *model_id* once for this process (a no-op if it is already loaded).

    *device*: "cuda", "cpu", ... ; None picks cuda when available. Gemma 4 checkpoints are
    multimodal, so the loader tries the multimodal auto class first and falls back to a causal LM.
    The JSON-constraint tokenizer data (a pass over the whole vocabulary) is built here too, so it
    is paid once at startup rather than inside a GPU-billed call.
    """
    with _load_lock:
        if _state.get("model_id") == model_id:
            return
        import torch
        import transformers

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        templater: Any = tokenizer
        if not getattr(tokenizer, "chat_template", None):
            templater = transformers.AutoProcessor.from_pretrained(model_id)

        model = None
        errors = []
        for cls_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText",
                         "AutoModelForCausalLM"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(model_id, dtype=dtype)
                break
            except (ValueError, KeyError, OSError) as exc:
                errors.append(f"{cls_name}: {exc}")
        if model is None:
            raise RuntimeError(f"Could not load {model_id}: " + " | ".join(errors))
        model = model.to(device).eval()

        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
            eos_ids.append(tokenizer.eos_token_id)

        token_data = None
        try:
            token_data = _enforcer_tokenizer_data(tokenizer, eos_ids)
        except Exception:  # noqa: BLE001 - constraint is an accuracy aid, not a requirement
            logger.warning("lm-format-enforcer unavailable; JSON output is unconstrained",
                           exc_info=True)

        _state.update(model_id=model_id, model=model, tokenizer=tokenizer, templater=templater,
                      eos_ids=eos_ids, token_data=token_data, device=device)
        logger.info("Loaded %s on %s (JSON constraint: %s)", model_id, device,
                    token_data is not None)


def _enforcer_tokenizer_data(tokenizer: Any, eos_ids: list[int]) -> Any:
    """lm-format-enforcer's per-tokenizer tables, built without its transformers integration
    (which imports a module transformers 5 removed). Same construction as the integration's
    ``_build_regular_tokens_list``: each non-special token's text, and whether it starts a word
    (decoding it after "0" adds a leading space), decoded in batches to cut startup time on a
    262k-token vocabulary."""
    from lmformatenforcer import TokenEnforcerTokenizerData

    vocab_size = len(tokenizer)
    special = set(tokenizer.all_special_ids)
    ids = [i for i in range(vocab_size) if i not in special]
    token_0 = tokenizer.encode("0", add_special_tokens=False)[-1]
    alone = tokenizer.batch_decode([[i] for i in ids])
    after_0 = tokenizer.batch_decode([[token_0, i] for i in ids])
    regular = [(i, a0[1:], len(a0[1:]) > len(a)) for i, a, a0 in zip(ids, alone, after_0,
                                                                      strict=True)]

    def decode(tokens: list[int]) -> str:
        return tokenizer.decode(tokens).rstrip("�")

    return TokenEnforcerTokenizerData(regular, decode, eos_ids, False, vocab_size)


def render_prompt(messages: list[dict], enable_thinking: bool = False) -> str:
    """The chat-templated prompt text for *messages* ([{"role", "content"}, ...])."""
    templater = _state["templater"]
    try:
        return templater.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
    except TypeError:  # a template without the enable_thinking switch
        return templater.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── One batched, per-row constrained generate() ───────────────────────────────

def _row_constraint(schema: dict | None):
    """A (batch_id, token_ids) -> allowed-token-ids function forcing *schema*, or None."""
    token_data = _state.get("token_data")
    if not schema or token_data is None:
        return None
    try:
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer
        enforcer = TokenEnforcer(token_data, JsonSchemaParser(schema))
    except Exception:  # noqa: BLE001 - an unsupported schema construct: generate unconstrained
        logger.warning("Could not build a JSON constraint for this schema", exc_info=True)
        return None

    def allowed(_row: int, ids: Any) -> list[int]:
        tokens = enforcer.get_allowed_tokens(ids.tolist())
        return list(getattr(tokens, "allowed_tokens", tokens))

    return allowed


def _constraint_processor(fns: list, prompt_len: int, eos_ids: set[int]):
    """A logits processor applying each row's constraint (rows without one are untouched)."""
    import torch
    from transformers import LogitsProcessor

    class _RowConstraints(LogitsProcessor):
        def __call__(self, input_ids, scores):
            for row, fn in enumerate(fns):
                if fn is None:
                    continue
                generated = input_ids[row, prompt_len:]
                if generated.numel() and int(generated[-1]) in eos_ids:
                    continue  # finished row: generate() pads it regardless of its scores
                try:
                    allowed = fn(row, input_ids[row])
                except Exception:  # noqa: BLE001 - a confused enforcer must not kill the batch
                    logger.debug("JSON constraint failed on row %d; unconstraining it", row,
                                 exc_info=True)
                    fns[row] = None
                    continue
                if not allowed:
                    continue
                mask = torch.full_like(scores[row], float("-inf"))
                mask[torch.as_tensor(allowed, device=scores.device)] = 0
                scores[row] = scores[row] + mask
            return scores

    return _RowConstraints()


def generate_batch(requests: list[dict]) -> list[dict]:
    """Generate for every request in one ``generate()`` call.

    Each request: {"prompt": str (already chat-templated), "schema": dict | None,
    "max_new_tokens": int, "temperature": float}. Returns, per request, {"text",
    "input_tokens", "output_tokens"}. Plain dicts in and out, because ZeroGPU pickles both.
    """
    import torch

    model, tokenizer = _state["model"], _state["tokenizer"]
    eos_ids = set(_state["eos_ids"])
    enc = tokenizer([r["prompt"] for r in requests], return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    fns = [_row_constraint(r.get("schema")) for r in requests]
    temperature = max(float(r.get("temperature") or 0.0) for r in requests)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max(int(r.get("max_new_tokens") or 1024) for r in requests),
        "eos_token_id": sorted(eos_ids),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    if any(fns):
        from transformers import LogitsProcessorList
        kwargs["logits_processor"] = LogitsProcessorList(
            [_constraint_processor(fns, prompt_len, eos_ids)])
    with torch.inference_mode():
        out = model.generate(**enc, **kwargs)

    results = []
    for row, req in enumerate(requests):
        ids = out[row, prompt_len:].tolist()
        n = next((i for i, t in enumerate(ids) if t in eos_ids), len(ids))
        ids = ids[: min(n, int(req.get("max_new_tokens") or len(ids)))]
        results.append({
            "text": tokenizer.decode(ids, skip_special_tokens=True).strip(),
            "input_tokens": int(enc["attention_mask"][row].sum()),
            "output_tokens": len(ids),
        })
    return results


_runner: Callable[[list[dict]], list[dict]] = generate_batch


def set_gpu_runner(runner: Callable[[list[dict]], list[dict]]) -> None:
    """Route batches through *runner*, e.g. ``spaces.GPU(duration=...)(generate_batch)``."""
    global _runner
    _runner = runner


def estimate_gpu_seconds(requests: list[dict]) -> int:
    """A generous upper bound for ZeroGPU's per-call duration (quota is charged on actual time).

    Batched decode runs at roughly the speed of one row, so the longest ``max_new_tokens`` sets
    the bound; ~25 tokens/s is a conservative floor for a 5B model with constrained decoding.
    """
    from research_swarm.config import settings

    longest = max((int(r.get("max_new_tokens") or 1024) for r in requests), default=1024)
    return int(min(settings.hf_gpu_duration_max_s, 15 + longest / 25))


# ── Per-session micro-batcher ─────────────────────────────────────────────────

class GPUQuotaExceeded(RuntimeError):
    """The visitor's ZeroGPU quota ran out; the rest of the run's local calls fail fast."""


_quota_hit: set[str] = set()


def _is_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "quota" in text and "gpu" in text


def pop_quota_error(session_id: str) -> bool:
    """True (once) if *session_id*'s run hit the GPU quota; the Space reports it to the visitor."""
    if session_id in _quota_hit:
        _quota_hit.discard(session_id)
        return True
    return False


class _Batcher:
    """Collects concurrent requests from one session into batched runner calls.

    One batch runs at a time (one GPU). Keyed per session and created from inside that
    session's call, so a ZeroGPU call runs in (and is charged to) the visitor whose run made it;
    the worker exits once idle, and the next request starts a new one in its own context.
    """

    def __init__(self, window_s: float, max_batch: int, session_id: str = "",
                 idle_s: float = 30.0) -> None:
        self.session_id = session_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.window_s, self.max_batch, self.idle_s = window_s, max_batch, idle_s
        self.task: asyncio.Task | None = None

    async def submit(self, request: dict) -> dict:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((request, fut))
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._work())
        return await fut

    async def _work(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                first = await asyncio.wait_for(self.queue.get(), self.idle_s)
            except _WAIT_TIMEOUT:
                if self.queue.empty():
                    return
                continue
            batch = [first]
            deadline = loop.time() + self.window_s
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except _WAIT_TIMEOUT:
                    break
            batch = [(r, f) for r, f in batch if not f.cancelled()]
            if not batch:
                continue
            started = time.monotonic()
            try:
                results = await asyncio.to_thread(_runner, [r for r, _ in batch])
            except Exception as exc:  # noqa: BLE001 - delivered to every caller in the batch
                if _is_quota_error(exc):
                    _quota_hit.add(self.session_id)
                    exc = GPUQuotaExceeded(str(exc))
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            logger.debug("hf batch of %d took %.1fs", len(batch), time.monotonic() - started)
            for (_, fut), res in zip(batch, results, strict=True):
                if not fut.done():
                    fut.set_result(res)


_batchers: dict[tuple[int, str, str], _Batcher] = {}


def _event_key() -> str:
    """The current Gradio event id, if any. ZeroGPU finds the visitor to charge through Gradio's
    context variables, which a batcher's worker task captures when it is created; a HITL resume
    is a new event, so it must not reuse a worker left over from the first one."""
    try:
        from gradio.context import LocalContext
    except ImportError:
        return ""
    return str(LocalContext.event_id.get(None) or "")


async def submit(request: dict) -> dict:
    """Queue *request* on this session's batcher and wait for its result."""
    from research_swarm.config import settings
    from research_swarm.runtime.limits import current_llm_session

    session_id = current_llm_session.get() or ""
    if session_id in _quota_hit:
        raise GPUQuotaExceeded("GPU quota exhausted earlier in this run")
    key = (id(asyncio.get_running_loop()), session_id, _event_key())
    for stale in [k for k, b in _batchers.items()
                  if k != key and b.task is not None and b.task.done() and b.queue.empty()]:
        del _batchers[stale]   # finished sessions / events
    batcher = _batchers.get(key)
    if batcher is None:
        batcher = _batchers[key] = _Batcher(settings.hf_batch_window_s, settings.hf_max_batch,
                                            session_id)
    return await batcher.submit(request)


# ── LangChain chat model ──────────────────────────────────────────────────────

_ROLES = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}


def _to_chat(messages: list[BaseMessage]) -> list[dict]:
    chat = []
    for m in messages:
        role = next((r for cls, r in _ROLES.items() if isinstance(m, cls)), "user")
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        chat.append({"role": role, "content": content})
    return chat


class ChatHFLocal(BaseChatModel):
    """Chat model over the in-process transformers backend.

    ``reasoning`` and ``num_predict`` mirror ChatOllama's fields so ``base.without_thinking``
    (thinking off + output cap for structured stages) works unchanged.
    """

    model: str
    temperature: float = 0.0
    reasoning: bool | None = None
    num_predict: int | None = None
    default_max_new_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "huggingface-local"

    def _request(self, messages: list[BaseMessage], schema: dict | None) -> dict:
        if loaded_model_id() != self.model:
            load(self.model)
        return {
            "prompt": render_prompt(_to_chat(messages), enable_thinking=bool(self.reasoning)),
            "schema": schema,
            "max_new_tokens": self.num_predict or self.default_max_new_tokens,
            "temperature": self.temperature,
        }

    @staticmethod
    def _result(res: dict) -> ChatResult:
        message = AIMessage(content=res["text"], usage_metadata={
            "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            "total_tokens": res["input_tokens"] + res["output_tokens"],
        })
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None,
                  **kwargs: Any) -> ChatResult:
        return self._result(_runner([self._request(messages, kwargs.get("response_schema"))])[0])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: AsyncCallbackManagerForLLMRun | None = None,
                         **kwargs: Any) -> ChatResult:
        request = await asyncio.to_thread(self._request, messages, kwargs.get("response_schema"))
        return self._result(await submit(request))

    def with_structured_output(self, schema: Any, *, include_raw: bool = False,
                               **kwargs: Any) -> Runnable:
        """Constrain generation to *schema*'s JSON Schema and parse the reply into it.

        A reply that still fails to parse raises ``OutputParserException`` carrying the raw text
        (``llm_output``), which is what ``_utils.recover_from_parse_failure`` repairs from.
        """
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise NotImplementedError("ChatHFLocal.with_structured_output needs a Pydantic model")
        if include_raw:
            raise NotImplementedError("include_raw is not supported")

        def parse(message: AIMessage) -> BaseModel:
            text = message.content if isinstance(message.content, str) else str(message.content)
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                raise OutputParserException(
                    f"Failed to parse {schema.__name__} from completion {text}. Got: {exc}",
                    llm_output=text,
                ) from exc

        return self.bind(response_schema=schema.model_json_schema()) | RunnableLambda(parse)
