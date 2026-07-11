from __future__ import annotations

import httpx
from fastapi import APIRouter

from research_swarm.config import settings

router = APIRouter(prefix="/api/config")

_ANTHROPIC_MODELS = ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-3-5"]
_OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
_OLLAMA_CLOUD_MODELS = [
    "minimax-m2.5:cloud",
    "gpt-oss:120b-cloud",
    "qwen3-coder-next:cloud",
    "qwen3.5:cloud",
    "minimax-m2.7:cloud",
]


@router.get("/options")
async def get_options():
    return {
        "providers": ["ollama", "anthropic", "openai"],
        "models": {
            "anthropic": _ANTHROPIC_MODELS,
            "openai": _OPENAI_MODELS,
            "ollama_cloud": _OLLAMA_CLOUD_MODELS,
        },
        "depths": ["shallow", "standard", "deep"],
        "defaults": {
            "provider": settings.default_model_provider,
            "model": settings.default_model_name,
            "max_sources": settings.max_sources,
            "ollama_url": settings.ollama_base_url,
            "ollama_model": settings.ollama_model,
            "ollama_cloud_model": settings.ollama_cloud_model,
            "ollama_deployment": settings.ollama_deployment,
        },
    }


async def _ollama_get(base_url: str, path: str, timeout: float = 2.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"{base_url.rstrip('/')}{path}")
        resp.raise_for_status()
        return resp.json()


@router.get("/ollama/status")
async def ollama_status(url: str, model: str, deployment: str = "local"):
    try:
        data = await _ollama_get(url, "/api/tags")
        names = [t.get("name", "") for t in data.get("models", [])]
        model_pulled = model in names
    except Exception as exc:
        return {
            "reachable": False,
            "model_pulled": False,
            "logged_in": None,
            "message": f"Cannot reach Ollama daemon at {url} — {exc}",
        }

    if deployment == "local":
        message = (
            f"Daemon reachable · {model} is pulled and ready"
            if model_pulled
            else f"Daemon reachable but {model} is not pulled. Run `ollama pull {model}`."
        )
        return {"reachable": True, "model_pulled": model_pulled, "logged_in": None, "message": message}

    # cloud deployment
    username = None
    whoami_unsupported = False
    try:
        who = await _ollama_get(url, "/api/whoami")
        username = who.get("username") or who.get("name")
    except Exception as exc:
        whoami_unsupported = "404" in str(exc) or "Not Found" in str(exc)

    logged_in = True if username else (None if whoami_unsupported else False)
    if model_pulled:
        message = f"Daemon reachable · {model} available locally"
    elif logged_in is False:
        message = f"Daemon reachable · {model} not pulled. Log in with `ollama login` or pull it locally."
    else:
        message = f"Daemon reachable · {model} will stream from ollama.com at inference time."

    return {"reachable": True, "model_pulled": model_pulled, "logged_in": logged_in, "message": message}
