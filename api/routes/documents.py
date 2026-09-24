from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, Form, HTTPException, UploadFile

from api.runs import get_run
from research_swarm.tools.pdf_loader import load_pdf
from research_swarm.tools.url_fetcher import fetch_url

router = APIRouter(prefix="/api/sessions")


@router.post("/{session_id}/documents")
async def ingest_documents(
    session_id: str, files: list[UploadFile] | None = None, urls: str = Form(""),
):
    """Extract full text from uploaded PDFs / URLs onto the pending run.

    The text is stored on the run's initial state as ``ingested_documents``;
    the graph's document pass extracts claims from each document in one call.
    No chunking, no embedding, no vector store. Must be called between
    ``POST /api/research`` and the first ``/stream`` connect.
    """
    files = files or []
    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    if not files and not url_list:
        return {"documents_added": 0}

    run = await get_run(session_id)
    if run is None or run.initial_state is None:
        raise HTTPException(status_code=409, detail="Run already started or unknown session")

    documents: list[dict] = []

    for uf in files:
        suffix = os.path.splitext(uf.filename or "")[1] or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await uf.read())
            tmp_path = tmp.name
        try:
            result = load_pdf.invoke({"file_path": tmp_path})
            text = "\n\n".join(c["text"] for c in result.get("chunks", []) if c.get("text"))
            if text:
                documents.append({
                    "url": uf.filename or result.get("url", tmp_path),
                    "title": result.get("title", "") or (uf.filename or ""),
                    "text": text,
                    "source_type": "pdf",
                })
        finally:
            os.unlink(tmp_path)

    for url in url_list:
        result = fetch_url.invoke({"url": url, "max_chars": 20000})
        snippet = result.get("snippet", "")
        if snippet.startswith("["):
            continue  # fetch error placeholder
        documents.append({
            "url": result.get("url", url),
            "title": result.get("title", ""),
            "text": snippet,
            "source_type": "web",
        })

    run.initial_state["ingested_documents"] = (
        run.initial_state.get("ingested_documents") or []
    ) + documents
    return {"documents_added": len(documents)}
