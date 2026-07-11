from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, Form, UploadFile

from research_swarm.rag.indexes import get_embed_model
from research_swarm.rag.ingestion import IngestionPipeline

router = APIRouter(prefix="/api/sessions")


@router.post("/{session_id}/documents")
async def ingest_documents(session_id: str, files: list[UploadFile] | None = None, urls: str = Form("")):
    files = files or []
    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    if not files and not url_list:
        return {"chunks_added": 0}

    pipeline = IngestionPipeline(session_id)
    embed = get_embed_model()
    total = 0

    for uf in files:
        suffix = os.path.splitext(uf.filename or "")[1] or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await uf.read())
            tmp_path = tmp.name
        try:
            total += pipeline.ingest_pdf(tmp_path, embed)
        finally:
            os.unlink(tmp_path)

    for url in url_list:
        total += pipeline.ingest_url(url, embed)

    return {"chunks_added": total}
