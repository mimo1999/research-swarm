"""PDF loader tool -- extracts text from a local PDF file with page metadata."""
import textwrap

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pypdf import PdfReader

from research_swarm.utils.security import validate_file_path


class PDFChunk(BaseModel):
    page: int = Field(..., description="1-based page number")
    text: str = Field(..., description="Extracted text from this page")
    char_count: int = Field(..., description="Number of characters on this page")


class PDFLoaderInput(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to the PDF file")
    max_pages: int = Field(default=50, ge=1, le=500, description="Maximum pages to read")
    snippet_chars: int = Field(
        default=500, ge=50, le=5000, description="Characters per page included in snippet"
    )


class PDFLoadResult(BaseModel):
    url: str
    title: str
    total_pages: int
    chunks: list[PDFChunk]
    snippet: str
    source_type: str = "pdf"
    credibility_score: float = 0.7


@tool("load_pdf", args_schema=PDFLoaderInput)
def load_pdf(file_path: str, max_pages: int = 50, snippet_chars: int = 500) -> dict:
    """Extract text from a local PDF file.

    Returns a dict matching the PDFLoadResult schema: includes per-page chunks
    and a combined Source-compatible snippet from the first few pages.
    """
    # Security: reject paths that escape allowed directories (path traversal).
    try:
        path = validate_file_path(file_path)
    except ValueError as exc:
        return PDFLoadResult(
            url=str(file_path),
            title="",
            total_pages=0,
            chunks=[],
            snippet=f"[Access denied: {exc}]",
            credibility_score=0.0,
        ).model_dump(mode="json")

    if not path.exists():
        return PDFLoadResult(
            url=str(path),
            title="",
            total_pages=0,
            chunks=[],
            snippet=f"[File not found: {file_path}]",
            credibility_score=0.0,
        ).model_dump(mode="json")

    try:
        reader = PdfReader(str(path))
        pages_to_read = min(max_pages, len(reader.pages))
        chunks: list[PDFChunk] = []

        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text() or ""
            page_text = page_text.strip()
            chunks.append(
                PDFChunk(page=i + 1, text=page_text, char_count=len(page_text))
            )

        # Build a combined snippet from the first few pages
        combined = " ".join(c.text for c in chunks[:5])
        snippet = textwrap.shorten(combined, width=snippet_chars * 3, placeholder="...")

        # Attempt to extract title from PDF metadata
        meta = reader.metadata or {}
        title = str(meta.get("/Title", "")).strip() or path.stem

        return PDFLoadResult(
            url=path.as_uri(),
            title=title,
            total_pages=len(reader.pages),
            chunks=chunks,
            snippet=snippet,
        ).model_dump(mode="json")

    except Exception as exc:
        return PDFLoadResult(
            url=str(path),
            title="",
            total_pages=0,
            chunks=[],
            snippet=f"[PDF read error: {type(exc).__name__}: {exc}]",
            credibility_score=0.0,
        ).model_dump(mode="json")
