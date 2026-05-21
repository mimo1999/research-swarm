"""Unit tests for Phase 2 tools — all network calls are mocked."""
import io
from unittest.mock import MagicMock, patch

from pypdf import PdfWriter

from research_swarm.schemas.source import SourceType

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Create a minimal in-memory PDF with one text annotation per page."""
    writer = PdfWriter()
    for _ in pages:
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ── web_search ────────────────────────────────────────────────────────────────

class TestWebSearch:
    def _make_tavily_result(self, url: str, title: str, content: str) -> dict:
        return {"url": url, "title": title, "content": content}

    @patch("research_swarm.tools.web_search.TavilyClient")
    def test_returns_list_of_source_dicts(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                self._make_tavily_result(
                    "https://example.edu/paper",
                    "Great Paper",
                    "Some content here",
                )
            ]
        }

        from research_swarm.tools.web_search import web_search
        result = web_search.invoke({"query": "AI safety", "max_results": 1})

        assert isinstance(result, list)
        assert len(result) == 1
        src = result[0]
        assert src["url"] == "https://example.edu/paper"
        assert src["title"] == "Great Paper"
        assert src["source_type"] == SourceType.web.value
        assert src["credibility_score"] == 0.85  # .edu domain

    @patch("research_swarm.tools.web_search.TavilyClient")
    def test_empty_results(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = {"results": []}

        from research_swarm.tools.web_search import web_search
        result = web_search.invoke({"query": "obscure topic"})
        assert result == []

    @patch("research_swarm.tools.web_search.TavilyClient")
    def test_credibility_low_for_social_media(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                self._make_tavily_result("https://reddit.com/r/foo", "Reddit", "post")
            ]
        }

        from research_swarm.tools.web_search import web_search
        result = web_search.invoke({"query": "test"})
        assert result[0]["credibility_score"] == 0.3

    @patch("research_swarm.tools.web_search.TavilyClient")
    def test_snippet_truncated_at_1000_chars(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                self._make_tavily_result(
                    "https://example.com", "Title", "x" * 2000
                )
            ]
        }

        from research_swarm.tools.web_search import web_search
        result = web_search.invoke({"query": "test"})
        assert len(result[0]["snippet"]) <= 1000

    @patch("research_swarm.tools.web_search.TavilyClient")
    def test_gov_domain_gets_high_credibility(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [self._make_tavily_result("https://cdc.gov/health", "CDC", "info")]
        }

        from research_swarm.tools.web_search import web_search
        result = web_search.invoke({"query": "health"})
        assert result[0]["credibility_score"] == 0.85

    @patch("research_swarm.tools.web_search.TavilyClient")
    def test_generic_domain_gets_default_credibility(self, mock_client_cls):
        """URLs that are neither .edu/.gov nor social media should score 0.6."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [self._make_tavily_result("https://techblog.com/post", "Blog", "text")]
        }

        from research_swarm.tools.web_search import web_search
        result = web_search.invoke({"query": "tech"})
        assert result[0]["credibility_score"] == 0.6

    @patch("research_swarm.tools.web_search.TavilyClient")
    def test_tavily_exception_returns_error_source(self, mock_client_cls):
        """If TavilyClient raises, return a trace-visible error source."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.side_effect = RuntimeError("API key invalid")

        from research_swarm.tools.web_search import web_search
        result = web_search.invoke({"query": "test"})
        assert len(result) == 1
        assert result[0]["credibility_score"] == 0.0
        assert "Search error" in result[0]["snippet"]


# ── arxiv_search ──────────────────────────────────────────────────────────────

class TestArxivSearch:
    def _mock_paper(self, title: str, summary: str, entry_id: str) -> MagicMock:
        paper = MagicMock()
        paper.title = title
        paper.summary = summary
        paper.entry_id = entry_id
        paper.pdf_url = "https://arxiv.org/pdf/0000.00001v1"
        return paper

    @patch("research_swarm.tools.arxiv_tool.arxiv.Client")
    def test_returns_arxiv_sources(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.results.return_value = [
            self._mock_paper("LLMs in Science", "Abstract text here.", "http://arxiv.org/abs/2301.00001")
        ]

        from research_swarm.tools.arxiv_tool import arxiv_search
        result = arxiv_search.invoke({"query": "LLM science", "max_results": 1})

        assert len(result) == 1
        src = result[0]
        assert src["source_type"] == SourceType.arxiv.value
        assert src["credibility_score"] == 0.85
        assert "LLMs in Science" == src["title"]

    @patch("research_swarm.tools.arxiv_tool.arxiv.Client")
    def test_snippet_is_shortened_abstract(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        long_abstract = "word " * 300
        mock_client.results.return_value = [
            self._mock_paper("Paper", long_abstract, "http://arxiv.org/abs/0001")
        ]

        from research_swarm.tools.arxiv_tool import arxiv_search
        result = arxiv_search.invoke({"query": "test"})
        assert len(result[0]["snippet"]) <= 600

    @patch("research_swarm.tools.arxiv_tool.arxiv.Client")
    def test_empty_results_returns_empty_list(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.results.return_value = []  # no papers found

        from research_swarm.tools.arxiv_tool import arxiv_search
        result = arxiv_search.invoke({"query": "obscure topic no one researches"})
        assert result == []

    @patch("research_swarm.tools.arxiv_tool.arxiv.Client")
    def test_no_pdf_fetch_by_default(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.results.return_value = [
            self._mock_paper("Paper", "Abstract.", "http://arxiv.org/abs/0001")
        ]

        with patch("research_swarm.tools.arxiv_tool._extract_pdf_snippet") as mock_pdf:
            from research_swarm.tools.arxiv_tool import arxiv_search
            arxiv_search.invoke({"query": "test", "fetch_pdf_text": False})
            mock_pdf.assert_not_called()


# ── url_fetcher ───────────────────────────────────────────────────────────────

class TestURLFetcher:
    def _mock_response(self, html: str, status: int = 200, content_type: str = "text/html"):
        resp = MagicMock()
        resp.status_code = status
        resp.text = html
        resp.headers = {"content-type": content_type}
        resp.raise_for_status = MagicMock()
        return resp

    @patch("research_swarm.tools.url_fetcher.httpx.get")
    def test_extracts_article_text(self, mock_get):
        html = """<html><head><title>Test Page</title></head>
        <body><article><p>Main article content here.</p></article>
        <nav>Navigation cruft</nav></body></html>"""
        mock_get.return_value = self._mock_response(html)

        from research_swarm.tools.url_fetcher import fetch_url
        result = fetch_url.invoke({"url": "https://example.com/article"})

        assert result["title"] == "Test Page"
        assert "Main article content" in result["snippet"]
        assert "Navigation cruft" not in result["snippet"]

    @patch("research_swarm.tools.url_fetcher.httpx.get")
    def test_non_html_returns_note(self, mock_get):
        mock_get.return_value = self._mock_response(
            "", content_type="application/pdf"
        )

        from research_swarm.tools.url_fetcher import fetch_url
        result = fetch_url.invoke({"url": "https://example.com/file.pdf"})
        assert "Non-HTML" in result["snippet"]
        assert result["credibility_score"] == 0.5

    @patch("research_swarm.tools.url_fetcher.httpx.get")
    def test_http_error_returns_error_source(self, mock_get):
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"
        mock_get.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=mock_response
        )

        from research_swarm.tools.url_fetcher import fetch_url
        result = fetch_url.invoke({"url": "https://example.com/missing"})
        assert "404" in result["snippet"]
        assert result["credibility_score"] == 0.0

    @patch("research_swarm.tools.url_fetcher.httpx.get")
    def test_body_fallback_when_no_article_tag(self, mock_get):
        """When the page has no <article> tag, content should come from <body>."""
        html = """<html><head><title>Plain Page</title></head>
        <body><p>Body content here without an article wrapper.</p></body></html>"""
        mock_get.return_value = self._mock_response(html)

        from research_swarm.tools.url_fetcher import fetch_url
        result = fetch_url.invoke({"url": "https://example.com/plain"})

        assert result["title"] == "Plain Page"
        assert "Body content here" in result["snippet"]

    @patch("research_swarm.tools.url_fetcher.httpx.get")
    def test_snippet_respects_max_chars(self, mock_get):
        html = f"<html><body><article>{'word ' * 2000}</article></body></html>"
        mock_get.return_value = self._mock_response(html)

        from research_swarm.tools.url_fetcher import fetch_url
        result = fetch_url.invoke({"url": "https://example.com", "max_chars": 500})
        assert len(result["snippet"]) <= 510  # slight slack for placeholder


# ── pdf_loader ────────────────────────────────────────────────────────────────

class TestPDFLoader:
    def test_missing_file_returns_error(self, tmp_path):
        from research_swarm.tools.pdf_loader import load_pdf
        result = load_pdf.invoke({"file_path": str(tmp_path / "nonexistent.pdf")})
        assert "File not found" in result["snippet"]
        assert result["credibility_score"] == 0.0
        assert result["total_pages"] == 0

    def test_valid_pdf_returns_chunks(self, tmp_path):
        # Write a real minimal PDF
        pdf_path = tmp_path / "test.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_blank_page(width=612, height=792)
        with open(pdf_path, "wb") as f:
            writer.write(f)

        from research_swarm.tools.pdf_loader import load_pdf
        result = load_pdf.invoke({"file_path": str(pdf_path)})

        assert result["total_pages"] == 2
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) == 2
        assert result["source_type"] == "pdf"

    def test_respects_max_pages(self, tmp_path):
        pdf_path = tmp_path / "big.pdf"
        writer = PdfWriter()
        for _ in range(10):
            writer.add_blank_page(width=612, height=792)
        with open(pdf_path, "wb") as f:
            writer.write(f)

        from research_swarm.tools.pdf_loader import load_pdf
        result = load_pdf.invoke({"file_path": str(pdf_path), "max_pages": 3})
        assert len(result["chunks"]) == 3


# ── retriever_tool ────────────────────────────────────────────────────────────

class TestRetrieverTool:
    def test_stub_returns_empty_list(self):
        from research_swarm.tools.retriever_tool import retriever_stub
        result = retriever_stub.invoke({"query": "anything", "session_id": "sess-1"})
        assert result == []

    def test_make_retriever_tool_wraps_engine(self):
        from research_swarm.tools.retriever_tool import make_retriever_tool

        mock_node = MagicMock()
        mock_node.metadata = {"url": "https://example.com", "title": "Doc"}
        mock_node.get_content.return_value = "Relevant chunk text"
        mock_node.score = 0.9

        mock_response = MagicMock()
        mock_response.source_nodes = [mock_node]

        mock_engine = MagicMock()
        mock_engine.query.return_value = mock_response

        tool_fn = make_retriever_tool(lambda session_id: mock_engine)
        result = tool_fn.invoke({"query": "test query", "session_id": "s1", "top_k": 1})

        assert len(result) == 1
        assert result[0]["url"] == "https://example.com"
        assert result[0]["snippet"] == "Relevant chunk text"
        assert result[0]["source_type"] == SourceType.retriever.value
        assert result[0]["credibility_score"] == 0.9

    def test_make_retriever_tool_caps_at_top_k(self):
        from research_swarm.tools.retriever_tool import make_retriever_tool

        nodes = []
        for i in range(10):
            node = MagicMock()
            node.metadata = {"url": f"https://example.com/{i}"}
            node.get_content.return_value = f"chunk {i}"
            node.score = 0.5
            nodes.append(node)

        mock_response = MagicMock()
        mock_response.source_nodes = nodes
        mock_engine = MagicMock()
        mock_engine.query.return_value = mock_response

        tool_fn = make_retriever_tool(lambda session_id: mock_engine)
        result = tool_fn.invoke({"query": "test", "session_id": "s1", "top_k": 3})
        assert len(result) == 3
