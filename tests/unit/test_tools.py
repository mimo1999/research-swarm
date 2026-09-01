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
    def setup_method(self):
        # Reset the module-level TavilyClient singleton so each test gets a
        # fresh mock when patching TavilyClient.
        #
        # NOTE: `import research_swarm.tools.web_search as ws` resolves to the
        # StructuredTool object (because __init__.py shadows the submodule name)
        # rather than the module.  Use sys.modules to get the real module object.
        import sys
        ws = sys.modules.get("research_swarm.tools.web_search")
        if ws is not None:
            ws._tavily_client = None
            ws._tavily_key_cache = ""

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

    def test_is_configured_reflects_settings_key(self, monkeypatch):
        from pydantic import SecretStr

        from research_swarm.config import settings
        from research_swarm.tools.web_search import is_configured

        monkeypatch.setattr(settings, "tavily_api_key", SecretStr(""))
        assert is_configured() is False

        monkeypatch.setattr(settings, "tavily_api_key", SecretStr("tvly-real-key"))
        assert is_configured() is True


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


# ── pubmed_search ────────────────────────────────────────────────────────────

_EFETCH_XML_TEMPLATE = """\
<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>{pmid}</PMID>
      <Article>
        <ArticleTitle>{title}</ArticleTitle>
        <Abstract>
          <AbstractText>{abstract}</AbstractText>
        </Abstract>
        <Journal><Title>{journal}</Title></Journal>
      </Article>
    </MedlineCitation>
    <PubDate><Year>{year}</Year></PubDate>
    <PubmedData>
      <History>
        <PubMedPubDate><Year>{year}</Year></PubMedPubDate>
      </History>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def _mock_esearch_response(pmids: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"esearchresult": {"idlist": pmids}}
    return resp


def _mock_efetch_response(xml: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = xml.encode("utf-8")
    return resp


class TestPubmedSearch:
    @patch("research_swarm.tools.pubmed_tool.httpx.get")
    def test_returns_list_of_source_dicts(self, mock_get):
        xml = _EFETCH_XML_TEMPLATE.format(
            pmid="12345", title="GLP-1 in Parkinson's Disease",
            abstract="A phase 2 trial found improvement.", journal="Lancet", year="2025",
        )
        mock_get.side_effect = [
            _mock_esearch_response(["12345"]),
            _mock_efetch_response(xml),
        ]

        from research_swarm.tools.pubmed_tool import pubmed_search
        result = pubmed_search.invoke({"query": "GLP-1 Parkinson's disease", "max_results": 1})

        assert len(result) == 1
        src = result[0]
        assert src["source_type"] == SourceType.pubmed.value
        assert src["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345"
        assert "GLP-1 in Parkinson's Disease" in src["title"]
        assert "improvement" in src["snippet"]
        assert src["credibility_score"] == 0.9

    @patch("research_swarm.tools.pubmed_tool.httpx.get")
    def test_empty_esearch_results_returns_no_results_placeholder(self, mock_get):
        mock_get.side_effect = [_mock_esearch_response([])]

        from research_swarm.tools.pubmed_tool import pubmed_search
        result = pubmed_search.invoke({"query": "an extremely obscure query"})

        assert len(result) == 1
        assert result[0]["source_type"] == SourceType.pubmed.value
        assert result[0]["credibility_score"] == 0.0

    @patch("research_swarm.tools.pubmed_tool.httpx.get")
    def test_articles_without_abstract_are_skipped(self, mock_get):
        # e.g. letters/errata: PMID present but no <AbstractText>
        xml = """\
<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>999</PMID>
      <Article>
        <ArticleTitle>Erratum</ArticleTitle>
        <Journal><Title>Lancet</Title></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""
        mock_get.side_effect = [
            _mock_esearch_response(["999"]),
            _mock_efetch_response(xml),
        ]

        from research_swarm.tools.pubmed_tool import pubmed_search
        result = pubmed_search.invoke({"query": "test"})

        # No usable abstract -- falls through to the "no results" placeholder,
        # not a Source built from an empty snippet.
        assert len(result) == 1
        assert result[0]["credibility_score"] == 0.0

    @patch("research_swarm.tools.pubmed_tool.httpx.get")
    def test_network_error_returns_error_placeholder(self, mock_get):
        mock_get.side_effect = Exception("connection reset")

        from research_swarm.tools.pubmed_tool import pubmed_search
        result = pubmed_search.invoke({"query": "test"})

        assert len(result) == 1
        assert result[0]["credibility_score"] == 0.0
        assert "Search error" in result[0]["snippet"]

    @patch("research_swarm.tools.pubmed_tool.httpx.get")
    def test_ncbi_api_key_forwarded_when_configured(self, mock_get):
        from research_swarm.config import settings
        mock_get.side_effect = [_mock_esearch_response([])]

        original = settings.ncbi_api_key
        try:
            from pydantic import SecretStr
            settings.ncbi_api_key = SecretStr("test-key-123")

            from research_swarm.tools.pubmed_tool import pubmed_search
            pubmed_search.invoke({"query": "test"})

            call_kwargs = mock_get.call_args.kwargs
            assert call_kwargs["params"]["api_key"] == "test-key-123"
        finally:
            settings.ncbi_api_key = original


# ── europe_pmc_search ────────────────────────────────────────────────────────

def _mock_epmc_response(results: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"resultList": {"result": results}}
    return resp


def _epmc_result(**overrides) -> dict:
    base = {
        "id": "42197043",
        "source": "MED",
        "pmid": "42197043",
        "pmcid": "PMC13210284",
        "doi": "10.3390/nu18101583",
        "title": "Intermittent Fasting: Health Impacts and Therapeutic Potential.",
        "journalInfo": {"journal": {"title": "Nutrients"}},
        "pubYear": "2026",
        "isOpenAccess": "Y",
        "abstractText": "<h4>Background</h4>Intermittent fasting has emerged as a strategy.",
    }
    base.update(overrides)
    return base


class TestEuropePmcSearch:
    @patch("research_swarm.tools.europe_pmc_tool.httpx.get")
    def test_returns_list_of_source_dicts(self, mock_get):
        mock_get.return_value = _mock_epmc_response([_epmc_result()])

        from research_swarm.tools.europe_pmc_tool import europe_pmc_search
        result = europe_pmc_search.invoke({"query": "intermittent fasting", "max_results": 1})

        assert len(result) == 1
        src = result[0]
        assert src["source_type"] == SourceType.europe_pmc.value
        assert "Intermittent Fasting" in src["title"]
        assert src["credibility_score"] == 0.9

    def test_html_tags_stripped_from_abstract(self):
        with patch("research_swarm.tools.europe_pmc_tool.httpx.get") as mock_get:
            mock_get.return_value = _mock_epmc_response([_epmc_result()])
            from research_swarm.tools.europe_pmc_tool import europe_pmc_search
            result = europe_pmc_search.invoke({"query": "test"})

        snippet = result[0]["snippet"]
        assert "<h4>" not in snippet
        assert "Background" in snippet
        assert "Intermittent fasting has emerged" in snippet

    def test_open_access_result_gets_pmc_url(self):
        with patch("research_swarm.tools.europe_pmc_tool.httpx.get") as mock_get:
            mock_get.return_value = _mock_epmc_response(
                [_epmc_result(isOpenAccess="Y", pmcid="PMC13210284")]
            )
            from research_swarm.tools.europe_pmc_tool import europe_pmc_search
            result = europe_pmc_search.invoke({"query": "test"})

        assert result[0]["url"] == "https://europepmc.org/article/PMC/PMC13210284"

    def test_non_open_access_result_gets_generic_article_url(self):
        with patch("research_swarm.tools.europe_pmc_tool.httpx.get") as mock_get:
            mock_get.return_value = _mock_epmc_response(
                [_epmc_result(isOpenAccess="N", pmcid=None, source="MED", id="99")]
            )
            from research_swarm.tools.europe_pmc_tool import europe_pmc_search
            result = europe_pmc_search.invoke({"query": "test"})

        url = result[0]["url"]
        assert "/article/PMC/" not in url
        assert url == "https://europepmc.org/article/MED/99"

    def test_empty_results_returns_no_results_placeholder(self):
        with patch("research_swarm.tools.europe_pmc_tool.httpx.get") as mock_get:
            mock_get.return_value = _mock_epmc_response([])
            from research_swarm.tools.europe_pmc_tool import europe_pmc_search
            result = europe_pmc_search.invoke({"query": "an extremely obscure query"})

        assert len(result) == 1
        assert result[0]["source_type"] == SourceType.europe_pmc.value
        assert result[0]["credibility_score"] == 0.0

    def test_network_error_returns_error_placeholder(self):
        with patch("research_swarm.tools.europe_pmc_tool.httpx.get") as mock_get:
            mock_get.side_effect = Exception("connection reset")
            from research_swarm.tools.europe_pmc_tool import europe_pmc_search
            result = europe_pmc_search.invoke({"query": "test"})

        assert len(result) == 1
        assert result[0]["credibility_score"] == 0.0
        assert "Search error" in result[0]["snippet"]

    def test_results_without_abstract_are_skipped(self):
        with patch("research_swarm.tools.europe_pmc_tool.httpx.get") as mock_get:
            mock_get.return_value = _mock_epmc_response([_epmc_result(abstractText="")])
            from research_swarm.tools.europe_pmc_tool import europe_pmc_search
            result = europe_pmc_search.invoke({"query": "test"})

        # No usable abstract -- falls through to the "no results" placeholder,
        # not a Source built from an empty snippet.
        assert len(result) == 1
        assert result[0]["credibility_score"] == 0.0


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
        # PDF responses now get a descriptive note rather than a generic Non-HTML message
        assert "PDF" in result["snippet"]
        assert result["credibility_score"] == 0.3

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
