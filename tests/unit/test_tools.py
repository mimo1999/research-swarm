"""Unit tests for Phase 2 tools — all network calls are mocked."""
from unittest.mock import MagicMock, patch

from research_swarm.schemas.source import SourceType

# ── helpers ──────────────────────────────────────────────────────────────────


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
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"esearchresult": {"idlist": pmids}}
    return resp


def _mock_efetch_response(xml: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
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
    def test_network_error_returns_error_placeholder(self, mock_get):
        mock_get.side_effect = Exception("connection reset")

        from research_swarm.tools.pubmed_tool import pubmed_search
        result = pubmed_search.invoke({"query": "test"})

        assert len(result) == 1
        assert result[0]["credibility_score"] == 0.0
        assert "Search error" in result[0]["snippet"]


# ── europe_pmc_search ────────────────────────────────────────────────────────

def _mock_epmc_response(results: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
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


# ── pdf_loader ────────────────────────────────────────────────────────────────


# ── retriever_tool ────────────────────────────────────────────────────────────
