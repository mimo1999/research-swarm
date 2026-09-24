"""LangSmith tracing helpers (runtime/langsmith_trace.py) and the live graph-diagram styling
(ui/graph_view.py) added for the Streamlit UI's stage-progress view."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from research_swarm.runtime import langsmith_trace as lst
from research_swarm.ui.graph_view import _mermaid_source

# --- langsmith_trace ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    from pydantic import SecretStr

    from research_swarm.config import settings

    # Tracing config lives on `settings` (parsed from .env like everything else in this app),
    # not on os.environ -- pydantic-settings' env_file loading never touches the real process
    # environment, only the declared Settings fields. Isolate every test from whatever this
    # machine's real .env happens to have.
    monkeypatch.setattr(settings, "langchain_tracing_v2", False)
    monkeypatch.setattr(settings, "langchain_project", "research-swarm")
    monkeypatch.setattr(settings, "langsmith_api_key", SecretStr(""))


def test_tracing_disabled_without_the_flag(monkeypatch):
    from pydantic import SecretStr

    from research_swarm.config import settings

    monkeypatch.setattr(settings, "langsmith_api_key", SecretStr("sk-real"))
    assert lst.tracing_enabled() is False        # langchain_tracing_v2 still False


def test_tracing_disabled_without_a_key(monkeypatch):
    from research_swarm.config import settings

    monkeypatch.setattr(settings, "langchain_tracing_v2", True)
    assert lst.tracing_enabled() is False


def test_tracing_enabled_with_flag_and_key(monkeypatch):
    from pydantic import SecretStr

    from research_swarm.config import settings

    monkeypatch.setattr(settings, "langchain_tracing_v2", True)
    monkeypatch.setattr(settings, "langsmith_api_key", SecretStr("sk-real"))
    assert lst.tracing_enabled() is True


def test_project_name_falls_back_to_default():
    assert lst.project_name() == "research-swarm"


def test_project_name_reads_settings(monkeypatch):
    from research_swarm.config import settings

    monkeypatch.setattr(settings, "langchain_project", "my-project")
    assert lst.project_name() == "my-project"


def test_make_tracer_returns_none_when_tracing_is_off():
    assert lst.make_tracer("session-1") is None


def test_make_tracer_builds_a_tagged_tracer_when_enabled(monkeypatch):
    from pydantic import SecretStr

    from research_swarm.config import settings

    monkeypatch.setattr(settings, "langchain_tracing_v2", True)
    monkeypatch.setattr(settings, "langsmith_api_key", SecretStr("sk-real"))
    tracer = lst.make_tracer("session-1")
    assert tracer is not None
    assert tracer.tags == ["session:session-1"]
    assert tracer.project_name == "research-swarm"


def test_run_url_is_none_for_no_tracer():
    assert lst.run_url(None) is None


def test_run_url_swallows_a_lookup_failure():
    tracer = MagicMock()
    tracer.get_run_url.side_effect = RuntimeError("no such run")
    assert lst.run_url(tracer) is None


def test_run_url_returns_the_tracers_url():
    tracer = MagicMock()
    tracer.get_run_url.return_value = "https://smith.langchain.com/o/x/r/abc"
    assert lst.run_url(tracer) == "https://smith.langchain.com/o/x/r/abc"


# --- graph_view mermaid styling ------------------------------------------------------------

class _FakeGraph:
    """Stands in for a compiled LangGraph: only .get_graph().draw_mermaid() is used."""

    def get_graph(self):
        gg = MagicMock()
        gg.draw_mermaid.return_value = "graph TD;\n\tsupervisor(supervisor)\n\twriter(writer)"
        return gg


def test_mermaid_source_highlights_the_active_node():
    src = _mermaid_source(_FakeGraph(), "supervisor", set())
    assert "class supervisor active;" in src
    assert "class" not in src.split("class supervisor active;")[0].split("classDef")[-1]


def test_mermaid_source_marks_done_nodes_excluding_the_active_one():
    src = _mermaid_source(_FakeGraph(), "writer", {"supervisor", "writer"})
    assert "class supervisor done;" in src
    assert "class writer active;" in src
    assert "writer done" not in src


def test_mermaid_source_with_nothing_visited_yet_has_no_class_lines():
    src = _mermaid_source(_FakeGraph(), None, set())
    assert "class " not in src.replace("classDef", "")
