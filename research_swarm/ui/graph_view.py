"""Live graph-topology diagram: graph.get_graph() rendered as Mermaid, with the node the swarm
is currently on (and the ones it has already passed through) highlighted.

Node ids in the Mermaid source are the graph's own node names (``supervisor``,
``paper_scout_node``, ...), so highlighting is just appending ``class <id> <style>`` lines --
no parsing of the diagram itself. The two conditional edges that fan out via ``Send`` (out of
``document_pass_node`` and ``dispatch_node``) carry an explicit ``path_map`` in
``graph/builder.py`` purely so this diagram can draw them; LangGraph can't infer a
Send-returning routing function's targets on its own, and silently drops the edge without it.
"""
from __future__ import annotations

import streamlit.components.v1 as components

_MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs"

_STYLE = """
classDef active fill:#f59e0b,stroke:#b45309,color:#111827,stroke-width:3px;
classDef done fill:#22c55e,stroke:#15803d,color:#111827;
"""


def _mermaid_source(graph, active_node: str | None, done_nodes: set[str]) -> str:
    mermaid = graph.get_graph().draw_mermaid()
    lines = [mermaid, _STYLE]
    done = sorted(done_nodes - ({active_node} if active_node else set()))
    if done:
        lines.append(f"class {','.join(done)} done;")
    if active_node:
        lines.append(f"class {active_node} active;")
    return "\n".join(lines)


def render_graph_diagram(
    graph, active_node: str | None, done_nodes: set[str], *, height: int = 340,
) -> None:
    """Render *graph*'s topology in the caller's current Streamlit container. *active_node* is
    highlighted amber; *done_nodes* (nodes already visited this run) are green.
    """
    mermaid = _mermaid_source(graph, active_node, done_nodes)
    # A fresh id per call so mermaid.js (loaded once per component iframe) always re-initialises
    # against the new diagram content instead of a stale cached render.
    html = f"""
    <div class="mermaid">{mermaid}</div>
    <script type="module">
      import mermaid from '{_MERMAID_CDN}';
      mermaid.initialize({{ startOnLoad: true, theme: 'dark', securityLevel: 'loose' }});
      mermaid.run();
    </script>
    <style>
      body {{ margin: 0; background: transparent; }}
      .mermaid {{ display: flex; justify-content: center; }}
    </style>
    """
    components.html(html, height=height, scrolling=True)
