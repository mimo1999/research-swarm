"""benchmarks/bench_common.py: statistics, BEIR loader, lexical helpers, README blocks."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_common.py"
_spec = importlib.util.spec_from_file_location("bench_common", _PATH)
bc = importlib.util.module_from_spec(_spec)
sys.modules["bench_common"] = bc
_spec.loader.exec_module(bc)


class TestBootstrap:


    def test_interval_brackets_the_mean_and_is_deterministic(self):
        values = [0.0, 1.0] * 25
        a = bc.bootstrap_ci(values)
        assert a == bc.bootstrap_ci(values)
        mean, lo, hi = a
        assert mean == 0.5 and lo < mean < hi


class TestUpdateBlock:

    def test_replaces_only_its_own_block_and_is_idempotent(self, tmp_path):
        p = tmp_path / "README.md"
        p.write_text("top\n\n<!-- A:START -->\nold a\n<!-- A:END -->\n\nmid\n\n"
                     "<!-- B:START -->\nold b\n<!-- B:END -->\n\nbottom\n", encoding="utf-8")
        bc.update_block(p, "A", "new a")
        bc.update_block(p, "A", "new a")
        text = p.read_text(encoding="utf-8")
        assert "new a" in text and "old a" not in text
        assert "old b" in text and text.count("<!-- A:START -->") == 1
        assert text.startswith("top") and text.rstrip().endswith("bottom")
