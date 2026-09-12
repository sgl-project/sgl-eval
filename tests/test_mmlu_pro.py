"""MMLU-Pro: upstream metadata, the argparse-``main`` prepare shape, and the
prompt it must not share with ``mmmu_pro``.
"""

from __future__ import annotations

import json
import types

from sgl_eval.evals import _loader
from sgl_eval.evals._prompts import render_prompt, resolve_prompt
from sgl_eval.registry import get


def test_mmlu_pro_registered_from_upstream_metadata():
    """``metrics_type`` and the prompt basename are derived from the vendored
    ``dataset/mmlu_pro/__init__.py``, never hand-mirrored -- so an upstream
    sync that retargets either one has to show up here."""
    from sgl_eval.evals._registry import _resolve_upstream_metadata

    assert get("mmlu_pro").category == "multichoice"
    assert _resolve_upstream_metadata("mmlu_pro") == ("multichoice", "mcq-10choices")


def test_mmlu_pro_prompt_enumerates_ten_letters():
    rendered = render_prompt(resolve_prompt("mcq-10choices"), problem="Q?\n\nA) x\nB) y")
    assert "Answer: A/B/C/D/E/F/G/H/I/J" in rendered
    assert "Q?" in rendered


def _fake_prepare_module(tmp_path, rows=3):
    """Stands in for a vendored prepare.py: records which entry point ran and
    writes the jsonl where the real one would."""
    calls = {}

    def _write():
        with (tmp_path / "test.jsonl").open("w") as f:
            for i in range(rows):
                f.write(json.dumps({"problem": f"q{i}", "expected_answer": "A"}) + "\n")

    def main(args):
        calls["main"] = args.split
        _write()

    def save_data(*args, **kwargs):
        calls["save_data"] = args
        _write()

    mod = types.SimpleNamespace(
        __file__=str(tmp_path / "prepare.py"), main=main, save_data=save_data
    )
    return mod, calls


def _install(monkeypatch, tmp_path, mod):
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _path: mod)


def test_argparse_main_shape_calls_main_with_the_split(tmp_path, monkeypatch):
    """mmlu-pro's prepare.py has no ``save_data``; calling the wrong entry
    point raises AttributeError only once someone actually runs the benchmark,
    which needs a 12k-row download."""
    mod, calls = _fake_prepare_module(tmp_path)
    _install(monkeypatch, tmp_path, mod)

    examples = _loader.load_via_prepare("mmlu_pro", ["test"], {}, argparse_main=True)(None)

    assert calls == {"main": "test"}
    assert [e.inputs["problem"] for e in examples] == ["q0", "q1", "q2"]


def test_default_shape_still_calls_save_data(tmp_path, monkeypatch):
    """The flag is opt-in: every other prepare benchmark keeps save_data."""
    mod, calls = _fake_prepare_module(tmp_path)
    _install(monkeypatch, tmp_path, mod)

    _loader.load_via_prepare("gpqa", ["test"], {"random_seed": 42})(None)

    assert calls == {"save_data": ("test",)}


def test_registry_wires_mmlu_pro_to_the_argparse_shape(tmp_path, monkeypatch):
    """Dropping ``argparse_main`` from the row, or forgetting to forward it in
    ``_build_loader``, leaves mmlu_pro calling a ``save_data`` its prepare.py
    does not have."""
    from sgl_eval.evals._registry import _TABLE, _build_loader

    mod, calls = _fake_prepare_module(tmp_path)
    _install(monkeypatch, tmp_path, mod)

    row = next(r for r in _TABLE if r["name"] == "mmlu_pro")
    _build_loader(row)(None)

    assert calls == {"main": "test"}
