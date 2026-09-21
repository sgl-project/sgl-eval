"""A cache is served only when its digest marker vouches for it."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from sgl_eval.evals import _loader


def _install(monkeypatch, tmp_path, calls, payload='{"problem": "q", "expected_answer": "A"}\n'):
    mod = types.ModuleType("fake_prepare")
    mod.__file__ = str(tmp_path / "site_packages" / "gpqa" / "prepare.py")

    def save_data(split: str) -> None:
        calls.append(split)
        out_dir = Path(mod.__file__).absolute().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{split}.jsonl").write_text(payload)

    mod.save_data = save_data
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)


@pytest.mark.parametrize("damage", ["drop_marker", "edit_rows"])
def test_unvouched_cache_is_rebuilt(tmp_path, monkeypatch, damage):
    calls: list[str] = []
    _install(monkeypatch, tmp_path, calls)
    loader = _loader.load_via_prepare("gpqa", ["test"], {})

    loader(None)
    loader(None)
    assert calls == ["test"]

    cache_path = tmp_path / "cache" / "gpqa" / "test.jsonl"
    if damage == "drop_marker":
        # Shape of every cache written before the marker existed.
        _loader._cache_marker(cache_path).unlink()
    else:
        cache_path.write_text('{"problem": "q", "expec')

    assert loader(None)[0].inputs["problem"] == "q"
    assert calls == ["test", "test"]


def test_invalid_row_names_the_file_and_line(tmp_path, monkeypatch):
    _install(
        monkeypatch, tmp_path, [], payload='{"problem": "q", "expected_answer": "A"}\n{"p": \n'
    )

    with pytest.raises(ValueError, match=r"test\.jsonl:2: prepare wrote invalid JSON"):
        _loader.load_via_prepare("gpqa", ["test"], {})(None)
