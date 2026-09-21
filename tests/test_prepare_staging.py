"""Prepare output is staged outside the install tree and committed atomically."""

from __future__ import annotations

import json
import threading
import types
from pathlib import Path

import pytest

from sgl_eval.evals import _loader


def _fake_prepare_module(pkg_dir: Path, mid_row=None):
    """Stands in for a vendored prepare.py: derives its output dir from its own
    ``__file__``, exactly as every real one does."""
    mod = types.ModuleType("fake_prepare")
    mod.__file__ = str(pkg_dir / "prepare.py")

    def save_data(split: str) -> None:
        out_dir = Path(mod.__file__).absolute().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for i in range(2):
                f.write(json.dumps({"problem": f"q{i}", "expected_answer": "A"}))
                f.flush()
                if mid_row is not None:
                    mid_row()
                f.write("\n")

    mod.save_data = save_data
    return mod


def test_concurrent_prepares_stage_separately(tmp_path, monkeypatch):
    """Two processes cold-starting one dataset used to write a single file in
    the install tree at once, leaving a spliced JSONL cached for good."""
    pkg_dir = tmp_path / "site_packages" / "gpqa"
    pkg_dir.mkdir(parents=True)
    barrier = threading.Barrier(2, timeout=10)
    local = threading.local()
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: local.mod)

    untouched = []

    def mid_row() -> None:
        # Sampled while both writers sit mid-row: checking only afterwards
        # would be satisfied by moving the file out at the end.
        untouched.append(not any(pkg_dir.iterdir()))
        barrier.wait()

    results = {}

    def run(tag: str) -> None:
        local.mod = _fake_prepare_module(pkg_dir, mid_row=mid_row)
        results[tag] = _loader.load_via_prepare("gpqa", ["test"], {})(None)

    threads = [threading.Thread(target=run, args=(tag,)) for tag in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == ["a", "b"]
    assert all(len(v) == 2 for v in results.values())
    assert untouched == [True] * 4
    rows = (tmp_path / "cache" / "gpqa" / "test.jsonl").read_text().splitlines()
    assert [json.loads(r)["problem"] for r in rows] == ["q0", "q1"]


def test_failed_prepare_leaves_no_cache_dir_or_staging(tmp_path, monkeypatch):
    """A half-built cache dir would be indistinguishable from a real one."""
    mod = types.ModuleType("fake_prepare")
    mod.__file__ = str(tmp_path / "site_packages" / "gpqa" / "prepare.py")

    def save_data(_split: str) -> None:
        raise RuntimeError("upstream is down")

    mod.save_data = save_data
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(_loader, "_CACHE_ROOT", cache_root)
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)

    with pytest.raises(RuntimeError, match="upstream is down"):
        _loader.load_via_prepare("gpqa", ["test"], {})(None)

    assert not (cache_root / "gpqa").exists()
    assert list(cache_root.iterdir()) == []
