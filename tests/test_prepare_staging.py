"""Prepare output is staged outside the install tree and committed atomically."""

from __future__ import annotations

import json
import threading
import types
from pathlib import Path

import pytest

from sgl_eval.evals import _loader


def _fake_prepare_module(pkg_dir: Path, rows: int = 2, before_close=None):
    """Stands in for a vendored prepare.py: derives its output dir from its own
    ``__file__``, exactly as every real one does."""
    mod = types.ModuleType("fake_prepare")
    mod.__file__ = str(pkg_dir / "prepare.py")

    def save_data(split: str) -> None:
        out_dir = Path(mod.__file__).absolute().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for i in range(rows):
                f.write(json.dumps({"problem": f"q{i}", "expected_answer": "A"}))
                f.flush()
                if before_close is not None:
                    before_close()
                f.write("\n")

    mod.save_data = save_data
    return mod


def test_prepare_output_never_lands_in_the_install_tree(tmp_path, monkeypatch):
    """The real scripts write their split -- and mmlu a 166 MB ``data.tar`` --
    beside ``__file__``, a path every process on the machine shares."""
    pkg_dir = tmp_path / "site_packages" / "gpqa"
    pkg_dir.mkdir(parents=True)
    # Sampled while the script is writing: moving the result out afterwards
    # would hide the window the whole machine shares.
    seen_mid_write = []
    mod = _fake_prepare_module(
        pkg_dir,
        before_close=lambda: seen_mid_write.append(sorted(p.name for p in pkg_dir.iterdir())),
    )
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)

    examples = _loader.load_via_prepare("gpqa", ["test"], {})(None)

    assert len(examples) == 2
    assert (tmp_path / "cache" / "gpqa" / "test.jsonl").is_file()
    assert seen_mid_write == [[], []]
    assert list(pkg_dir.iterdir()) == []
    assert mod.__file__ == str(pkg_dir / "prepare.py")


def test_concurrent_prepares_do_not_share_an_output_path(tmp_path, monkeypatch):
    """Two processes cold-starting the same dataset used to write one file in
    the install tree at once, leaving a spliced JSONL cached for good. Modelled
    here as two prepare modules aimed at the same package dir."""
    pkg_dir = tmp_path / "site_packages" / "gpqa"
    pkg_dir.mkdir(parents=True)
    barrier = threading.Barrier(2, timeout=10)
    local = threading.local()
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: local.mod)

    results = {}

    def run(tag: str) -> None:
        # Both writers sit mid-row together, so a shared output path splices.
        local.mod = _fake_prepare_module(pkg_dir, before_close=barrier.wait)
        results[tag] = _loader.load_via_prepare("gpqa", ["test"], {})(None)

    threads = [threading.Thread(target=run, args=(tag,)) for tag in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == ["a", "b"]
    assert all(len(v) == 2 for v in results.values())
    cache_path = tmp_path / "cache" / "gpqa" / "test.jsonl"
    rows = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert [r["problem"] for r in rows] == ["q0", "q1"]
    assert list(pkg_dir.iterdir()) == []


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
