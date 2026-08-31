"""Download fallback for the vendored prepare scripts.

The failure this guards is a cold-cache run dying on one unreachable host
(sgl-project/sgl-eval#33). Nothing here touches the network: the point is the
ordering and the restore, not the transfer.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from sgl_eval.evals import _fetch

PRIMARY = _fetch._MMLU_PRIMARY


def test_mmlu_falls_back_to_a_mirror():
    assert _fetch._candidates(PRIMARY)[0] == PRIMARY, "primary must stay first"
    assert len(_fetch._candidates(PRIMARY)) > 1, "mmlu needs a mirror"


def test_an_unmirrored_url_is_left_alone():
    """Only URLs in the table get alternates -- a typo'd host must not
    silently resolve to someone else's data."""
    assert _fetch._candidates("https://other.example/x.tar") == ["https://other.example/x.tar"]


def test_override_replaces_the_whole_chain(monkeypatch):
    """An air-gapped or internal-mirror run must not still reach for the
    public hosts, so the override is exclusive rather than prepended."""
    monkeypatch.setenv("SGL_EVAL_MMLU_URL", "file:///data/mmlu.tar")
    assert _fetch._candidates(PRIMARY) == ["file:///data/mmlu.tar"]


def test_retrieve_moves_on_after_the_primary_fails(monkeypatch):
    seen = []

    def fake(url, filename):
        seen.append(url)
        if url == PRIMARY:
            raise urllib.error.URLError("timed out")

    monkeypatch.setattr(_fetch, "_download", fake)
    monkeypatch.setattr(_fetch, "_BACKOFF_SECONDS", 0)

    used = _fetch.retrieve(PRIMARY, "/tmp/x.tar")
    assert used != PRIMARY
    assert seen.count(PRIMARY) == _fetch._ATTEMPTS_PER_URL, "primary retried before moving on"


def test_retrieve_reports_every_source_it_tried(monkeypatch):
    """A bare 'Errno 110' was the original complaint -- the error has to name
    what was attempted, or the next person cannot tell a dead mirror from a
    dead network."""
    monkeypatch.setattr(
        _fetch, "_download", lambda url, filename: (_ for _ in ()).throw(OSError("nope"))
    )
    monkeypatch.setattr(_fetch, "_BACKOFF_SECONDS", 0)

    with pytest.raises(RuntimeError) as excinfo:
        _fetch.retrieve(PRIMARY, "/tmp/x.tar")
    for candidate in _fetch._candidates(PRIMARY):
        assert candidate in str(excinfo.value)


def test_patch_is_scoped_to_the_block(monkeypatch):
    """The vendored scripts are called inside this context; leaking the patch
    would change urlretrieve for everything else in the process."""
    monkeypatch.setattr(_fetch, "_download", lambda url, filename: None)
    original = urllib.request.urlretrieve

    with _fetch.resilient_downloads():
        assert urllib.request.urlretrieve is not original
        path, headers = urllib.request.urlretrieve(PRIMARY, "/tmp/x.tar")
        assert path == "/tmp/x.tar"

    assert urllib.request.urlretrieve is original


def test_patch_restores_on_exception(monkeypatch):
    original = urllib.request.urlretrieve
    with pytest.raises(ValueError):
        with _fetch.resilient_downloads():
            raise ValueError("prepare blew up")
    assert urllib.request.urlretrieve is original


def test_loader_installs_the_fallback_around_prepare(tmp_path, monkeypatch):
    """The wrapper only matters at the one call site. Dropping it from
    `load_via_prepare` breaks nothing until a cold cache meets a dead host, so
    assert the vendored script really runs under the patch."""
    import json
    import types

    from sgl_eval.evals import _loader

    original = urllib.request.urlretrieve
    seen = {}

    def save_data(split):
        seen["patched"] = urllib.request.urlretrieve is not original
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"problem": "q", "expected_answer": "A"}) + "\n"
        )

    mod = types.SimpleNamespace(__file__=str(tmp_path / "prepare.py"), save_data=save_data)
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _p: mod)

    _loader.load_via_prepare("mmlu", ["test"], {})(None)

    assert seen["patched"], "prepare ran with the stock urlretrieve"
    assert urllib.request.urlretrieve is original, "patch leaked past the loader"
