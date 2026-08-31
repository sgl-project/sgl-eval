"""MMLU archive fallback tests (no network)."""

from __future__ import annotations

import hashlib
import json
import types
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from sgl_eval.evals import _loader

_SOURCE = "https://pinned.invalid/data.tar"


def _fake_prepare_module(tmp_path: Path, expected_archive: bytes):
    vendored_dir = tmp_path / "vendored_pkg"
    vendored_dir.mkdir()
    mod = types.ModuleType("fake_mmlu_prepare")
    mod.__file__ = str(vendored_dir / "prepare.py")
    mod.URL = "https://upstream.invalid/data.tar"
    consumed_urls = []

    def save_data(split: str) -> None:
        consumed_urls.append(mod.URL)
        parsed = urllib.parse.urlparse(mod.URL)
        assert parsed.scheme == "file"
        archive_path = Path(urllib.request.url2pathname(parsed.path))
        assert archive_path.read_bytes() == expected_archive
        row = {"id": "m1", "problem": "Question", "expected_answer": "A"}
        (vendored_dir / f"{split}.jsonl").write_text(json.dumps(row) + "\n")

    mod.save_data = save_data
    return mod, consumed_urls


def _loader_for(archive: bytes):
    return _loader.load_via_prepare(
        "mmlu",
        ["test"],
        archive_url=_SOURCE,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
    )


def test_archive_is_fetched_once_and_then_served_from_cache(tmp_path, monkeypatch):
    archive = b"pinned MMLU archive"
    mod, consumed_urls = _fake_prepare_module(tmp_path, archive)
    seen = []

    @contextmanager
    def fake_stream(method, url, **kwargs):
        seen.append((method, url, kwargs))
        yield httpx.Response(200, content=archive, request=httpx.Request(method, url))

    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)
    monkeypatch.setattr(_loader.httpx, "stream", fake_stream)

    loader = _loader_for(archive)
    [example] = loader(None)
    [cached_example] = loader(None)

    assert example.id == cached_example.id == "m1"
    assert [url for _method, url, _kwargs in seen] == [_SOURCE]
    assert all(kwargs["follow_redirects"] for _method, _url, kwargs in seen)
    assert all(kwargs["timeout"] is _loader._DOWNLOAD_TIMEOUT for _method, _url, kwargs in seen)
    assert len(consumed_urls) == 1
    assert consumed_urls[0].startswith("file:")
    assert mod.URL == "https://upstream.invalid/data.tar"
    assert not list((tmp_path / "cache" / "mmlu").glob(".sgl-eval-archive-*"))


def test_checksum_failure_cleans_the_temporary_file(tmp_path, monkeypatch):
    archive = b"expected archive"
    mod, consumed_urls = _fake_prepare_module(tmp_path, archive)
    seen = []

    @contextmanager
    def fake_stream(method, url, **kwargs):
        seen.append(url)
        yield httpx.Response(
            200,
            content=b"unexpected archive",
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)
    monkeypatch.setattr(_loader.httpx, "stream", fake_stream)

    # A single source means the digest error surfaces as itself, rather than
    # wrapped in the "every source failed" report a fallback chain produced.
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _loader_for(archive)(None)

    assert seen == [_SOURCE]
    assert consumed_urls == []
    assert mod.URL == "https://upstream.invalid/data.tar"
    assert not list((tmp_path / "cache" / "mmlu").glob(".sgl-eval-archive-*"))


def test_unexpected_stream_failure_cleans_temporary_file(tmp_path, monkeypatch):
    archive = b"expected archive"
    mod, consumed_urls = _fake_prepare_module(tmp_path, archive)

    class FailingStream:
        def __enter__(self):
            raise RuntimeError("unexpected transport failure")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)
    monkeypatch.setattr(_loader.httpx, "stream", lambda *_args, **_kwargs: FailingStream())

    with pytest.raises(RuntimeError, match="unexpected transport failure"):
        _loader_for(archive)(None)

    assert consumed_urls == []
    assert not list((tmp_path / "cache" / "mmlu").glob(".sgl-eval-archive-*"))


def test_prepare_failure_restores_url_and_removes_verified_archive(tmp_path, monkeypatch):
    archive = b"pinned MMLU archive"
    mod, _consumed_urls = _fake_prepare_module(tmp_path, archive)
    original_url = mod.URL
    local_urls = []

    def failing_save_data(_split):
        local_urls.append(mod.URL)
        raise RuntimeError("prepare failed")

    @contextmanager
    def fake_stream(method, url, **kwargs):
        yield httpx.Response(200, content=archive, request=httpx.Request(method, url))

    mod.save_data = failing_save_data
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)
    monkeypatch.setattr(_loader.httpx, "stream", fake_stream)

    with pytest.raises(RuntimeError, match="prepare failed"):
        _loader.load_via_prepare(
            "mmlu",
            ["test"],
            archive_url=_SOURCE,
            archive_sha256=hashlib.sha256(archive).hexdigest(),
        )(None)

    assert len(local_urls) == 1 and local_urls[0].startswith("file:")
    assert mod.URL == original_url
    assert not list((tmp_path / "cache" / "mmlu").glob(".sgl-eval-archive-*"))


def test_missing_prepare_url_fails_before_download(tmp_path, monkeypatch):
    archive = b"pinned MMLU archive"
    mod, _consumed_urls = _fake_prepare_module(tmp_path, archive)
    del mod.URL

    def unexpected_stream(*_args, **_kwargs):
        raise AssertionError("download should not start")

    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)
    monkeypatch.setattr(_loader.httpx, "stream", unexpected_stream)

    with pytest.raises(RuntimeError, match="prepare module to define URL"):
        _loader_for(archive)(None)

    assert not (tmp_path / "cache" / "mmlu").exists()


@pytest.mark.parametrize(
    ("url", "digest"),
    [(_SOURCE, None), (None, "0" * 64), (_SOURCE, "not-a-digest")],
)
def test_archive_configuration_is_complete_and_valid(url, digest):
    with pytest.raises(ValueError, match="archive_"):
        _loader.load_via_prepare(
            "mmlu",
            ["test"],
            archive_url=url,
            archive_sha256=digest,
        )


def test_archive_fallback_cannot_replace_argparse_entry_point():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _loader.load_via_prepare(
            "invalid",
            ["test"],
            argparse_main=True,
            archive_url=_SOURCE,
            archive_sha256="0" * 64,
        )


def test_registry_pins_the_mirror_revision_and_archive_digest():
    from sgl_eval.evals._registry import _TABLE

    [mmlu] = [entry for entry in _TABLE if entry["name"] == "mmlu"]
    assert "c30699e8356da336a370243923dbaf21066bb9fe" in mmlu["archive_url"]
    assert "huggingface.co/datasets/cais/mmlu" in mmlu["archive_url"]
    assert mmlu["archive_sha256"] == (
        "bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b"
    )
