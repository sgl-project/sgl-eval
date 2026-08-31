"""Generic dataset loaders for vendored NeMo-Skills benchmarks.

Two patterns are needed across the current benchmark set:

  - **bundled**: data ships in the upstream repo (aime24/aime25 each have a
    ``test.txt`` jsonl beside ``prepare.py``). The vendored ``test.txt`` is
    read directly.

  - **prepare**: data is downloaded by upstream's ``prepare.py`` (gsm8k from
    openai/grade-school-math, mmlu from a pinned CAIS archive, and gpqa and
    mmlu_pro from HuggingFace). We invoke it once, move its output to
    ``~/.cache/sgl_eval/<name>/``, and serve from cache on subsequent runs.
    Upstream's entry point is not uniform -- most expose ``save_data(split,
    ...)``, mmlu-pro only an argparse ``main(args)`` -- so ``argparse_main``
    selects the call shape.

A multimodal **prepare** benchmark also writes a media sidecar dir beside its
jsonl and references it by relative path (mmmu_pro_vision -> ``images/`` +
``image_path``); see ``load_via_prepare``.

``--num-examples N`` keeps the first N rows, matching upstream's ``max_samples``.
A benchmark whose jsonl is grouped rather than shuffled declares ``sample_seed``
instead, or a small N scores one group only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from sgl_eval import VENDORED_NS_ROOT
from sgl_eval.types import Example, MediaItem

_CACHE_ROOT = Path.home() / ".cache" / "sgl_eval"
_VENDORED_DATASET_ROOT = VENDORED_NS_ROOT / "dataset"
_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _row_to_example(
    row: dict, idx: int, name: str, media: Optional[List[MediaItem]] = None
) -> Example:
    return Example(
        id=row.get("id") or f"{name}-{idx}",
        inputs={"problem": row["problem"]},
        target=row["expected_answer"],
        meta={k: v for k, v in row.items() if k not in ("problem", "expected_answer", "id")},
        media=media or [],
    )


def _row_media(row: dict, media_field: str, base_dir: Path) -> List[MediaItem]:
    """A missing media file is a corrupt cache, not a data variation -- fail
    loudly rather than silently evaluating a screenshot benchmark as text (the
    failure mode behind MMMU-Pro's original 9% baseline)."""
    rel = row.get(media_field)
    if not rel:
        return []
    path = base_dir / rel
    if not path.is_file():
        raise FileNotFoundError(
            f"{media_field}={rel!r} missing under {base_dir}; "
            f"delete the cache dir to force a re-prepare"
        )
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")
    return [MediaItem(kind="image", data=path.read_bytes(), mime=mime)]


def _select_rows(
    path: Path, num_examples: Optional[int], sample_seed: Optional[int]
) -> List[Tuple[int, dict]]:
    """Rows to score, paired with their line number so ids stay stable across
    ``num_examples``. Without ``sample_seed`` reading stops early -- sampling
    needs every row, which a ~500KB-per-row split cannot afford."""
    rows: List[Tuple[int, dict]] = []
    with path.open("rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows.append((i, json.loads(line)))
            if sample_seed is None and num_examples and len(rows) >= num_examples:
                break
    if sample_seed is not None and num_examples and num_examples < len(rows):
        rows = random.Random(sample_seed).sample(rows, num_examples)
    return rows


def _read_jsonl(
    path: Path,
    name: str,
    num_examples: Optional[int],
    media_field: Optional[str] = None,
    media_base: Optional[Path] = None,
    sample_seed: Optional[int] = None,
) -> List[Example]:
    # Media loads after selection, so a sampled run only reads what it scores.
    return [
        _row_to_example(
            row, i, name, _row_media(row, media_field, media_base) if media_field else []
        )
        for i, row in _select_rows(path, num_examples, sample_seed)
    ]


def load_bundled(name: str, filename: str = "test.txt") -> Callable[[Optional[int]], List[Example]]:
    """Loader for benchmarks whose data is vendored as a bundled JSONL file."""
    path = _VENDORED_DATASET_ROOT / name / filename

    def loader(num_examples: Optional[int]) -> List[Example]:
        return _read_jsonl(path, name, num_examples)

    return loader


def load_from_path(path: str) -> Callable[[Optional[int]], List[Example]]:
    """User-supplied NS-shape jsonl. Each row needs ``problem`` and
    ``expected_answer``; ``id`` is optional (auto-filled as ``custom-<idx>``).
    Other fields land in ``Example.meta``. Errors out with file:line on
    malformed JSON or missing required fields."""
    p = Path(path).expanduser()
    if not p.is_file():
        sys.exit(f"error: --from-dataset file not found: {p}")

    def loader(num_examples: Optional[int]) -> List[Example]:
        examples: List[Example] = []
        with p.open("rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"error: {p}:{i + 1}: invalid JSON: {e}")
                for required in ("problem", "expected_answer"):
                    if required not in row:
                        sys.exit(f"error: {p}:{i + 1}: missing required field {required!r}")
                examples.append(_row_to_example(row, i, "custom"))
                if num_examples and len(examples) >= num_examples:
                    break
        return examples

    return loader


def load_via_prepare(
    name: str,
    save_args: Sequence[Any],
    save_kwargs: Optional[Dict[str, Any]] = None,
    media_dir: Optional[str] = None,
    media_field: Optional[str] = None,
    sample_seed: Optional[int] = None,
    argparse_main: bool = False,
    archive_url: Optional[str] = None,
    archive_sha256: Optional[str] = None,
) -> Callable[[Optional[int]], List[Example]]:
    """Loader that runs vendored ``<name>/prepare.py`` once and caches the
    resulting JSONL.

    ``media_dir`` is a sidecar dir ``save_data`` writes beside the jsonl;
    ``media_field`` is the jsonl column pointing into it. Both land in the cache
    dir together so the relative paths keep resolving.

    ``sample_seed`` makes ``num_examples`` a seeded sample of the whole split
    rather than its first N rows.

    ``archive_url`` replaces the source a prepare module downloads through its
    module-level ``URL``. sgl-eval fetches and verifies the archive first, then
    lets the vendored transformation consume the local copy unchanged.
    """
    save_kwargs = save_kwargs or {}
    if bool(archive_url) != bool(archive_sha256):
        raise ValueError("archive_url and archive_sha256 must be configured together")
    if archive_sha256 is not None:
        archive_sha256 = archive_sha256.lower()
        if len(archive_sha256) != 64 or any(c not in "0123456789abcdef" for c in archive_sha256):
            raise ValueError("archive_sha256 must be a 64-character hexadecimal digest")
    output_basename = f"{save_args[0]}.jsonl"

    def loader(num_examples: Optional[int]) -> List[Example]:
        cache_dir = _CACHE_ROOT / name
        cache_path = cache_dir / output_basename
        if not cache_path.exists():
            mod = importlib.import_module(f"sgl_eval._vendored.nemo_skills.dataset.{name}.prepare")
            vendored_dir = Path(mod.__file__).resolve().parent
            if archive_url:
                _save_data_from_verified_archive(
                    mod,
                    save_args,
                    save_kwargs,
                    archive_url,
                    archive_sha256,
                    cache_dir,
                )
            elif argparse_main:
                mod.main(argparse.Namespace(split=save_args[0], **save_kwargs))
            else:
                mod.save_data(*save_args, **save_kwargs)
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(vendored_dir / output_basename), str(cache_path))
            # save_data's data_dir is `Path(__file__).parent`, i.e. inside
            # _vendored -- the sidecar has to come out with the jsonl.
            if media_dir:
                _move_tree(vendored_dir / media_dir, cache_dir / media_dir)
        return _read_jsonl(cache_path, name, num_examples, media_field, cache_dir, sample_seed)

    return loader


def _save_data_from_verified_archive(
    mod: Any,
    save_args: Sequence[Any],
    save_kwargs: Dict[str, Any],
    archive_url: str,
    archive_sha256: str,
    cache_dir: Path,
) -> None:
    """Run a vendored prepare transform against a verified local archive."""
    if not hasattr(mod, "URL"):
        raise RuntimeError("archive prefetch requires the prepare module to define URL")
    original_url = mod.URL
    archive_path = _download_verified_archive(
        archive_url,
        expected_sha256=archive_sha256,
        directory=cache_dir,
    )
    try:
        try:
            mod.URL = archive_path.as_uri()
            mod.save_data(*save_args, **save_kwargs)
        finally:
            mod.URL = original_url
    finally:
        archive_path.unlink(missing_ok=True)


def _download_verified_archive(
    url: str,
    expected_sha256: str,
    directory: Path,
) -> Path:
    """Download the archive and return it only if its SHA-256 matches."""
    directory.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".sgl-eval-archive-",
        suffix=".tmp",
        dir=directory,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    verified = False
    try:
        with (
            temp_path.open("wb") as temp_file,
            httpx.stream(
                "GET",
                url,
                follow_redirects=True,
                timeout=_DOWNLOAD_TIMEOUT,
            ) as response,
        ):
            response.raise_for_status()
            for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_BYTES):
                temp_file.write(chunk)
                digest.update(chunk)

        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {url} (expected {expected_sha256}, got {actual_sha256})"
            )
        verified = True
        return temp_path
    finally:
        if not verified:
            temp_path.unlink(missing_ok=True)


def _move_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"prepare.py produced no media dir at {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
