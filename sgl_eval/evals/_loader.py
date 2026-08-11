"""Generic dataset loaders for vendored NeMo-Skills benchmarks.

Two patterns are needed across the current benchmark set:

  - **bundled**: data ships in the upstream repo (aime24/aime25 each have a
    ``test.txt`` jsonl beside ``prepare.py``). The vendored ``test.txt`` is
    read directly.

  - **prepare**: data is downloaded by upstream's ``prepare.py:save_data``
    (gsm8k from openai/grade-school-math, mmlu from Berkeley tar, gpqa from
    HuggingFace). We invoke the vendored function once, move its output to
    ``~/.cache/sgl_eval/<name>/``, and serve from cache on subsequent runs.

Multimodal ``prepare`` benchmarks additionally write a media sidecar
directory beside the jsonl and reference it by relative path (mmmu_pro_vision
-> ``images/`` + ``image_path``). ``media_dir`` / ``media_field`` move that
directory into the same cache dir and resolve each row's file into
``Example.media``.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sgl_eval import VENDORED_NS_ROOT
from sgl_eval.types import Example, MediaItem

_CACHE_ROOT = Path.home() / ".cache" / "sgl_eval"
_VENDORED_DATASET_ROOT = VENDORED_NS_ROOT / "dataset"

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
    """Resolve a row's media path into an in-memory ``MediaItem``.

    A row whose media file is missing is a corrupt cache, not a data variation
    -- fail loudly rather than silently evaluating a screenshot benchmark as
    text (the failure mode behind MMMU-Pro's original 9% baseline).
    """
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


def _read_jsonl(
    path: Path,
    name: str,
    num_examples: Optional[int],
    media_field: Optional[str] = None,
    media_base: Optional[Path] = None,
) -> List[Example]:
    examples: List[Example] = []
    with path.open("rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            media = _row_media(row, media_field, media_base) if media_field else []
            examples.append(_row_to_example(row, i, name, media))
            if num_examples and len(examples) >= num_examples:
                break
    return examples


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
    save_args: List[Any],
    save_kwargs: Optional[Dict[str, Any]] = None,
    media_dir: Optional[str] = None,
    media_field: Optional[str] = None,
) -> Callable[[Optional[int]], List[Example]]:
    """Loader that runs vendored ``<name>/prepare.py:save_data`` once and
    caches the resulting JSONL.

    ``media_dir`` is a sidecar directory ``save_data`` writes beside the jsonl
    (relative to the vendored package dir); ``media_field`` is the jsonl column
    holding each row's path into it. Both move to the cache dir together so the
    relative paths keep resolving.
    """
    save_kwargs = save_kwargs or {}
    output_basename = f"{save_args[0]}.jsonl"

    def loader(num_examples: Optional[int]) -> List[Example]:
        cache_dir = _CACHE_ROOT / name
        cache_path = cache_dir / output_basename
        if not cache_path.exists():
            mod = importlib.import_module(f"sgl_eval._vendored.nemo_skills.dataset.{name}.prepare")
            vendored_dir = Path(mod.__file__).resolve().parent
            mod.save_data(*save_args, **save_kwargs)
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(vendored_dir / output_basename), str(cache_path))
            # save_data writes the sidecar into _vendored (its data_dir is
            # `Path(__file__).parent`), so it has to move out alongside the
            # jsonl -- otherwise the cached relative paths dangle and
            # _vendored accumulates generated data.
            if media_dir:
                _move_tree(vendored_dir / media_dir, cache_dir / media_dir)
        return _read_jsonl(cache_path, name, num_examples, media_field, cache_dir)

    return loader


def _move_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"prepare.py produced no media dir at {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
