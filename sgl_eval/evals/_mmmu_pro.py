"""Load MMMU-Pro standard (10 options) into the NeMo-Skills MCQ schema.

NeMo-Skills provides the vision config, whose questions are screenshots.
This loader uses the standard config with text questions and separate images.
"""

from __future__ import annotations

import ast
import io
import re
import warnings
from typing import List, Optional

from sgl_eval._vendored.nemo_skills.dataset._utils import get_mcq_fields
from sgl_eval.types import Example, MediaItem

_DATASET = "MMMU/MMMU_Pro"
# MMMU-Pro ships 3 configs; "standard (10 options)" is the hard variant.
_CONFIG = "standard (10 options)"
# Questions reference images as <image n>; the dataset stores them in columns
# image_1..image_7 (there is no single "image" column).
_IMAGE_REF_RE = re.compile(r"<image\s+(\d+)>")


def load_mmmu_pro(split: str = "test", num_examples: Optional[int] = None) -> List[Example]:
    """Load the standard split with referenced images attached.

    Malformed options or missing referenced images warn and skip the row;
    these rows cannot be evaluated faithfully as text-only questions.
    """
    from datasets import load_dataset  # lazy: heavy import, benchmark-specific

    ds = load_dataset(_DATASET, _CONFIG, split=split)
    examples: List[Example] = []
    for i, row in enumerate(ds):
        try:
            examples.append(_build_example(row, i))
        except ValueError as e:
            warnings.warn(f"skipping MMMU-Pro row {row.get('id') or i}: {e}")
            continue
        if num_examples is not None and len(examples) >= num_examples:
            break
    return examples


def _build_example(row, i: int) -> Example:
    question = row["question"]
    images = _images_for_question(row, question)
    # Replace <image n> with [image] so build_user_content splices it in place.
    question = _IMAGE_REF_RE.sub("[image]" if images else "", question)
    options = _parse_options(row)
    answer = _normalize_answer(row.get("answer"))
    problem = get_mcq_fields(question, options)["problem"]
    return Example(
        id=row.get("id") or f"mmmu_pro-{i}",
        inputs={"problem": problem},
        target=answer,
        meta={
            "subject": row.get("subject"),
            "difficulty": row.get("topic_difficulty"),
            "image_type": row.get("img_type"),
        },
        media=_image_media(images),
    )


def _parse_options(row) -> list:
    options = row.get("options")
    if isinstance(options, str):
        # HF options may be a Python literal; literal_eval can also yield a non-list.
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"cannot parse options literal: {e}") from e
    if not isinstance(options, (list, tuple)):
        raise ValueError(f"options is {type(options).__name__}, expected a list")
    options = list(options)
    if not options:
        raise ValueError("empty options")
    return options


def _normalize_answer(answer) -> Optional[str]:
    """MMMU-Pro answers are letters; tolerate an int index just in case."""
    if answer is None:
        return None
    if isinstance(answer, int):
        return chr(ord("A") + answer)
    return str(answer).strip().upper()[:1] or None


def _images_for_question(row, question: str) -> List:
    """Referenced images are required; unmarked questions may use image_1 or image."""
    ids = [int(m.group(1)) for m in _IMAGE_REF_RE.finditer(question)]
    if ids:
        images = []
        for n in ids:
            img = row.get(f"image_{n}")
            if img is None:
                raise ValueError(f"question references <image {n}> but image_{n} is missing")
            images.append(img)
        return images
    img = row.get("image_1")
    if img is None:
        img = row.get("image")  # back-compat if a flat image column ever appears
    return [img] if img is not None else []


def _image_media(images) -> List[MediaItem]:
    if not images:
        return []
    if not isinstance(images, (list, tuple)):
        images = [images]
    media: List[MediaItem] = []
    for image in images:
        if image is None:
            continue
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        media.append(MediaItem(kind="image", data=buf.getvalue(), mime="image/png"))
    return media
