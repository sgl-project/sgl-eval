"""Vision-aware message construction shared by multimodal benchmarks.

``build_user_content`` turns a prompt + ``Example.media`` list into an
OpenAI-style ``content``: a plain string when there is no media (so text
benchmarks stay byte-for-byte compatible), or a list of text / image_url /
video_url blocks otherwise. The sampler passes it through unchanged.

Image placement: if the prompt contains ``[image]`` placeholders (left by
the MMMU-Pro loader after stripping ``<image n>``), images are inserted at
those positions to preserve in-question order; otherwise ``image_position``
decides, defaulting to appending after the text.
"""

from __future__ import annotations

import base64
from typing import List, Union

from sgl_eval.types import MediaItem

ContentType = Union[str, list]

_IMAGE_PLACEHOLDER = "[image]"


def build_user_content(
    prompt: str, media: List[MediaItem], image_position: str = "after"
) -> ContentType:
    """Render a user message ``content`` for the given prompt + media.

    No media -> plain string (text-benchmark path, unchanged). Images inline
    as ``data:`` base64; video uses a ``video_url`` block (too large to inline).

    ``image_position`` (from the prompt yaml, see
    ``_prompts.prompt_media_config``) applies only when the prompt carries no
    ``[image]`` placeholder: ``"before"`` puts every image ahead of the text,
    which is what screenshot-style benchmarks need since the text is just an
    answer-format instruction. Explicit placeholders always win over it.
    """
    if not media:
        return prompt
    content: list = []
    image_media = [m for m in media if m.kind == "image"]
    image_idx = 0

    # Insert images at [image] placeholders to preserve in-question order
    # (e.g. MMMU-Pro's <image n> position); fall back to image_position.
    if image_media and _IMAGE_PLACEHOLDER in prompt:
        parts = prompt.split(_IMAGE_PLACEHOLDER)
        for idx, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if idx < len(parts) - 1:
                if image_idx < len(image_media):
                    content.append(_image_block(image_media[image_idx]))
                    image_idx += 1
                else:
                    # placeholder with no matching image: visible, not silently dropped
                    content.append({"type": "text", "text": "[image missing]"})
    elif image_media and image_position == "before":
        for m in image_media:
            content.append(_image_block(m))
        image_idx = len(image_media)
        content.append({"type": "text", "text": prompt})
    else:
        content.append({"type": "text", "text": prompt})

    # Append media not consumed above (video, and images with no placeholder).
    inserted_images = image_idx
    for m in media:
        if m.kind == "image":
            if inserted_images > 0:
                inserted_images -= 1
                continue
            content.append(_image_block(m))
        elif m.kind == "video":
            if not m.url:
                raise ValueError("video MediaItem requires a url (too large to base64-inline)")
            content.append({"type": "video_url", "video_url": {"url": m.url}})
        else:
            raise ValueError(f"unsupported media kind: {m.kind!r}")
    return content


def _image_block(m: MediaItem) -> dict:
    url = m.url or _data_url(m.data, m.mime or "image/png")
    return {"type": "image_url", "image_url": {"url": url}}


def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"
