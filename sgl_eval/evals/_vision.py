"""Build OpenAI message content from prompts and media.

Text-only inputs remain strings. Image placeholders take precedence over
the YAML placement default; unused media follow the text.
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
    """Return plain text without media, otherwise text/image_url/video_url blocks.

    Explicit [image] placeholders override image_position. Images without URLs
    are encoded as data URLs; videos require a server-accessible URL.
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
