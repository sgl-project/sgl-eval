"""Tests for vision-aware message construction (``build_user_content``)."""

from __future__ import annotations

import base64

import pytest

from sgl_eval.evals._vision import build_user_content
from sgl_eval.types import Example, MediaItem


def test_no_media_returns_plain_string():
    assert build_user_content("hello", []) == "hello"


def test_example_media_defaults_empty():
    ex = Example(id="x", inputs={}, target="A")
    assert ex.media == []
    assert build_user_content("p", ex.media) == "p"


def test_image_inlines_as_base64_data_url():
    m = MediaItem(kind="image", data=b"\x89PNG fake", mime="image/png")
    content = build_user_content("q", [m])
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "q"}
    block = content[1]
    assert block["type"] == "image_url"
    expected = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()
    assert block["image_url"]["url"] == expected


def test_image_prefers_url_when_given():
    m = MediaItem(kind="image", url="https://x/a.png", data=b"bytes", mime="image/png")
    content = build_user_content("q", [m])
    assert content[1]["image_url"]["url"] == "https://x/a.png"


def test_image_default_mime_png():
    m = MediaItem(kind="image", data=b"x")  # no mime -> defaults to image/png
    content = build_user_content("q", [m])
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_video_uses_url():
    m = MediaItem(kind="video", url="https://x/a.mp4", mime="video/mp4")
    content = build_user_content("q", [m])
    assert content[1] == {"type": "video_url", "video_url": {"url": "https://x/a.mp4"}}


def test_video_without_url_raises():
    m = MediaItem(kind="video", data=b"x", mime="video/mp4")
    with pytest.raises(ValueError, match="requires a url"):
        build_user_content("q", [m])


def test_mixed_image_and_video():
    media = [
        MediaItem(kind="image", data=b"i", mime="image/png"),
        MediaItem(kind="video", url="https://x/v.mp4"),
    ]
    content = build_user_content("q", media)
    assert [b["type"] for b in content] == ["text", "image_url", "video_url"]


def test_audio_appended_as_audio_url_block():
    media = [MediaItem(kind="audio", data=b"a", mime="audio/wav")]
    content = build_user_content("q", media)
    assert [b["type"] for b in content] == ["text", "audio_url"]
    assert content[1]["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_audio_url_passthrough():
    media = [MediaItem(kind="audio", url="https://x/a.wav")]
    content = build_user_content("q", media)
    assert content[1] == {"type": "audio_url", "audio_url": {"url": "https://x/a.wav"}}


def test_unsupported_kind_raises():
    with pytest.raises(ValueError, match="unsupported media kind"):
        build_user_content("q", [MediaItem(kind="file", data=b"x")])


def test_image_inserted_at_placeholder_preserves_order():
    """Image goes where [image] sits in the prompt, not appended after text."""
    media = [MediaItem(kind="image", data=b"i", mime="image/png")]
    content = build_user_content("before [image] after", media)
    assert [b["type"] for b in content] == ["text", "image_url", "text"]
    assert content[0]["text"] == "before "
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["text"] == " after"


def test_no_placeholder_appends_image_after_text():
    """No [image] marker -> text first, image appended (backward-compat)."""
    media = [MediaItem(kind="image", data=b"i", mime="image/png")]
    content = build_user_content("plain prompt", media)
    assert [b["type"] for b in content] == ["text", "image_url"]
    assert content[0]["text"] == "plain prompt"


def test_two_placeholders_two_images_in_order():
    media = [
        MediaItem(kind="image", data=b"a", mime="image/png"),
        MediaItem(kind="image", data=b"b", mime="image/png"),
    ]
    content = build_user_content("x[image]y[image]z", media)
    assert [b["type"] for b in content] == ["text", "image_url", "text", "image_url", "text"]


def test_image_position_before_puts_image_ahead_of_text():
    """Screenshot-style benchmarks (mmmu_pro_vision) carry the question inside
    the image; the text is only an answer-format instruction."""
    media = [MediaItem(kind="image", data=b"i", mime="image/png")]
    content = build_user_content("answer format...", media, image_position="before")
    assert [b["type"] for b in content] == ["image_url", "text"]
    assert content[1]["text"] == "answer format..."


def test_image_position_before_with_multiple_images():
    media = [
        MediaItem(kind="image", data=b"a", mime="image/png"),
        MediaItem(kind="image", data=b"b", mime="image/png"),
    ]
    content = build_user_content("q", media, image_position="before")
    assert [b["type"] for b in content] == ["image_url", "image_url", "text"]


def test_placeholder_wins_over_image_position_before():
    """An explicit ``[image]`` marker is a per-question position and must not be
    overridden by the yaml-level default."""
    media = [MediaItem(kind="image", data=b"i", mime="image/png")]
    content = build_user_content("x [image] y", media, image_position="before")
    assert [b["type"] for b in content] == ["text", "image_url", "text"]


def test_image_position_before_still_appends_video():
    media = [
        MediaItem(kind="image", data=b"i", mime="image/png"),
        MediaItem(kind="video", url="https://x/v.mp4"),
    ]
    content = build_user_content("q", media, image_position="before")
    assert [b["type"] for b in content] == ["image_url", "text", "video_url"]


def test_image_position_defaults_to_after():
    media = [MediaItem(kind="image", data=b"i", mime="image/png")]
    assert [b["type"] for b in build_user_content("q", media)] == ["text", "image_url"]


def test_more_placeholders_than_images_inserts_missing_marker():
    """A ``[image]`` placeholder with no matching image becomes a visible
    ``[image missing]`` text block, not a silent drop."""
    media = [MediaItem(kind="image", data=b"i", mime="image/png")]
    content = build_user_content("x[image]y[image]z", media)
    assert [b["type"] for b in content] == ["text", "image_url", "text", "text", "text"]
    assert content[3]["text"] == "[image missing]"
