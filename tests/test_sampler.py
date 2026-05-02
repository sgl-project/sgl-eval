"""Sampler integration tests using a stub OpenAI client.

We don't hit a real server here (no fixtures for that). Instead we verify
the sampler builds the right request kwargs and unpacks responses into a
``Sample``. End-to-end against a live SGLang server is tested manually via
``sgl-eval ping --base-url ...``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from sgl_eval.runner import WorkerAborted
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import GenConfig


def _stub_response(text: str, completion: int = 7, prompt: int = 11, reasoning: int | None = None):
    usage_kw = {"completion_tokens": completion, "prompt_tokens": prompt}
    if reasoning is not None:
        usage_kw["completion_tokens_details"] = SimpleNamespace(reasoning_tokens=reasoning)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(**usage_kw),
    )


@pytest.fixture
def sampler(monkeypatch):
    s = ChatCompletionSampler.__new__(ChatCompletionSampler)
    captured = {}

    def fake_create(**kwargs):
        captured["kwargs"] = kwargs
        return _stub_response("hello")

    s.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    s.model = "stub-model"
    s.max_retries = 1
    s._abort_event = threading.Event()
    s._captured = captured
    return s


def test_basic_call(sampler):
    out = sampler([{"role": "user", "content": "hi"}])
    assert out.text == "hello"
    assert out.completion_tokens == 7
    assert out.prompt_tokens == 11
    assert out.finish_reason == "stop"


def test_gen_config_propagates(sampler):
    gen = GenConfig(temperature=0.7, top_p=0.9, max_tokens=128, seed=42)
    sampler([{"role": "user", "content": "hi"}], gen)
    kw = sampler._captured["kwargs"]
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9
    assert kw["max_tokens"] == 128
    assert kw["seed"] == 42


def test_max_tokens_none_omitted(sampler):
    """When ``GenConfig.max_tokens is None`` (NS-aligned default), the
    sampler omits the kwarg so the server picks its own context cap."""
    sampler([{"role": "user", "content": "hi"}], GenConfig(max_tokens=None))
    kw = sampler._captured["kwargs"]
    assert "max_tokens" not in kw


def test_chat_template_kwargs_become_extra_body(sampler):
    gen = GenConfig(chat_template_kwargs={"thinking": True})
    sampler([{"role": "user", "content": "hi"}], gen)
    kw = sampler._captured["kwargs"]
    assert kw["extra_body"]["chat_template_kwargs"] == {"thinking": True}


def test_system_message_prepended(sampler):
    gen = GenConfig(system_message="you are a helper")
    sampler([{"role": "user", "content": "hi"}], gen)
    msgs = sampler._captured["kwargs"]["messages"]
    assert msgs[0] == {"role": "system", "content": "you are a helper"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_timestamps_recorded(sampler):
    """Sampler stamps generation_start_time / end_time on each call so the
    data_point fed into vendored MathMetrics has gen_seconds info."""
    out = sampler([{"role": "user", "content": "hi"}])
    assert out.generation_start_time is not None
    assert out.generation_end_time is not None
    assert out.generation_end_time >= out.generation_start_time


def test_reasoning_tokens_extracted(sampler):
    """When the response carries usage.completion_tokens_details.reasoning_tokens
    (reasoning models), it is mirrored onto Sample.reasoning_tokens."""
    sampler.client.chat.completions.create = lambda **_: _stub_response(
        "hi", completion=20, reasoning=15
    )
    out = sampler([{"role": "user", "content": "hi"}])
    assert out.completion_tokens == 20
    assert out.reasoning_tokens == 15


def test_reasoning_tokens_absent(sampler):
    """Plain models without reasoning split leave reasoning_tokens=None;
    NeMo-Skills' fallback in BaseMetrics.update treats it as zero."""
    out = sampler([{"role": "user", "content": "hi"}])
    assert out.reasoning_tokens is None


def test_aborted_property_reflects_event(sampler):
    """``sampler.aborted`` mirrors the internal event so callers (e.g. the
    CLI) don't need to share the ``threading.Event`` directly."""
    assert sampler.aborted is False
    sampler._abort_event.set()
    assert sampler.aborted is True


def test_abort_before_call_raises(sampler):
    """If ``abort_event`` is already set when ``__call__`` enters, the
    sampler raises immediately without hitting the network."""
    sampler._abort_event.set()
    called = {"n": 0}

    def fake_create(**_kwargs):
        called["n"] += 1
        return _stub_response("hi")

    sampler.client.chat.completions.create = fake_create
    with pytest.raises(WorkerAborted):
        sampler([{"role": "user", "content": "hi"}])
    assert called["n"] == 0


def test_abort_short_circuits_retry(sampler):
    """If a request fails AND abort fires before the next retry, the sampler
    bails with ``WorkerAborted`` instead of looping through ``max_retries``."""
    sampler.max_retries = 5
    calls = {"n": 0}

    def failing_create(**_kwargs):
        calls["n"] += 1
        # Mark abort *during* the first failure so the retry loop sees it.
        sampler._abort_event.set()
        raise RuntimeError("boom")

    sampler.client.chat.completions.create = failing_create
    with pytest.raises(WorkerAborted):
        sampler([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1  # no retries after abort


def test_bad_request_returns_empty_sample(sampler):
    import openai

    def fail(**kwargs):
        raise openai.BadRequestError(
            "bad", response=SimpleNamespace(status_code=400), body={"error": "x"}
        )

    sampler.client.chat.completions.create = fail
    out = sampler([{"role": "user", "content": "hi"}])
    assert out.text == ""
    assert out.finish_reason == "error"
