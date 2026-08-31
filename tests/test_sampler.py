"""Sampler integration tests using a stub OpenAI client.

We don't hit a real server here (no fixtures for that). Instead we verify
the sampler builds the right request kwargs and unpacks responses into a
``Sample``. End-to-end against a live SGLang server is tested manually via
``sgl-eval ping --base-url ...``.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import pytest

from sgl_eval import sampler as sampler_module
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


def test_nofile_soft_limit_is_raised(monkeypatch):
    calls = []
    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=lambda _: (1024, 1_048_576),
        setrlimit=lambda resource_type, limits: calls.append((resource_type, limits)),
    )
    monkeypatch.setattr(sampler_module, "resource", fake_resource)

    sampler_module._set_nofile_soft_limit()

    assert calls == [(7, (sampler_module._TARGET_NOFILE, 1_048_576))]


def test_nofile_soft_limit_is_capped_by_hard_limit(monkeypatch):
    calls = []
    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=lambda _: (1024, 4096),
        setrlimit=lambda resource_type, limits: calls.append((resource_type, limits)),
    )
    monkeypatch.setattr(sampler_module, "resource", fake_resource)

    sampler_module._set_nofile_soft_limit()

    assert calls == [(7, (4096, 4096))]


def test_nofile_soft_limit_handles_negative_infinity_sentinel(monkeypatch):
    calls = []
    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        RLIM_INFINITY=-1,
        getrlimit=lambda _: (1024, -1),
        setrlimit=lambda resource_type, limits: calls.append((resource_type, limits)),
    )
    monkeypatch.setattr(sampler_module, "resource", fake_resource)

    sampler_module._set_nofile_soft_limit()

    assert calls == [(7, (sampler_module._TARGET_NOFILE, -1))]


def test_nofile_soft_limit_is_unchanged_when_already_sufficient(monkeypatch):
    calls = []
    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=lambda _: (131072, 1_048_576),
        setrlimit=lambda resource_type, limits: calls.append((resource_type, limits)),
    )
    monkeypatch.setattr(sampler_module, "resource", fake_resource)

    sampler_module._set_nofile_soft_limit()

    assert calls == []


def test_nofile_soft_limit_is_a_noop_without_resource(monkeypatch):
    monkeypatch.setattr(sampler_module, "resource", None)

    sampler_module._set_nofile_soft_limit()


@pytest.mark.parametrize("operation", ["getrlimit", "setrlimit"])
def test_nofile_soft_limit_logs_and_continues_on_resource_error(monkeypatch, caplog, operation):
    def raise_oserror(*_args):
        raise OSError("not permitted")

    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=raise_oserror if operation == "getrlimit" else lambda _: (1024, 1_048_576),
        setrlimit=raise_oserror if operation == "setrlimit" else lambda *_: None,
    )
    monkeypatch.setattr(sampler_module, "resource", fake_resource)

    with caplog.at_level(logging.WARNING):
        sampler_module._set_nofile_soft_limit()

    assert "RLIMIT_NOFILE" in caplog.text
    assert "not permitted" in caplog.text


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


def test_min_p_and_repetition_penalty_always_sent(sampler):
    """Both are NS ``InferenceConfig`` fields. Left unsent, sglang resolves
    them from the served model's generation_config.json -- so a model shipping
    e.g. repetition_penalty=1.05 would silently decode differently than under
    an NS run, and the scores would not be comparable."""
    sampler([{"role": "user", "content": "hi"}], GenConfig())
    extra = sampler._captured["kwargs"]["extra_body"]
    assert extra["min_p"] == 0.0
    assert extra["repetition_penalty"] == 1.0


def test_explicit_extra_body_wins_over_the_ns_defaults(sampler):
    gen = GenConfig(extra_body={"repetition_penalty": 1.05})
    sampler([{"role": "user", "content": "hi"}], gen)
    assert sampler._captured["kwargs"]["extra_body"]["repetition_penalty"] == 1.05


def test_system_message_prepended(sampler):
    gen = GenConfig(system_message="you are a helper")
    sampler([{"role": "user", "content": "hi"}], gen)
    msgs = sampler._captured["kwargs"]["messages"]
    assert msgs[0] == {"role": "system", "content": "you are a helper"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_reasoning_tokens_extracted(sampler):
    """When the response carries usage.completion_tokens_details.reasoning_tokens
    (reasoning models), it is mirrored onto Sample.reasoning_tokens."""
    sampler.client.chat.completions.create = lambda **_: _stub_response(
        "hi", completion=20, reasoning=15
    )
    out = sampler([{"role": "user", "content": "hi"}])
    assert out.completion_tokens == 20
    assert out.reasoning_tokens == 15


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
