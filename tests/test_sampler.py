"""Check request and response contracts with a stub OpenAI client; no server required."""

from __future__ import annotations

import http.server
import logging
import socket
import threading
from types import SimpleNamespace

import pytest

from sgl_eval import sampler as sampler_module
from sgl_eval.runner import WorkerAborted
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import GenConfig


def _stub_response(
    text: str,
    completion: int = 7,
    prompt: int = 11,
    reasoning: int | None = None,
    reasoning_content: str | None = None,
    sglang_reasoning: int | None = None,
):
    usage_kw = {"completion_tokens": completion, "prompt_tokens": prompt}
    if reasoning is not None:
        usage_kw["completion_tokens_details"] = SimpleNamespace(reasoning_tokens=reasoning)
    if sglang_reasoning is not None:
        usage_kw["reasoning_tokens"] = sglang_reasoning
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, reasoning_content=reasoning_content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(**usage_kw),
    )


@pytest.mark.parametrize(
    "soft, hard, expected",
    [
        pytest.param(1024, 1_048_576, 65_535, id="raise"),
        pytest.param(1024, 4096, 4096, id="hard-cap"),
        pytest.param(1024, -1, 65_535, id="unlimited-hard"),
        pytest.param(131072, 1_048_576, None, id="already-sufficient"),
    ],
)
def test_nofile_soft_limit(monkeypatch, soft, hard, expected):
    calls = []
    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        RLIM_INFINITY=-1,
        getrlimit=lambda _: (soft, hard),
        setrlimit=lambda resource_type, limits: calls.append((resource_type, limits)),
    )
    monkeypatch.setattr(sampler_module, "resource", fake_resource)

    sampler_module._raise_nofile_soft_limit()

    assert calls == ([] if expected is None else [(7, (expected, hard))])


def test_nofile_soft_limit_is_a_noop_without_resource(monkeypatch):
    monkeypatch.setattr(sampler_module, "resource", None)

    sampler_module._raise_nofile_soft_limit()


@pytest.mark.parametrize("operation", ["getrlimit", "setrlimit"])
def test_nofile_soft_limit_logs_and_continues_on_resource_error(monkeypatch, caplog, operation):
    def raise_oserror(*_args):
        raise OSError("not permitted")

    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        RLIM_INFINITY=-1,
        getrlimit=lambda _: (1024, 1_048_576),
        setrlimit=lambda *_: None,
    )
    setattr(fake_resource, operation, raise_oserror)
    monkeypatch.setattr(sampler_module, "resource", fake_resource)

    with caplog.at_level(logging.WARNING):
        sampler_module._raise_nofile_soft_limit()

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
    """Explicit NS sampling defaults must override model-specific server defaults."""
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


def test_sglang_reasoning_fields_extracted(sampler):
    """SGLang exposes reasoning text on the message and, in older releases,
    its token count directly on usage."""
    sampler.client.chat.completions.create = lambda **_: _stub_response(
        "answer",
        completion=20,
        reasoning_content="thinking",
        sglang_reasoning=15,
    )
    out = sampler([{"role": "user", "content": "hi"}])
    assert out.text == "answer"
    assert out.reasoning_content == "thinking"
    assert out.reasoning_tokens == 15


def test_standard_reasoning_token_count_takes_precedence(sampler):
    sampler.client.chat.completions.create = lambda **_: _stub_response(
        "answer", reasoning=12, sglang_reasoning=10
    )
    out = sampler([{"role": "user", "content": "hi"}])
    assert out.reasoning_tokens == 12


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


_PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _clear_proxy_env(monkeypatch):
    for name in _PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.fixture
def local_http_server():
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


def test_sampler_connections_enable_tcp_keepalive(monkeypatch, local_http_server):
    _clear_proxy_env(monkeypatch)
    expected = {
        getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None): 60,
        getattr(socket, "TCP_KEEPINTVL", None): 15,
        getattr(socket, "TCP_KEEPCNT", None): 4,
    }
    expected.pop(None, None)

    client = sampler_module._LargeHttpxClient()
    try:
        with client.stream("GET", local_http_server) as response:
            sock = response.extensions["network_stream"].get_extra_info("socket")
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
            for name, value in expected.items():
                assert sock.getsockopt(socket.IPPROTO_TCP, name) == value
    finally:
        client.close()


def test_env_proxy_transport_also_gets_keepalive_and_limits(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    built = []
    real_transport = sampler_module.httpx.HTTPTransport

    def spy(**kwargs):
        built.append(kwargs)
        return real_transport(**kwargs)

    monkeypatch.setattr(sampler_module.httpx, "HTTPTransport", spy)

    client = sampler_module._LargeHttpxClient()
    client.close()

    proxied = [kw for kw in built if kw.get("proxy") is not None]
    direct = [kw for kw in built if kw.get("proxy") is None]
    assert len(proxied) == 1 and len(direct) == 1
    for kwargs in built:
        assert kwargs["socket_options"] == sampler_module._keepalive_socket_options()
        assert kwargs["limits"].max_connections == sampler_module._MAX_CONNECTIONS


def test_keepalive_options_rejected_by_the_os_are_dropped(monkeypatch, caplog):
    rejected = getattr(socket, "TCP_KEEPCNT", None)
    if rejected is None:
        pytest.skip("TCP_KEEPCNT not exposed on this platform")

    class _FakeSocket:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def setsockopt(self, level, name, value):
            if name == rejected:
                raise OSError("Protocol not available")

    monkeypatch.setattr(sampler_module.socket, "socket", _FakeSocket)

    with caplog.at_level(logging.WARNING):
        options = sampler_module._keepalive_socket_options()

    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
    assert all(name != rejected for _level, name, _value in options)
    assert "not supported" in caplog.text
