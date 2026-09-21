import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace as NS

from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.streaming import sample_identity
from sgl_eval.types import GenConfig


def chunk(text=None, reasoning=None, finish=None, usage=None):
    return NS(
        usage=usage,
        choices=(
            []
            if usage
            else [
                NS(
                    index=0,
                    delta=NS(content=text, reasoning_content=reasoning),
                    finish_reason=finish,
                )
            ]
        ),
    )


def make_sampler(tmp_path, chunks, retries=1):
    sampler = ChatCompletionSampler.__new__(ChatCompletionSampler)
    sampler.stream_dir = tmp_path
    sampler.model = "test"
    sampler.max_retries = retries
    sampler._abort_event = threading.Event()
    calls = []

    @contextmanager
    def create(**kwargs):
        calls.append(kwargs)
        yield iter(chunks(len(calls)))

    sampler.client = NS(chat=NS(completions=NS(create=create)))
    return sampler, calls


def test_live_flush_and_usage(tmp_path):
    usage = NS(
        completion_tokens=8,
        prompt_tokens=4,
        completion_tokens_details=NS(reasoning_tokens=3),
        model_dump=lambda: {"completion_tokens": 8, "prompt_tokens": 4},
    )

    def chunks(_):
        yield chunk(reasoning="思考")
        assert next(tmp_path.glob("*.reasoning.txt")).read_text() == "思考"
        yield chunk(text="answer ")
        assert next(tmp_path.glob("*.content.txt")).read_text() == "answer "
        yield chunk(text="42", finish="stop")
        yield chunk(usage=usage)

    sampler, calls = make_sampler(tmp_path, chunks)
    token = sample_identity.set({"example_id": "aime-1", "repeat": 3})
    try:
        result = sampler([{"role": "user", "content": "test"}], GenConfig())
    finally:
        sample_identity.reset(token)
    assert result.text == "answer 42"
    assert result.reasoning_tokens == 3 and result.completion_tokens == 8
    assert result.finish_reason == "stop"
    assert calls[0]["stream"] and "max_tokens" not in calls[0]
    events = [json.loads(x) for x in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert events[0]["sample"]["repeat"] == 3
    assert events[-1]["event"] == "complete"


def test_disconnect_retains_partial_but_retry_scores_only_complete(tmp_path, monkeypatch):
    def chunks(attempt):
        yield chunk(text="partial" if attempt == 1 else "final")
        if attempt == 1:
            raise ConnectionError("disconnected")
        yield chunk(finish="length")

    sampler, calls = make_sampler(tmp_path, chunks, retries=2)
    monkeypatch.setattr(sampler._abort_event, "wait", lambda _: False)
    result = sampler([{"role": "user", "content": "test"}])
    assert result.text == "final" and result.finish_reason == "length"
    assert len(calls) == 2
    assert sorted(p.read_text() for p in tmp_path.glob("*.content.txt")) == ["final", "partial"]
    endings = [
        json.loads(p.read_text().splitlines()[-1])["event"] for p in tmp_path.glob("*.jsonl")
    ]
    assert sorted(endings) == ["complete", "error"]


def test_missing_finish_is_error_not_partial_answer(tmp_path, monkeypatch):
    sampler, _ = make_sampler(tmp_path, lambda _: [chunk(text="unfinished")])
    monkeypatch.setattr(sampler._abort_event, "wait", lambda _: False)
    result = sampler([{"role": "user", "content": "test"}])
    assert result.finish_reason == "error" and result.text == ""
    assert next(tmp_path.glob("*.content.txt")).read_text() == "unfinished"


def test_concurrent_attempt_files_and_identity(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from sgl_eval.streaming import read_stream

    @contextmanager
    def create(**kwargs):
        yield iter([chunk(text=kwargs["model"], finish="stop")])

    client = NS(chat=NS(completions=NS(create=create)))

    def run(i):
        token = sample_identity.set({"example_id": str(i), "repeat": i % 4})
        try:
            return read_stream(client, {"model": str(i)}, tmp_path, 0, lambda: None)
        finally:
            sample_identity.reset(token)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(32)))
    assert [r.choices[0].message.content for r in results] == [str(i) for i in range(32)]
    for path in tmp_path.glob("*.jsonl"):
        event = json.loads(path.read_text().splitlines()[0])
        assert path.with_suffix(".content.txt").read_text() == event["sample"]["example_id"]
    assert len(list(tmp_path.glob("*.jsonl"))) == 32


def test_missing_usage_stays_unknown(tmp_path):
    sampler, _ = make_sampler(tmp_path, lambda _: [chunk(text="answer", finish="stop")])
    result = sampler([{"role": "user", "content": "test"}])
    assert result.completion_tokens is None
    assert result.prompt_tokens is None
    assert result.reasoning_tokens is None


def test_abort_after_last_chunk_preserves_partial_without_retry(tmp_path):
    import pytest

    from sgl_eval.runner import WorkerAborted

    def chunks(_):
        yield chunk(text="partial", finish="stop")
        sampler._abort_event.set()

    sampler, calls = make_sampler(tmp_path, chunks, retries=3)
    with pytest.raises(WorkerAborted):
        sampler([{"role": "user", "content": "test"}])
    assert len(calls) == 1
    assert next(tmp_path.glob("*.content.txt")).read_text() == "partial"
    event = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[-1])
    assert event["event"] == "error"
    assert event["error_type"] == "WorkerAborted"


def test_streaming_preserves_generation_kwargs(tmp_path):
    sampler, calls = make_sampler(tmp_path, lambda _: [chunk(text="42", finish="stop")])
    messages = [{"role": "user", "content": "test"}]
    gen = GenConfig(
        temperature=0.8,
        top_p=0.95,
        max_tokens=123,
        seed=42,
        chat_template_kwargs={"thinking": True},
    )
    expected = sampler._build_kwargs(messages, gen)
    sampler(messages, gen)
    assert calls[0] == {**expected, "stream": True, "stream_options": {"include_usage": True}}


def test_runner_sets_and_restores_identity_in_both_paths():
    from sgl_eval.runner import run_examples
    from sgl_eval.types import Example, Sample

    for workers in (1, 4):

        def sample(ex, rep):
            assert sample_identity.get() == {"example_id": ex.id, "repeat": rep}
            return Sample(text="ok", finish_reason="stop")

        sentinel = {"example_id": "outer", "repeat": -1}
        token = sample_identity.set(sentinel)
        try:
            result = run_examples(
                "test",
                [Example(id=str(i), inputs={}, target="ok") for i in range(4)],
                sample,
                lambda ex, out: (1.0, out.text),
                num_threads=workers,
                n_repeats=3,
                progress=False,
            )
            assert result.aggregate["score"] == 1.0
            assert sample_identity.get() is sentinel
        finally:
            sample_identity.reset(token)
