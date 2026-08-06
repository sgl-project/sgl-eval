"""RULER2 glue tests. No generation and no server: everything here is either
a vendored-consistency assertion or a scoring path driven by hand-built rows.

The dataset itself cannot be built in CI (it needs a tokenizer, HF downloads,
and minutes per subtask), so the invariants that would otherwise only surface
at runtime are asserted statically instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sgl_eval.evals._ruler2 import (
    ALL_TASKS,
    PRED_SCHEMA,
    Ruler2Config,
    _grader_for,
    _headline,
    _load_task,
    _make_score_one_fn,
    _task_metrics,
)
from sgl_eval.types import Example, ExampleResult, GenConfig, Sample

# --- vendored consistency -------------------------------------------------


def test_all_tasks_matches_vendored_compute_score():
    """``compute_score`` KeyErrors on a task it does not know, and silently
    under-averages if we omit one. Pin our list against its source."""
    import inspect

    from sgl_eval._vendored.nemo_skills.dataset.ruler2 import ruler2_score

    src = inspect.getsource(ruler2_score.compute_score)
    for task in ALL_TASKS:
        assert f'"{task}"' in src, f"{task} missing from vendored compute_score"
    # And the reverse: no task in its list that we do not run.
    listed = [line.strip().strip('",') for line in src.splitlines() if line.strip().startswith('"')]
    assert set(listed) == set(ALL_TASKS)


def test_every_task_has_a_vendored_prepare_fn():
    from sgl_eval._vendored.nemo_skills.dataset.ruler2 import prepare

    for task in ALL_TASKS:
        assert callable(getattr(prepare, f"prepare_{task}", None)), task


def test_ruler2_registered_with_lowered_concurrency():
    from sgl_eval.registry import get

    spec = get("ruler2")
    assert spec.category == "ruler2"
    # 64 concurrent 128k prompts would be ~8M tokens in flight.
    assert spec.default_num_threads < 64
    assert spec.pred_schema == PRED_SCHEMA


@pytest.mark.parametrize(
    "task, eval_type, match_type",
    [
        ("mk_niah_basic", "ruler2", "all"),
        ("mk_niah_medium", "multichoice", ""),
        ("mk_niah_hard", "multichoice", ""),
        ("mv_niah_medium", "ruler2", "2steps"),
        ("qa_hard", "ruler2", "part"),
    ],
)
def test_grader_read_back_from_vendored_prepare(tmp_path, monkeypatch, task, eval_type, match_type):
    """The grader mapping lives in vendored ``prepare_task_for_ns``; this
    checks we read it rather than having drifted from a local copy."""
    import sgl_eval.evals._ruler2 as mod

    monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_path)
    cfg = Ruler2Config(max_seq_length=4096, tokenizer_path="dummy/tok")
    assert _grader_for(cfg, task) == (eval_type, match_type)


@pytest.mark.parametrize("task", ["mk_niah_medium", "mk_niah_hard"])
def test_ruler2_metrics_matches_upstreams_multichoice_number(tmp_path, monkeypatch, task):
    """Upstream marks these two ``METRICS_TYPE=multichoice`` (i.e. MathMetrics)
    while we run all 12 subtasks through ``Ruler2Metrics``. The two publish the
    same value under different keys -- ``accuracy`` vs ``symbolic_correct`` --
    which is why vendored ``compute_score`` accepts either. Pin both halves so
    a future NS bump cannot turn this into a silent scoring difference."""
    import sgl_eval.evals._ruler2 as mod
    from sgl_eval._vendored.nemo_skills.dataset.ruler2 import prepare
    from sgl_eval._vendored.nemo_skills.math_metrics import MathMetrics
    from sgl_eval._vendored.nemo_skills.ruler2_metrics import Ruler2Metrics

    monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_path)
    prepare.prepare_task_for_ns(str(tmp_path), task)
    assert 'METRICS_TYPE = "multichoice"' in (tmp_path / task / "__init__.py").read_text()

    rows = [
        [{"symbolic_correct": correct, "predicted_answer": "A", "expected_answer": "A"}]
        for correct in (True, False, True, True, False, True, True)
    ]
    ours, upstreams = Ruler2Metrics(), MathMetrics()
    for row in rows:
        ours.update(row)
        upstreams.update(row)
    assert (
        ours.get_metrics()["pass@1"]["accuracy"]
        == upstreams.get_metrics()["pass@1"]["symbolic_correct"]
    )


# --- the stringified-answer regression ------------------------------------


def test_pred_schema_keeps_list_expected_answer():
    """Stringifying a list-valued answer makes ``eval_ruler2`` walk the repr
    character by character, scoring nearly everything correct."""
    from sgl_eval.predictions import sample_to_pred

    ex = Example(id="x", inputs={"question": "q"}, target=["alpha", "beta"])
    pred = sample_to_pred(Sample(text="out"), ex, PRED_SCHEMA)
    assert pred["expected_answer"] == ["alpha", "beta"]
    # And the default schema still stringifies, for math/mcq.
    assert sample_to_pred(Sample(text="out"), ex)["expected_answer"] == "['alpha', 'beta']"


def test_pred_schema_omits_the_prompt():
    """RULER2 prompts are ~500KB; echoing them into output-rs*.jsonl would
    write hundreds of MB per repeat."""
    from sgl_eval.predictions import sample_to_pred

    ex = Example(id="x", inputs={"problem": "P" * 1000}, target=["a"])
    assert sample_to_pred(Sample(text="o"), ex, PRED_SCHEMA)["problem"] == ""


def test_stringified_target_would_destroy_the_signal():
    """Quantifies why PRED_SCHEMA exists. With the default (stringifying)
    schema, ``eval_ruler2`` iterates the repr's characters, so a wrong answer
    and the right one land on the SAME score -- the grader stops discriminating
    while still reporting a plausible-looking number."""
    from sgl_eval._vendored.nemo_skills.evaluator.ruler import eval_ruler2
    from sgl_eval.evals._ruler2 import _score_via
    from sgl_eval.predictions import PredSchema, sample_to_pred

    ex = Example(id="x", inputs={"question": "q"}, target=["4656572173"])
    right = "The special magic number is 4656572173."
    wrong = "The special magic number for the key is 1234567890."

    def score(schema, gen):
        row = {"generation": gen, **sample_to_pred(Sample(text=gen), ex, schema)}
        return _score_via(lambda c: eval_ruler2({**c, "match_type": "all"}), row)["is_correct"]

    assert score(PRED_SCHEMA, right) == pytest.approx(1.0)
    assert score(PRED_SCHEMA, wrong) == pytest.approx(0.0)
    # Stringified: identical scores for a correct and an incorrect answer.
    assert score(PredSchema(), right) == pytest.approx(score(PredSchema(), wrong))


def test_score_one_ruler2_soft_match_is_not_character_wise():
    """A wrong answer must score 0, not ~1 via per-character substring hits."""
    score_one = _make_score_one_fn("ruler2", "all")
    ex = Example(id="x", inputs={"question": "q"}, target=["alpha", "beta"])

    hit, _ = score_one(ex, Sample(text="the answers are alpha and beta"))
    miss, _ = score_one(ex, Sample(text="zzzzzz"))
    assert hit == pytest.approx(1.0)
    assert miss < 0.5


def test_score_one_ruler2_partial_credit():
    """``match_type=all`` averages over the reference list."""
    score_one = _make_score_one_fn("ruler2", "all")
    ex = Example(id="x", inputs={"question": "q"}, target=["alpha", "beta"])
    half, _ = score_one(ex, Sample(text="only alpha here"))
    assert 0.0 < half < 1.0


def test_score_one_multichoice_path():
    score_one = _make_score_one_fn("multichoice", "")
    ex = Example(id="x", inputs={"question": "q"}, target="B")
    good, letter = score_one(ex, Sample(text="The final answer is \\boxed{B}"))
    bad, _ = score_one(ex, Sample(text="The final answer is \\boxed{C}"))
    assert good == 1.0 and letter == "B"
    assert bad == 0.0


# --- loader ---------------------------------------------------------------


def test_load_task_reads_question_field(tmp_path):
    """Rows use ``question``, not ``problem`` -- the field the generic loader
    hardcodes."""
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps({"index": 7, "question": "long ctx", "expected_answer": ["a"], "length": 4000})
        + "\n"
    )
    (ex,) = _load_task(path, "qa_basic", None)
    assert ex.inputs["question"] == "long ctx"
    assert ex.target == ["a"]
    assert ex.id == "qa_basic-7"
    assert ex.meta["task"] == "qa_basic"


def test_load_task_respects_num_examples(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        "".join(
            json.dumps({"index": i, "question": f"q{i}", "expected_answer": ["a"]}) + "\n"
            for i in range(5)
        )
    )
    assert len(_load_task(path, "qa_basic", 2)) == 2


# --- aggregation ----------------------------------------------------------


def _results(scores):
    out = []
    for i, s in enumerate(scores):
        ex = Example(id=f"e{i}", inputs={"question": "q"}, target=["a"])
        out.append(
            ExampleResult(example=ex, samples=[Sample(text="t")], scores=[s], extracted=["t"])
        )
    return out


def test_headline_full_group_uses_vendored_compute_score():
    """All 12 present -> the average comes from vendored ``compute_score``."""
    per_task = {task: _task_metrics(_results([1.0, 0.0]), 1, "ruler2") for task in ALL_TASKS}
    flat = _headline(per_task, 1, namespace="setup")
    assert flat["score"] == pytest.approx(0.5)
    assert len({k for k in flat if k.startswith("task.")}) == 12
    assert "partial_group" not in flat


def test_headline_subset_is_flagged():
    """A task subset cannot use ``compute_score`` (it KeyErrors on a partial
    group), so the local average has to announce itself."""
    per_task = {
        "qa_basic": _task_metrics(_results([1.0]), 1, "ruler2"),
        "qa_easy": _task_metrics(_results([0.0]), 1, "ruler2"),
    }
    flat = _headline(per_task, 1, namespace="setup")
    assert flat["score"] == pytest.approx(0.5)
    assert flat["partial_group"] == 1.0


@pytest.mark.parametrize("completed", [11, 5, 1, 0])
def test_headline_survives_an_aborted_full_group(completed):
    """Ctrl-C during a full-group run leaves fewer subtasks than were asked
    for. Gating on the requested set sent this into vendored ``compute_score``,
    which raised (KeyError on the first missing task, IndexError when nothing
    finished) and took the whole partial-metrics dump with it."""
    per_task = {t: _task_metrics(_results([1.0]), 1, "ruler2") for t in ALL_TASKS[:completed]}
    flat = _headline(per_task, 1, namespace="setup")
    assert flat["score"] == pytest.approx(1.0 if completed else 0.0)
    assert flat["partial_group"] == 1.0
    assert len({k for k in flat if k.startswith("task.")}) == completed


def test_task_metrics_accepts_float_scores():
    """RULER2 scores are floats; ``Ruler2Metrics`` must not binarize them."""
    raw = _task_metrics(_results([0.25, 0.75]), 1, "ruler2")
    assert raw["pass@1"]["accuracy"] == pytest.approx(50.0)


# --- config ---------------------------------------------------------------


def test_seq_len_reaches_prepare_unchanged(tmp_path, monkeypatch):
    """NS口径: ``seq_len`` is the dataset definition and must be handed to the
    vendored generator verbatim. sgl-eval adds no knob that shrinks it -- doing
    so would make our numbers incomparable with any published RULER2 result."""
    import sgl_eval.evals._ruler2 as mod

    monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_path)
    seen = {}

    def fake_prepare(out_folder, tok_type, tok_path, seq_len, size):
        seen.update(seq_len=seq_len, size=size, tok_type=tok_type, tok_path=tok_path)
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        (Path(out_folder) / "test.jsonl").write_text(
            json.dumps({"index": 0, "question": "q", "expected_answer": ["a"]}) + "\n"
        )

    monkeypatch.setattr(mod._prepare, "prepare_qa_basic", fake_prepare)
    cfg = Ruler2Config(max_seq_length=131072, tokenizer_path="some/tok", dataset_size=100)
    mod._ensure_task_data(cfg, "qa_basic")
    assert seen == {
        "seq_len": 131072,
        "size": 100,
        "tok_type": "hf",
        "tok_path": "some/tok",
    }


def test_setup_slug_separates_every_generation_input():
    base = Ruler2Config(max_seq_length=4096, tokenizer_path="a/b")
    variants = [
        Ruler2Config(max_seq_length=8192, tokenizer_path="a/b"),
        Ruler2Config(max_seq_length=4096, tokenizer_path="c/d"),
        Ruler2Config(max_seq_length=4096, tokenizer_path="a/b", dataset_size=50),
    ]
    slugs = {base.setup_slug} | {v.setup_slug for v in variants}
    assert len(slugs) == 4
    assert "/" not in base.setup_slug


def test_from_bench_args_requires_seq_len():
    with pytest.raises(SystemExit):
        Ruler2Config.from_bench_args({}, model="m")


def test_from_bench_args_defaults_tokenizer_to_model():
    cfg = Ruler2Config.from_bench_args({"seq_len": "8192"}, model="Qwen/Qwen3-8B")
    assert cfg.tokenizer_path == "Qwen/Qwen3-8B"
    assert cfg.tasks == ALL_TASKS


def test_from_bench_args_rejects_unknown_keys():
    with pytest.raises(SystemExit):
        Ruler2Config.from_bench_args({"seq_len": "8192", "nope": "1"}, model="m")


def test_from_bench_args_rejects_unknown_task():
    with pytest.raises(SystemExit):
        Ruler2Config.from_bench_args({"seq_len": "8192", "tasks": "qa_basic,bogus"}, model="m")


def test_from_bench_args_rejects_gemini_tokenizer():
    """``GeminiTokenizer`` is dropped from the vendored slice, so selecting it
    would NameError deep inside generation."""
    with pytest.raises(SystemExit):
        Ruler2Config.from_bench_args({"seq_len": "8192", "tokenizer_type": "gemini"}, model="m")


def test_from_bench_args_rejects_headroom_knob():
    """``headroom`` was removed on purpose: it shrank the dataset and made our
    scores incomparable with NeMo-Skills. Rejecting it loudly beats silently
    ignoring a flag someone copied from an older invocation."""
    with pytest.raises(SystemExit):
        Ruler2Config.from_bench_args({"seq_len": "8192", "headroom": "512"}, model="m")


# --- preflight ------------------------------------------------------------


class _StubClient:
    def __init__(self, url):
        self.base_url = url


class _StubSampler:
    model = "m"

    def __init__(self, url="http://host:30000/v1"):
        self.client = _StubClient(url)


def _stub_model_info(monkeypatch, payload):
    import httpx

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return payload

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())


def test_preflight_requires_room_for_the_answer(monkeypatch, capsys):
    """A window exactly equal to seq_len leaves nothing to generate when
    max_tokens is None -- every sample comes back empty and scores 0."""
    from sgl_eval.evals._ruler2 import _preflight_context_length

    _stub_model_info(monkeypatch, {"max_context_length": 131072})
    with pytest.raises(SystemExit):
        _preflight_context_length(
            _StubSampler(),
            Ruler2Config(max_seq_length=131072, tokenizer_path="t"),
            GenConfig(),
        )


def test_preflight_passes_with_headroom_in_the_window(monkeypatch, capsys):
    from sgl_eval.evals._ruler2 import _preflight_context_length

    _stub_model_info(monkeypatch, {"max_context_length": 262144})
    _preflight_context_length(
        _StubSampler(),
        Ruler2Config(max_seq_length=131072, tokenizer_path="t"),
        GenConfig(),
    )
    assert "Preflight" in capsys.readouterr().out


def test_preflight_uses_explicit_max_tokens_as_the_budget(monkeypatch):
    """With --max-tokens given, that is the room the window must have."""
    from sgl_eval.evals._ruler2 import _preflight_context_length

    _stub_model_info(monkeypatch, {"max_context_length": 5000})
    cfg = Ruler2Config(max_seq_length=4096, tokenizer_path="t")
    # 4096 + 512 default budget fits in 5000.
    _preflight_context_length(_StubSampler(), cfg, GenConfig())
    # 4096 + 2048 does not.
    with pytest.raises(SystemExit):
        _preflight_context_length(_StubSampler(), cfg, GenConfig(max_tokens=2048))


def test_preflight_warns_when_limit_unreadable(monkeypatch, capsys):
    """Refusing to run against an endpoint we cannot introspect would be worse
    than warning."""
    from sgl_eval.evals._ruler2 import _preflight_context_length

    _stub_model_info(monkeypatch, {})
    _preflight_context_length(
        _StubSampler(), Ruler2Config(max_seq_length=4096, tokenizer_path="t"), GenConfig()
    )
    assert "WARNING" in capsys.readouterr().err


# --- refresh --------------------------------------------------------------


def _ruler2_row(task: str, index: int, score: float) -> dict:
    return {
        "generation": "g",
        "id": f"{task}-{index}",
        "expected_answer": ["a"],
        "num_generated_tokens": 5,
        "num_reasoning_tokens": 0,
        "num_answer_tokens": 5,
        "finish_reason": "stop",
        "problem": "",
        "predicted_answer": "g",
        "is_correct": score,
    }


def test_refresh_rebuilds_the_group_headline(tmp_path):
    """``sgl-eval refresh`` read a hardcoded ``symbolic_correct`` and had no
    group branch, so a ruler2 run refreshed to 0.00% -- overwriting the real
    metrics.json. Uneven subtask counts here separate the two failures: the
    sample-level mean is 0.25, the mean-of-subtask-means is 0.50.
    """
    from types import SimpleNamespace

    from sgl_eval.pipeline.refresh import cmd_refresh

    run_dir = tmp_path / "sgl_eval_ruler2_20260731-120000"
    run_dir.mkdir()
    rows = [_ruler2_row("qa_basic", 0, 1.0)] + [_ruler2_row("qa_easy", i, 0.0) for i in range(3)]
    with (run_dir / "output-rs0.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    assert cmd_refresh(SimpleNamespace(run_dir=str(run_dir))) == 0

    agg = json.loads((run_dir / "metrics.json").read_text())["aggregate"]
    assert agg["score"] == pytest.approx(0.5)
    assert agg["task.qa_basic"] == pytest.approx(1.0)
    assert agg["task.qa_easy"] == pytest.approx(0.0)
    assert agg["partial_group"] == 1.0


def test_refresh_rejects_unattributable_ids(tmp_path):
    """Ids that do not name a subtask cannot be grouped; guessing a headline
    from them would report a plausible number for the wrong denominator."""
    from types import SimpleNamespace

    from sgl_eval.pipeline.refresh import cmd_refresh

    run_dir = tmp_path / "sgl_eval_ruler2_20260731-120000"
    run_dir.mkdir()
    with (run_dir / "output-rs0.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(_ruler2_row("not_a_task", 0, 1.0)) + "\n")

    with pytest.raises(SystemExit):
        cmd_refresh(SimpleNamespace(run_dir=str(run_dir)))
