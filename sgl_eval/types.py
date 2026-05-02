"""Core data types shared across sampler, runner, and benchmark implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

Message = Dict[str, Any]
MessageList = List[Message]


@dataclass
class GenConfig:
    """Per-call generation parameters. Decoupled from the sampler so one
    sampler instance can serve multiple benchmarks with different configs.

    Defaults mirror NeMo-Skills' ``InferenceConfig`` (``temperature=0.0``,
    ``top_p=0.95``, ``max_tokens=None`` => server picks a cap). Per-benchmark
    overrides live in ``sgl_eval/evals/_registry.py``; CLI overrides live in
    ``sgl_eval/cli.py``.
    """

    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    extra_body: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    system_message: Optional[str] = None


@dataclass
class Sample:
    """One model response. Carries enough metadata for throughput,
    truncation, reasoning-token, and gen-time analysis without re-fetching
    the raw OpenAI response. Field names mirror upstream NeMo-Skills'
    prediction dict so the data feeds into vendored metrics directly."""

    text: str
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    generation_start_time: Optional[float] = None
    generation_end_time: Optional[float] = None
    raw: Any = None


@dataclass
class Example:
    """One benchmark sample as loaded from its dataset."""

    id: str
    inputs: Dict[str, Any]
    target: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExampleResult:
    """All n_repeats samples for one Example, plus per-sample scores."""

    example: Example
    samples: List[Sample]
    scores: List[float]
    extracted: List[Optional[str]]


@dataclass
class RunResult:
    """Top-level eval result. Aggregator metrics live in ``aggregate``.

    ``partial`` is True when at least one ``(example, repeat)`` sample
    didn't make it into ``per_example`` (e.g. the runner was aborted
    mid-flight). Defined at the sample level so an example whose 1/3 reps
    completed -- which the aggregator pads up to 3 by repeating the last
    sample -- still surfaces as partial.

    ``planned_examples`` is what the runner was asked to score; multiply
    by ``n_repeats`` for planned samples. ``num_examples`` is what
    survived (>=1 rep completed); ``sum(len(r.samples) for r in per_example)``
    gives completed samples.
    """

    name: str
    per_example: List[ExampleResult]
    aggregate: Dict[str, float]
    latency: float
    num_examples: int
    n_repeats: int
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    partial: bool = False
    planned_examples: int = 0

    @property
    def output_throughput(self) -> float:
        if self.latency <= 0:
            return 0.0
        return self.total_completion_tokens / self.latency
