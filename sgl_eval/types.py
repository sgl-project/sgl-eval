"""Core data types shared across sampler, runner, and benchmark implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

Message = Dict[str, Any]
MessageList = List[Message]


@dataclass
class GenConfig:
    """Per-call generation parameters with NeMo-Skills InferenceConfig defaults.

    Benchmark defaults live in evals/_registry.py; preset.py resolves CLI
    overrides. max_tokens=None leaves the output limit to the server.
    """

    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    # Sent on every request, like NS does. Left unsent they would resolve
    # from the served model's generation_config.json instead.
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    extra_body: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    system_message: Optional[str] = None


@dataclass
class Sample:
    """One response with timing and token counts for the NeMo-Skills prediction adapter."""

    text: str
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    generation_start_time: Optional[float] = None
    generation_end_time: Optional[float] = None
    raw: Any = None
    reasoning_content: Optional[str] = None


@dataclass
class MediaItem:
    """One input media attachment (image/video). ``kind`` selects the message
    block; ``data`` (raw bytes) is used for images via base64 data URL, ``url``
    for video (too large to inline) or pre-hosted images."""

    kind: str  # "image" | "video"
    data: bytes = b""
    url: str = ""
    mime: str = ""  # "image/png" | "video/mp4" | ...


@dataclass
class Example:
    """One benchmark sample as loaded from its dataset."""

    id: str
    inputs: Dict[str, Any]
    target: Any
    meta: Dict[str, Any] = field(default_factory=dict)
    media: List[MediaItem] = field(default_factory=list)


@dataclass
class ExampleResult:
    """All n_repeats samples for one Example, plus per-sample scores."""

    example: Example
    samples: List[Sample]
    scores: List[float]
    extracted: List[Optional[str]]


@dataclass
class RunResult:
    """Evaluation results and aggregate metrics.

    partial tracks missing samples, including incomplete repeats of a retained
    example. planned_examples counts requested examples; num_examples counts
    those with at least one completed sample.
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
    # Benchmark-owned provenance and display rows, written to metrics.json as
    # ``metadata`` by the shared report; must not shadow core fields.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For benchmarks whose sample is a whole agent trajectory: token stats per
    # model response, kept apart from the per-sample totals above.
    response_usage: Optional[Dict[str, Any]] = None

    @property
    def output_throughput(self) -> float:
        if self.latency <= 0:
            return 0.0
        return self.total_completion_tokens / self.latency
