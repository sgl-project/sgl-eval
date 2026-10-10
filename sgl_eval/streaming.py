"""Per-attempt, flushed streaming artifacts; partial answers are never scored."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Set inside the runner worker, since ContextVars do not propagate into a
# ThreadPoolExecutor automatically. Reset after each sample, including errors.
sample_identity: ContextVar[dict[str, Any] | None] = ContextVar("sample_identity", default=None)


def read_stream(
    client: Any,
    kwargs: dict[str, Any],
    directory: Path,
    trial: int,
    check_abort: Callable[[], None],
) -> SimpleNamespace:
    """Collect one attempt for the existing sampler and flush its live artifacts.

    Only a finished stream returns a response; errors preserve partial files
    and propagate to the sampler's existing retry/abort handling.
    """
    request_id = uuid.uuid4().hex
    prefix = directory / request_id
    content, reasoning = [], []
    finish_reason = None
    usage = None
    with (
        prefix.with_suffix(".jsonl").open("x", encoding="utf-8") as events,
        prefix.with_suffix(".content.txt").open("x", encoding="utf-8") as answer,
        prefix.with_suffix(".reasoning.txt").open("x", encoding="utf-8") as thinking,
    ):

        def emit(kind, **data):
            events.write(
                json.dumps({"event": kind, "time": time.time(), **data}, ensure_ascii=False) + "\n"
            )
            events.flush()

        emit("start", sample=sample_identity.get(), attempt=trial + 1, request_id=request_id)
        try:
            with client.chat.completions.create(
                **kwargs, stream=True, stream_options={"include_usage": True}
            ) as stream:
                for chunk in stream:
                    check_abort()
                    if chunk.usage is not None:
                        usage = chunk.usage
                    for choice in chunk.choices:
                        if choice.index != 0:
                            continue
                        delta = choice.delta
                        text = getattr(delta, "content", None) or ""
                        thought = getattr(delta, "reasoning_content", None) or ""
                        if text:
                            content.append(text)
                            answer.write(text)
                            answer.flush()
                        if thought:
                            reasoning.append(thought)
                            thinking.write(thought)
                            thinking.flush()
                        if text or thought:
                            emit("delta", content=text, reasoning_content=thought)
                        if choice.finish_reason is not None:
                            finish_reason = choice.finish_reason
            check_abort()
            if finish_reason is None:
                raise RuntimeError(
                    "Stream ended without a finish_reason; partial response retained"
                )
            emit(
                "complete",
                finish_reason=finish_reason,
                usage=usage.model_dump() if usage is not None else None,
            )
        except BaseException as exc:
            emit("error", error_type=type(exc).__name__, error=str(exc))
            raise
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="".join(content), reasoning_content="".join(reasoning)
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )
