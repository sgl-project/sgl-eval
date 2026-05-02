"""OpenAI-compatible chat completion sampler.

Adapted from sgl-project/sglang/python/sglang/test/simple_eval_common.py with a
``Sample`` dataclass return value that exposes ``completion_tokens`` and
``finish_reason`` as first-class fields (cleaner than the upstream
side-channel list).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import httpx
import openai
from openai import OpenAI

from sgl_eval.runner import WorkerAborted
from sgl_eval.types import GenConfig, MessageList, Sample

LOG = logging.getLogger(__name__)


class _LargeHttpxClient(httpx.Client):
    """httpx client tuned for long-running reasoning generations."""

    def __init__(self) -> None:
        timeout = httpx.Timeout(3600)
        limits = httpx.Limits(max_keepalive_connections=3600, max_connections=3600)
        super().__init__(timeout=timeout, limits=limits)


class ChatCompletionSampler:
    """Wraps an OpenAI-compatible endpoint as a callable returning ``Sample``."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "EMPTY",
        max_retries: int = 6,
    ) -> None:
        # Hold the httpx client directly so ``abort()`` can close it without
        # reaching into ``OpenAI``'s private ``_client`` attribute.
        self._http = _LargeHttpxClient()
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=self._http)
        self.model = model or self._resolve_default_model()
        self.max_retries = max_retries
        self._abort_event = threading.Event()

    @property
    def aborted(self) -> bool:
        """True if ``abort()`` was called. CLI uses this to decide exit code
        without needing to share a ``threading.Event`` directly."""
        return self._abort_event.is_set()

    def abort(self) -> None:
        """Set the abort flag and close the underlying httpx client.

        In-flight ``chat.completions.create(...)`` calls raise immediately;
        the retry loop short-circuits on the abort flag and re-raises
        ``WorkerAborted``. Idempotent.
        """
        self._abort_event.set()
        try:
            self._http.close()
        except Exception:
            pass

    def _resolve_default_model(self) -> str:
        models = self.client.models.list().data
        if not models:
            raise RuntimeError("No models reported by the endpoint; pass `model=` explicitly.")
        return models[0].id

    @staticmethod
    def pack_message(role: str, content: Any) -> Dict[str, Any]:
        return {"role": role, "content": content}

    def __call__(self, messages: MessageList, gen: Optional[GenConfig] = None) -> Sample:
        gen = gen or GenConfig()
        if gen.system_message:
            messages = [self.pack_message("system", gen.system_message), *messages]

        kwargs = self._build_kwargs(messages, gen)

        for trial in range(self.max_retries):
            if self._abort_event.is_set():
                raise WorkerAborted()
            try:
                start = time.time()
                response = self.client.chat.completions.create(**kwargs)
                end = time.time()
                return self._to_sample(response, start=start, end=end)
            except openai.BadRequestError as e:
                LOG.warning("BadRequestError, returning empty sample: %s", e)
                return Sample(text="", finish_reason="error", raw=e)
            except Exception as e:
                if self._abort_event.is_set():
                    raise WorkerAborted() from e
                backoff = 2**trial
                LOG.warning(
                    "Sampler exception (trial %d/%d), backing off %ds: %s",
                    trial + 1,
                    self.max_retries,
                    backoff,
                    e,
                )
                # Wake immediately if abort fires during the backoff sleep.
                if self._abort_event.wait(backoff):
                    raise WorkerAborted() from e

        LOG.error("Sampler exhausted retries; returning empty sample.")
        return Sample(text="", finish_reason="error")

    def _build_kwargs(self, messages: MessageList, gen: GenConfig) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
        }
        if gen.max_tokens is not None:
            kwargs["max_tokens"] = gen.max_tokens
        if gen.reasoning_effort is not None:
            kwargs["reasoning_effort"] = gen.reasoning_effort
        if gen.seed is not None:
            kwargs["seed"] = gen.seed

        extra_body: Dict[str, Any] = {}
        if gen.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = gen.chat_template_kwargs
        if gen.extra_body:
            extra_body.update(gen.extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    @staticmethod
    def _to_sample(
        response: Any, *, start: Optional[float] = None, end: Optional[float] = None
    ) -> Sample:
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None

        # Reasoning models (e.g. OpenAI o-series, sglang thinking-mode)
        # report split via ``usage.completion_tokens_details.reasoning_tokens``.
        reasoning_tokens = None
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)

        return Sample(
            text=text,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            finish_reason=getattr(choice, "finish_reason", None),
            generation_start_time=start,
            generation_end_time=end,
            raw=response,
        )
