"""Sample an OpenAI-compatible chat endpoint and retain response metadata."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None

import httpx
import openai
from openai import OpenAI

from sgl_eval.runner import WorkerAborted
from sgl_eval.streaming import read_stream
from sgl_eval.types import GenConfig, MessageList, Sample

LOG = logging.getLogger(__name__)


# Above any plausible --num-threads; httpx's default 100 would silently cap
# concurrency below what the runner was told to use.
_MAX_CONNECTIONS = 3600
_TARGET_NOFILE = 65_535


def _raise_nofile_soft_limit(target: int = _TARGET_NOFILE) -> None:
    """Leave room for concurrent HTTP sockets and asyncio selectors."""
    if resource is None:
        return
    try:
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired_soft = (
            target if current_hard == resource.RLIM_INFINITY else min(target, current_hard)
        )
        if current_soft == resource.RLIM_INFINITY or current_soft >= desired_soft:
            return
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired_soft, current_hard))
    except (OSError, ValueError) as exc:
        LOG.warning("Could not raise RLIMIT_NOFILE: %s", exc)
    else:
        LOG.info("Raised RLIMIT_NOFILE soft limit from %d to %d", current_soft, desired_soft)


class _LargeHttpxClient(httpx.Client):
    """Allow the four-hour read timeout used by NeMo-Skills InferenceConfig."""

    def __init__(self) -> None:
        timeout = httpx.Timeout(14400, connect=30)
        limits = httpx.Limits(
            max_keepalive_connections=_MAX_CONNECTIONS, max_connections=_MAX_CONNECTIONS
        )
        super().__init__(timeout=timeout, limits=limits)


class ChatCompletionSampler:
    """Wraps an OpenAI-compatible endpoint as a callable returning ``Sample``."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "EMPTY",
        max_retries: int = 6,
        stream_dir: Path | None = None,
    ) -> None:
        _raise_nofile_soft_limit()
        # Hold the httpx client directly so ``abort()`` can close it without
        # reaching into ``OpenAI``'s private ``_client`` attribute.
        self._http = _LargeHttpxClient()
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=self._http)
        self.model = model or self._resolve_default_model()
        self.max_retries = max_retries
        self.stream_dir = stream_dir
        if stream_dir is not None:
            stream_dir.mkdir(parents=True, exist_ok=True)
        self._abort_event = threading.Event()

    @property
    def aborted(self) -> bool:
        """Whether abort() was called; the CLI uses this to select its exit code."""
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
                if getattr(self, "stream_dir", None) is None:
                    response = self.client.chat.completions.create(**kwargs)
                else:

                    def check_abort():
                        if self.aborted:
                            raise WorkerAborted()

                    response = read_stream(self.client, kwargs, self.stream_dir, trial, check_abort)
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

        # Match NS by overriding model-specific generation_config.json defaults.
        # These extensions require an endpoint that accepts SGLang sampling fields.
        extra_body: Dict[str, Any] = {
            "min_p": gen.min_p,
            "repetition_penalty": gen.repetition_penalty,
        }
        if gen.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = gen.chat_template_kwargs
        if gen.extra_body:
            extra_body.update(gen.extra_body)
        kwargs["extra_body"] = extra_body
        return kwargs

    @staticmethod
    def _to_sample(
        response: Any, *, start: Optional[float] = None, end: Optional[float] = None
    ) -> Sample:
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None)
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None

        # OpenAI reports the split under completion_tokens_details. SGLang
        # historically exposed the same count directly on usage.
        reasoning_tokens = None
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
            if reasoning_tokens is None:
                reasoning_tokens = getattr(usage, "reasoning_tokens", None)

        return Sample(
            text=text,
            reasoning_content=reasoning_content,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            finish_reason=getattr(choice, "finish_reason", None),
            generation_start_time=start,
            generation_end_time=end,
            raw=response,
        )
