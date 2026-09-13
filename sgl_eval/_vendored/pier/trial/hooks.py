# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/trial/hooks.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable

from pydantic import BaseModel, Field

from sgl_eval._vendored.pier.models.trial.config import TrialConfig
from sgl_eval._vendored.pier.models.trial.result import TrialResult


class TrialEvent(Enum):
    """Events in a trial's lifecycle."""

    START = "start"
    ENVIRONMENT_START = "environment_start"
    AGENT_START = "agent_start"
    VERIFICATION_START = "verification_start"
    END = "end"
    CANCEL = "cancel"


class TrialHookEvent(BaseModel):
    """
    Unified event object passed to all trial lifecycle hooks.

    Provides context about the trial at the time of the event.
    The `result` field is only populated for END events.
    """

    model_config = {"arbitrary_types_allowed": True}

    event: TrialEvent
    trial_id: str
    task_name: str
    config: TrialConfig
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: TrialResult | None = None


HookCallback = Callable[["TrialHookEvent"], Awaitable[None]]
